"""
Transformer layer utilities in the Llama model
"""

import time
import torch
import torch.distributed as dist
try:
    import vllm_flash_attn_2_cuda as flash_attn_cuda
    _USE_VLLM_FLASH_ATTN = True
except ImportError:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    _USE_VLLM_FLASH_ATTN = False

# pylint: disable=no-name-in-module
from swiftllm_c import \
    fused_add_rmsnorm_inplace, \
    silu_and_mul_inplace, \
    rotary_embedding_inplace, \
    store_kvcache#, paged_attention
    # linear, \

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.block_swapper import Swapper
from swiftllm.structs import SubBatch

from swiftllm.worker.kernels.linear import linear
# from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
# from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
# from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace
# from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.prefill_attn import prefill_attention

class TransformerEvents:
    """
    Performance monitoring & synchronization events for a transformer layer
    """
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.stage_s = torch.cuda.Event(enable_timing=True)
        self.linr_e = torch.cuda.Event(enable_timing=True)
        self.pref_e = torch.cuda.Event(enable_timing=True)
        self.gdec_e = torch.cuda.Event(enable_timing=True)
        self.qkvtr_e = torch.cuda.Event()
        self.lnch_s = 0.0
        self.lnch_m = 0.0
        self.cdec_s = 0.0
        self.cdec_e = 0.0
        self.lnch_e = 0.0
    
    @property
    def linr_time(self) -> float:
        """
        Time for linear operations + other small operations other than attention
        """
        return self.stage_s.elapsed_time(self.linr_e)

    @property
    def pref_time(self) -> float:
        """
        Time for prefilling
        """
        return self.linr_e.elapsed_time(self.pref_e)

    @property
    def gdec_time(self) -> float:
        """
        Time for GPU decoding
        """
        return self.pref_e.elapsed_time(self.gdec_e)

    @property
    def cdec_time(self) -> float:
        """
        Time for CPU decoding
        """
        return self.cdec_e - self.cdec_s

    @property
    def lnch_time(self) -> float:
        """
        Time for kernel launch & other Python overheads other than CPU decoding
        """
        return self.lnch_e - self.cdec_e + self.lnch_m - self.lnch_s

    def pf_record(self, name: str):
        """
        Record the event if performance monitoring is enabled
        """
        if self.engine_config.monitor_performance:
            getattr(self, name).record()

    def pf_time(self, name: str):
        """
        Record the time if performance monitoring is enabled
        """
        if self.engine_config.monitor_performance:
            setattr(self, name, time.perf_counter() * 1e3) # ms

    def pf_time_nocpu(self):
        """
        If performance monitoring is enabled but there is no CPU decoding sequences, set CPU time stamps to the current time
        """
        if self.engine_config.monitor_performance:
            self.lnch_m = self.cdec_s = self.cdec_e = time.perf_counter()

class LlamaTransformerLayer:
    """
    A transformer layer in the Llama model, may contain weights of the next layer
    """
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        next_layer_weight: LlamaTransformerLayerWeight | None,
        cpu_communication_stream: torch.cuda.Stream,
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.next_layer_weight = next_layer_weight
        self.cpu_communication_stream = cpu_communication_stream
        self.layer_id = layer_id

        self.events = [TransformerEvents(engine_config) for _ in range(2)]

        # Set after KV cache initialization
        self.swapper = None


    def set_swapper(self, swapper: Swapper):
        """
        Set the swapper for the layer
        """
        self.swapper = swapper


    def _comm_wait_compute(self):
        """
        Do communication after necessary computation
        """
        self.cpu_communication_stream.wait_stream(torch.cuda.default_stream())


    def _compute_wait_comm(self):
        """
        Do computation after necessary communication
        """
        torch.cuda.default_stream().wait_stream(self.cpu_communication_stream)


    def _maybe_allreduce(self, x: torch.Tensor):
        if self.model_config.world_size > 1:
            dist.all_reduce(x)


    def _transfer_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: SubBatch,
        cur_stage: int = 0
    ):
        """
        Initiate transfer of QKV to CPU buffers
        """
        self._comm_wait_compute()
        if batch.num_cdecs > 0:
            with torch.cuda.stream(self.cpu_communication_stream):
                qc = self.swapper.q_cpu[:batch.num_cdecs]
                kc = self.swapper.k_cpu[:batch.num_cdecs]
                vc = self.swapper.v_cpu[:batch.num_cdecs]
                qc.copy_(q[-batch.num_cdecs:], non_blocking=True)
                kc.copy_(k[-batch.num_cdecs:], non_blocking=True)
                vc.copy_(v[-batch.num_cdecs:], non_blocking=True)
                self.events[cur_stage].qkvtr_e.record()


    def _swap_out_blocks(
        self,
        batch: SubBatch,
    ):
        """
        Swap blocks from GPU to CPU, assume that new prefilled KVs are ready in the last stage
        """
        if batch.num_cprfs > 0:
            with torch.cuda.stream(self.cpu_communication_stream):
                self.swapper.swap_blocks(
                    batch.src_blk_ids, 
                    batch.dst_blk_ids, 
                    is_swap_out=True, 
                    gpu_layer=self.model_config.num_layers if self.engine_config.extra_layer_for_cprf else self.layer_id,
                    cpu_layer=self.layer_id
                )


    def _preproj(
        self,
        embeddings: torch.Tensor,
        batch: SubBatch,
        layer_off: int = 0
    ) -> tuple[torch.Tensor]:
        """
        Perform pre-projection, including RMSNorm, QKV calculation, and rotary embedding
        """
        weight = self.weight if not layer_off else self.next_layer_weight

        self._maybe_allreduce(embeddings)
        fused_add_rmsnorm_inplace(
            embeddings,
            batch.residual_buf,
            weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        # Calculate QKV
        q = linear(embeddings, weight.q_proj)		# [iter_width, hidden_size / ws]
        k = linear(embeddings, weight.k_proj)		# [iter_width, num_kv_heads / ws * head_dim]
        v = linear(embeddings, weight.v_proj)		# [iter_width, num_kv_heads / ws * head_dim]
        q = q.view(batch.iter_width, -1, self.model_config.head_dim)
        k = k.view(batch.iter_width, -1, self.model_config.head_dim)
        v = v.view(batch.iter_width, -1, self.model_config.head_dim)

        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            batch.position_sin,
            batch.position_cos
        )

        # Here we only store k, v for prefilling, the kernel won't store decoding KVs
        if batch.num_prefs > 0 and self.swapper is not None:
            gpu_layer = (self.layer_id + layer_off) % self.model_config.num_layers
            itm_layer = self.model_config.num_layers if self.engine_config.extra_layer_for_cprf else gpu_layer
            # NOTE: if swapping is too slow, there's risk that we writes to what's being swapped out
            self._compute_wait_comm()
            store_kvcache(
                k,
                v,
                self.swapper.k_cache, 
                self.swapper.v_cache,
                self.swapper.gpu_block_table,
                batch.prgd_seq_ids[:batch.num_prefs],
                batch.pref_st_locs_we,
                batch.prgd_seq_lens[:batch.num_prefs],
                itm_layer,
                gpu_layer,
                batch.num_cprfs,
                batch.max_pref_toks
            )

        return q, k, v


    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: SubBatch,
        cur_stage: int = 0 # also as the offset of layer_id
    ):
        """
        Stores attention output of current batch into buffer o
        """

        events = self.events[cur_stage]
        o = batch.attn_out_buf.view(batch.iter_width, -1, self.model_config.head_dim)
        cur_layer_id = (self.layer_id + cur_stage) % self.model_config.num_layers

        if batch.num_prefs > 0:
            # If we are using Ampere GPU or above, we can use the flash attention kernel.
            # Otherwise, we use a customized prefilling kernel.
            if torch.cuda.get_device_properties(0).major >= 8:
                # pylint: disable=c-extension-no-member
                if _USE_VLLM_FLASH_ATTN:
                    flash_attn_cuda.varlen_fwd(
                        q[:batch.sum_pref_toks],
                        k[:batch.sum_pref_toks],
                        v[:batch.sum_pref_toks],
                        o[:batch.sum_pref_toks],
                        batch.pref_st_locs_we,
                        batch.pref_st_locs_we,
                        None,
                        None,  # block table
                        None,  # alibi slopes
                        batch.max_pref_toks,
                        batch.max_pref_toks,
                        0.0,
                        self.model_config.softmax_scale,
                        False,
                        True,  # causal
                        -1,    # window size 0
                        -1,    # window size 1
                        0.0,   # softcap
                        False, # return softmax
                        None
                    )
                else:
                    _out = _flash_attn_varlen_func(
                        q[:batch.sum_pref_toks],
                        k[:batch.sum_pref_toks],
                        v[:batch.sum_pref_toks],
                        cu_seqlens_q=batch.pref_st_locs_we,
                        cu_seqlens_k=batch.pref_st_locs_we,
                        max_seqlen_q=batch.max_pref_toks,
                        max_seqlen_k=batch.max_pref_toks,
                        dropout_p=0.0,
                        softmax_scale=self.model_config.softmax_scale,
                        causal=True,
                        window_size=(-1, -1),
                        softcap=0.0,
                        return_attn_probs=False,
                    )
                    o[:batch.sum_pref_toks].copy_(_out)
            else:
                prefill_attention(
                    q, k, v, o[:batch.sum_pref_toks],
                    self.model_config, self.engine_config, batch
                )
        events.pf_record("pref_e")

        # Actually we can further separate KV-cache storing for prefilling and decoding,
        # but since the kernel is fast enough, we put all to decoding stream for simplicity
        if batch.num_gdecs > 0:
            # with torch.cuda.stream(self.decoding_piggyback_stream):
            #     torch.cuda.current_stream().wait_event(self.events[cur_stage].stage_s)
            paged_attention(
                q[batch.sum_pref_toks:batch.sum_prgd_toks],
                k[batch.sum_pref_toks:batch.sum_prgd_toks],
                v[batch.sum_pref_toks:batch.sum_prgd_toks],
                o[batch.sum_pref_toks:batch.sum_prgd_toks],
                self.swapper.k_cache, 
                self.swapper.v_cache,
                self.model_config.softmax_scale,
                self.swapper.gpu_block_table,
                batch.prgd_seq_ids[batch.num_prefs:],
                batch.prgd_seq_lens[batch.num_prefs:],
                cur_layer_id,
                batch.seq_block_size,
                batch.num_seq_blocks,
            )
        events.pf_record("gdec_e")
                
        if batch.num_cdecs > 0:
            oc = self.swapper.o_cpu[:batch.num_cdecs]
            events.pf_time("lnch_m")
            self.events[cur_stage].qkvtr_e.synchronize()
            events.pf_time("cdec_s")
            torch.ops.pacpu.paged_attention_cpu(
                cur_layer_id,
                self.model_config.softmax_scale,
                batch.seq_ids_list[batch.num_prgds:],
                batch.seq_lens_list[batch.num_prgds:],

                self.swapper.q_cpu[:batch.num_cdecs],
                self.swapper.k_cpu[:batch.num_cdecs],
                self.swapper.v_cpu[:batch.num_cdecs],
                self.swapper.k_swap,
                self.swapper.v_swap,
                self.swapper.cpu_block_table,
                oc
            )
            events.pf_time("cdec_e")
            with torch.cuda.stream(self.cpu_communication_stream):
                o[-batch.num_cdecs:, :].copy_(oc, non_blocking=True)
        else:
            events.pf_time_nocpu()
        self._compute_wait_comm() # Wait for CPU decoding to finish


    def _postproj(
        self,
        batch: SubBatch        
    ) -> torch.Tensor:
        o = linear(batch.attn_out_buf, self.weight.o_proj)
        self._maybe_allreduce(o)
        fused_add_rmsnorm_inplace(o, batch.residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)
        ug = linear(o, self.weight.up_gate_proj)
        del o
        silu_and_mul_inplace(ug)
        embeddings = linear(ug[:, :ug.shape[1] // 2], self.weight.down_proj)
        del ug
        return embeddings


    def forward(self, batch: SubBatch, embeddings: torch.Tensor) -> torch.Tensor:
        """
        (fused) Add last layer's residual, and perform RMSNorm
        Before: input_embds is the output of the last FFN block, and residual_buf
                is the residual to be added to input_embds
        After: input_embds will be RMSNorm(input_embds + residual_buf), and
               residual_buf will be input_embds + residual_buf (which will be
               used as the residual after the attention block)
        """
        self.events[0].pf_record("stage_s")
        self.events[0].pf_time("lnch_s")
        q, k, v = self._preproj(embeddings, batch)
        self.events[0].pf_record("linr_e")
        self._transfer_qkv(q, k, v, batch)
        self._attention(q, k, v, batch)
        del q, k, v
        self.events[1].pf_record("stage_s")
        self._swap_out_blocks(batch)
        embeddings = self._postproj(batch)
        self.events[0].pf_time("lnch_e")
        self.events[1].pf_record("linr_e")
        return embeddings


    def _forward_pipeline_stage(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        batches: list[SubBatch],
        cur_stage: int,
    ) -> tuple[torch.Tensor]:
        """
        Do 1 pipeline stage:

            batch 0 : o0   |=> post-projection[i] -> pre-projection[i+1] |=> qkv0
            batch 1 : qkv1 |=>      attention[i + attn_layer_id_offs]    |=> [o1]

        buffer of o1 is given as input

        Note that batch-0 here may actually be batch-1 in the forward pass, should reverse the list
        in the second stage
        """
        self.events[cur_stage].pf_record("stage_s")
        self.events[cur_stage].pf_time("lnch_s")
        self._transfer_qkv(q1, k1, v1, batches[cur_stage^1], cur_stage=cur_stage)
        self._swap_out_blocks(batches[cur_stage])
        e0 = self._postproj(batches[cur_stage])
        q0, k0, v0 = self._preproj(e0, batches[cur_stage], layer_off=1)
        del e0
        self.events[cur_stage].pf_record("linr_e")
        self._attention(q1, k1, v1, batches[cur_stage^1], cur_stage=cur_stage)
        self.events[cur_stage].pf_time("lnch_e")

        return q0, k0, v0


    def forward_double(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        batches: list[SubBatch]
    ) -> tuple[torch.Tensor]:
        """
        Do all jobs for 1 transformer layer for 2 batches

        Note that the weights of pre-projection need to be of the next layer compared to the post-projection

            batch 0 : o0   |=>  post-projection[i] -> pre-projection[i+1]  |        attention[i+1]                     |=> [o0']
            batch 1 : qkv1 |=>       attention[i]                          | post-projection[i] -> pre-projection[i+1] |=> qkv1'
        """
        q0, k0, v0 = self._forward_pipeline_stage(q1, k1, v1, batches, cur_stage=0)
        q1, k1, v1 = self._forward_pipeline_stage(q0, k0, v0, batches, cur_stage=1)

        return q1, k1, v1


    def forward_first_stage(
        self,
        embeddings: torch.Tensor,
        batches: list[SubBatch]
    ) -> tuple[torch.Tensor]:
        """
        Do the first stage of the pipeline for 2 batches

        batch0 : embeddings0 |=> pre-projection -> attention       |=> [o0]
        batch1 : embeddings1 |=>                  pre-projection   |=> q1, k1, v1
        """
        embeddings = torch.split(embeddings, [batch.iter_width for batch in batches])
        q0, k0, v0 = self._preproj(embeddings[0], batches[0], layer_off=1)
        self._compute_wait_comm() # Here we must make sure all swaps are done before the first attention

        self.events[1].pf_record("stage_s")
        self.events[1].pf_time("lnch_s")
        self._transfer_qkv(q0, k0, v0, batches[0], cur_stage=1)
        q1, k1, v1 = self._preproj(embeddings[1], batches[1], layer_off=1)
        self.events[1].pf_record("linr_e")
        self._attention(q0, k0, v0, batches[0], cur_stage=1)
        self.events[1].pf_time("lnch_e")

        return q1, k1, v1


    def forward_last_stage(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        batches: list[SubBatch]
    ) -> torch.Tensor:
        """
        Do the last stage of the pipeline for 2 batches, return the concatenated output

        batch0 : o0   |=> post-projection              |=> [f0]
        batch1 : qkv1 |=> attention -> post-projection |=> [f1]
        """
        self.events[0].pf_record("stage_s")
        self.events[0].pf_time("lnch_s")
        self._transfer_qkv(q1, k1, v1, batches[1], cur_stage=0)
        self._swap_out_blocks(batches[0])
        e0 = self._postproj(batches[0])
        self.events[0].pf_record("linr_e")
        self._attention(q1, k1, v1, batches[1], cur_stage=0)
        self.events[0].pf_time("lnch_e")

        e1 = self._postproj(batches[1])

        return torch.cat((e0, e1))
 