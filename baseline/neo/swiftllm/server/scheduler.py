"""
A smart scheduler for the SwiftLLM engine that does batch picking and mode selection.
"""

import sys
import math
import logging
from collections import deque

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.utils import cdiv
from swiftllm.structs import Request, SubBatch
from swiftllm.perfpredictor import PerfPredictor

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class RequestIdManager:
    """
    A class that maintains available request ids
    """
    def __init__(self, max_id: int):
        # Id should be in range [0, max_id)
        self.max_id = max_id
        self.available_ids = deque(range(max_id))
    
    def get_id(self) -> int:
        """
        Get an available request id
        """
        if not self.available_ids:
            raise RuntimeError("No more available request ids. Please try to increase `max_seqs_in_block_table`")
        return self.available_ids.popleft()

    def get_num_available_ids(self) -> int:
        """
        Get the number of available request ids
        """
        return len(self.available_ids)
    
    def free_id(self, req_id: int):
        """
        Free a request id
        """
        self.available_ids.append(req_id)
    
    def free_ids(self, req_ids: list[int]):
        """
        Free a list of request ids
        """
        self.available_ids.extend(req_ids)

class ScheduleBudget:
    """
    A class that maintains the budget for scheduling
    """
    def __init__(self, max_batch_size: int, max_tokens_in_batch: int):
        self.remaining_batch_size = max_batch_size
        self.remaining_tokens_in_batch = max_tokens_in_batch
    
    @property
    def overspent(self) -> bool:
        """
        Check if the budget is overspent
        """
        return self.remaining_batch_size < 0 or self.remaining_tokens_in_batch < 0

    def check_and_substract(self, num_tokens) -> bool:
        """
        Check if the budget is enough. 
        
        If so, substract the tokens from the budget and return True. Otherwise, return False.
        """
        if self.remaining_batch_size >= 1 and \
            self.remaining_tokens_in_batch >= num_tokens:
            self.remaining_batch_size -= 1
            self.remaining_tokens_in_batch -= num_tokens
            return True
        return False

    def add(self, num_tokens) -> bool:
        """
        Add tokens to the budget
        """
        self.remaining_batch_size += 1
        self.remaining_tokens_in_batch += num_tokens

class Scheduler:
    """
    A strict FCFS scheduler for the LLM engine, which supports paged attention
    as well as swapping in/out
    """

    def __init__(
        self, 
        engine_config: EngineConfig,
        model_config: LlamaModelConfig,
        predictor: PerfPredictor
    ):
        self.engine_config = engine_config
        self.model_config = model_config

        self.predictor = predictor

        # Request in the following three deques are sorted by their arrival time
        self.waiting_q: deque[Request] = deque()
        self.gpu_decoding_q: list[Request] = []
        self.cpu_decoding_q: deque[Request] = deque()

        self.num_itm_blocks = engine_config.num_gpu_blocks
        self.num_gpu_blocks = engine_config.num_gpu_blocks * (not engine_config.disable_partial_offl)

        # Number of GPU blocks occupied by decoding requests
        # This number should always equal to sum(self._get_block_needed(req) for req in self.gpu_decoding_q)
        self.num_decoding_gpu_blocks = 0

        self.request_id_manager = RequestIdManager(self.engine_config.max_seqs_in_block_table)
    
    def _get_block_needed(self, request: Request) -> int:
        """
        Get the number of blocks needed for a request
        """
        return cdiv(request.seq_len, self.engine_config.block_size)
    
    def on_requests_arrival(self, requests: list[Request]):
        """
        Called when a batch of new requests arrives and finishes tokenization
        """
        self.waiting_q.extend(requests)

    def _get_remains(self, batches: list[SubBatch]) -> float:
        assert len(batches) == 2
        return [
            batches[j^1].perfdata.linr_T + 
            batches[j].perfdata.pref_T + 
            batches[j].perfdata.gdec_T - 
            batches[j].perfdata.cpu_time
            for j in range(2)
        ]

    def _decide_mode_and_gen_batch(
        self, 
        gpu_prefill_reqs: list[Request],
        cpu_prefill_reqs: list[Request],
        budget: ScheduleBudget
    ) -> list[SubBatch]:
        """
        Assume that self.gpu_decoding_q and self.prefilling_q are fixed.

        Pick 2 sub-batches using heuristics and compare the TP with seqential mode.

        Returns:
            (batch, None) if using sequential mode, or
            (batch0, batch1) if using pipelined mode
        """
        assert not self.engine_config.always_use_gpu, "This function is not designed for GPU-only mode"
        batches = [SubBatch(self.predictor) for _ in range(2)]
        gpu_only_batch = SubBatch(self.predictor)

        # Step 1: put all pref and gdec sequences into the first batch.
        for req in gpu_prefill_reqs:
            batches[0].add_pref(req, is_gpu=True)
            gpu_only_batch.add_pref(req, is_gpu=True)

        for req in cpu_prefill_reqs:
            batches[0].add_pref(req, is_gpu=False)
            gpu_only_batch.add_pref(req, is_gpu=False)
        
        for req in self.gpu_decoding_q:
            batches[0].add_gdec(req)
            gpu_only_batch.add_gdec(req)

        if not batches[0] and self.num_gpu_blocks > 0:
            return []

        # Step 2: adjust the number of prefilled sequences in gpu_only_batch
        while gpu_only_batch.get_num_prefs():
            req, is_gpu = gpu_only_batch.pop_pref()
            if is_gpu or gpu_only_batch.perfdata.s < self.predictor.linr_S_threshold:
                gpu_only_batch.add_pref(req, is_gpu)
                break

        # Step 3: split CPU decoding requests.
        min_out_cpu_len = 1e9
        next_batch_idx = 1
        for req in self.cpu_decoding_q:
            if not budget.check_and_substract(1):
                break
            if req.seq_len >= min_out_cpu_len:
                budget.add(1)
                continue
            # remains[i] is the remaining of Cdec capacity of batch i
            batches[next_batch_idx].add_cdec(req)
            remains = self._get_remains(batches)
            assert all(not math.isnan(r) for r in remains), remains
            if min(remains) < 0 and self.num_gpu_blocks > 0:
                # Skip this request
                min_out_cpu_len = req.seq_len
                budget.add(1)
                batches[next_batch_idx].pop_cdec()
                continue
            next_batch_idx = remains[1] > remains[0]

        if not batches[1] and self.num_gpu_blocks > 0:
            return [gpu_only_batch] # This is to prevent division by zero

        if self.num_gpu_blocks == 0:
            ret = []
            if len(batches[0]) > 0:
                ret.append(batches[0])
            if len(batches[1]) > 0:
                ret.append(batches[1])
            return ret


        # Step 4: reduce the number of prefilled sequences in the first batch if CPU is idle for too long
        while batches[0].get_num_prefs():
            req, is_gpu = batches[0].pop_pref()
            if is_gpu or batches[0].perfdata.s < self.predictor.linr_S_threshold or min(self._get_remains(batches)) < 0:
                batches[0].add_pref(req, is_gpu)
                break

        # Step 5: check if pipelined mode is better
        seqential_time = gpu_only_batch.perfdata.gpu_time * self.model_config.num_layers
        pipelined_time = (batches[0].perfdata.gpu_time + batches[1].perfdata.gpu_time) * self.model_config.num_layers
        seqential_rate = len(gpu_only_batch) / seqential_time
        pipelined_rate = sum(len(batches[i]) for i in range(2)) / pipelined_time
        # print(f"Sequential time: {seqential_time}, Pipelined time: {pipelined_time}")
        # print(f"Sequential rate: {seqential_rate}, Pipelined rate: {pipelined_rate}")
        if seqential_rate < pipelined_rate:
            return batches
        else:
            return [gpu_only_batch]
        # return [gpu_only_batch]

    def _get_next_batch_new(self) -> tuple[list[SubBatch], list[Request], list[Request]]:
        """
        Called when the engine wants a new batch to be forwarded
        Returns (new_batch, newly_swapped_in, newly_swapped_out)
        """
        # We have a budget mainly because we should avoid CUDA OOM
        budget = ScheduleBudget(
            self.engine_config.max_batch_size,
            self.engine_config.max_tokens_in_batch
        )
        pref_to_cpu = []
        pref_to_gpu = []
        swpout_reqs = []
        swpin_reqs = []

        # Policy may change, should recompute these thresholds on every iteration
        swap_out_threshold = self.num_gpu_blocks
        # swap_in_threshold = round(swap_out_threshold * 0.95)
        # swap_out_threshold = self.num_gpu_blocks
        swap_in_threshold = round(swap_out_threshold * 0.95)
        cpu_threshold = self.engine_config.num_cpu_blocks - self.engine_config.num_gpu_blocks
        
        # Step 1: Try to launch as many GPU decoding requests as possible
        gpu_block_needed = sum(self._get_block_needed(req) for req in self.gpu_decoding_q)
        budget.remaining_batch_size -= len(self.gpu_decoding_q)
        budget.remaining_tokens_in_batch -= len(self.gpu_decoding_q)

        # Step 2: Swap out requests if necessary
        while budget.overspent or gpu_block_needed > swap_out_threshold:
            # Preempt the last running seq
            victim = self.gpu_decoding_q.pop()
            self.cpu_decoding_q.appendleft(victim)
            swpout_reqs.append(victim)
            gpu_block_needed -= self._get_block_needed(victim)
            budget.add(1)

        # Step 3: Swap in requests if possible
        while self.cpu_decoding_q:
            candidate = self.cpu_decoding_q[0]
            cur_block_needed = self._get_block_needed(candidate)
            if gpu_block_needed + cur_block_needed > swap_in_threshold or \
            not budget.check_and_substract(1):
                break
            gpu_block_needed += cur_block_needed
            swpin_reqs.append(candidate)
            self.cpu_decoding_q.popleft()
            self.gpu_decoding_q.append(candidate)
        assert not swpout_reqs or not swpin_reqs

        # Step 4: Launch prefilling requests, just to know the uplimit
        itm_block_needed = 0
        cpu_block_needed = sum(self._get_block_needed(req) for req in self.cpu_decoding_q) # for bounding new prefillings
        for i, candidate in enumerate(self.waiting_q):
            cur_block_needed = self._get_block_needed(candidate)
            if  itm_block_needed + cur_block_needed > self.num_itm_blocks or \
                cpu_block_needed + cur_block_needed > cpu_threshold or \
                self.request_id_manager.get_num_available_ids() < i + 1 or \
                not budget.check_and_substract(candidate.prompt_len):
                break
            # The newly prefilled sequences are the major part of swapping out. Here we use a simple heuristic.
            # 1. We prefer to put a sequence into GPU.
            # 2. If the GPU is full, we put the sequence into CPU.
            # 3. For fairness, if some earlier sequences are in CPU, we should put the later sequences into CPU.
            if not pref_to_cpu and gpu_block_needed + cur_block_needed <= self.num_gpu_blocks:
                gpu_block_needed += cur_block_needed
                pref_to_gpu.append(candidate)
            else:
                cpu_block_needed += cur_block_needed
                itm_block_needed += cur_block_needed
                pref_to_cpu.append(candidate)

        # Step 5: Launch CPU decoding requests and form batches
        batches = self._decide_mode_and_gen_batch(pref_to_gpu, pref_to_cpu, budget)

        # Step 6: Launch prefilled requests for real
        real_num_prefs = sum(b.get_num_prefs() for b in batches)
        pref_to_gpu = pref_to_gpu[:real_num_prefs]
        pref_to_cpu = pref_to_cpu[:real_num_prefs - len(pref_to_gpu)]
        for _ in range(real_num_prefs):
            candidate = self.waiting_q.popleft()
            candidate.request_id = self.request_id_manager.get_id()

        if pref_to_gpu or pref_to_cpu:
            logger.info(
                "Gdecs: %d, Cdecs: %d, Pr2gs: %d, Pr2cs: %d, Waiting: %d",
                len(self.gpu_decoding_q), len(self.cpu_decoding_q), len(pref_to_gpu), len(pref_to_cpu), len(self.waiting_q)
            )

        self.gpu_decoding_q.extend(pref_to_gpu)
        self.cpu_decoding_q.extend(pref_to_cpu)

        return batches, swpout_reqs, swpin_reqs

    def _get_next_batch_old(self) -> tuple[list[SubBatch], list[Request], list[Request]]:
        # Try to launch a decoding batch
        self.num_decoding_gpu_blocks = sum(self._get_block_needed(req) for req in self.gpu_decoding_q)
        newly_swapped_out = []
        while len(self.gpu_decoding_q) > self.engine_config.max_batch_size or \
            self.num_decoding_gpu_blocks > self.num_gpu_blocks:
            # Preempt the last running seq
            victim = self.gpu_decoding_q.pop()
            self.num_decoding_gpu_blocks -= self._get_block_needed(victim)
            newly_swapped_out.append(victim)
        newly_swapped_out.reverse()   # Keep it in the order of arrival time
        self.cpu_decoding_q.extendleft(newly_swapped_out)

        if not self.cpu_decoding_q:
            cur_batch = SubBatch()
            cur_batch_block_needed = self.num_decoding_gpu_blocks
            for req in self.gpu_decoding_q:
                cur_batch.add_gdec(req)

            # Try to launch a new prefill batch
            while self.waiting_q:
                cur_seq: Request = self.waiting_q[0]
                cur_seq_block_needed = self._get_block_needed(cur_seq)
                if  len(cur_batch)+1 <= self.engine_config.max_batch_size and \
                    cur_batch_block_needed + cur_seq_block_needed <= self.num_gpu_blocks and \
                    cur_batch.perfdata.s + cur_seq.prompt_len <= self.engine_config.max_tokens_in_batch:
                    cur_batch.add_pref(cur_seq, True)
                    cur_batch_block_needed += cur_seq_block_needed
                    self.waiting_q.popleft()  
                else:
                    # Strict FCFS
                    break
            if len(cur_batch):
                # Going to launch a prefill batch
                # If you want decoding requests to be piggybacked, you can do it here
                for req in cur_batch.gprf_reqs:
                    req.request_id = self.request_id_manager.get_id()
                if len(cur_batch.gprf_reqs) > 0:
                    logger.info(f"Waiting: {len(self.waiting_q)}, Prefs: {len(cur_batch.gprf_reqs)}, Gdecs: {len(self.gpu_decoding_q)}, Cdecs: {len(self.cpu_decoding_q)}")
                self.gpu_decoding_q.extend(cur_batch.gprf_reqs)
                self.num_decoding_gpu_blocks = cur_batch_block_needed
                return [cur_batch], [], []
 
        newly_swapped_in = []
        if not newly_swapped_out:
            # No swap-out triggered, try to swap in some requests if possible
            while self.cpu_decoding_q:
                cur_seq = self.cpu_decoding_q[0]
                num_cur_seq_blocks = self._get_block_needed(cur_seq)
                if len(self.gpu_decoding_q) + 1 <= self.engine_config.max_batch_size and \
                    self.num_decoding_gpu_blocks + num_cur_seq_blocks <= self.num_gpu_blocks:
                    self.gpu_decoding_q.append(cur_seq)
                    self.num_decoding_gpu_blocks += num_cur_seq_blocks
                    self.cpu_decoding_q.popleft()
                    newly_swapped_in.append(cur_seq)
                else:
                    break
        
        cur_batch = SubBatch(self.predictor)
        for req in self.gpu_decoding_q:
            cur_batch.add_gdec(req)
        return ([cur_batch] if cur_batch else []), newly_swapped_out, newly_swapped_in

    def get_next_batch(self) -> tuple[list[SubBatch], list[Request], list[Request]]:
        """
        Get the next batch(es) to be forwarded.

        Returns a tuple: (new_batch(es), newly_swapped_out_reqs, newly_swapped_in_reqs)
        """
        if self.engine_config.always_use_gpu:
            return self._get_next_batch_old()
    
        return self._get_next_batch_new()
    
    def remove_finished_requests(self, reqs: list[Request]):
        """
        Remove the finished requests from the decoding queues, and free their request ids
        """
        def not_finished_func(req: Request) -> bool:
            return not req.is_finished()
        self.gpu_decoding_q = list(filter(not_finished_func, self.gpu_decoding_q))
        self.cpu_decoding_q = deque(filter(not_finished_func, self.cpu_decoding_q))

        self.request_id_manager.free_ids([req.request_id for req in reqs])
