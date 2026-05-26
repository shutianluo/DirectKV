"""
This module provides a profiler for the Llama model.
"""

import time
import os
import sys
import json
import math
import logging

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from swiftllm.perfpredictor import TablePerfPredictor
from swiftllm.structs import create_request, SubBatch
from swiftllm.utils import GB

from swiftllm.worker.model import ModelPerfResult
from swiftllm.server.block_manager import BlockManager
from swiftllm.server.executor import Executor

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class ModelProfiler:
    """
    A profiler for the Llama model.
    """
    @torch.inference_mode()
    def __init__(self, executor: Executor):
        self.engine_config = executor.engine_config
        self.model_config = executor.model_config
        self.executor = executor
        self.pp = None
        self.bm = None
        os.makedirs(self.engine_config.profile_result_path, exist_ok=True)

    def init_profile_tables(self, block_manager: BlockManager):
        """
        Initialize the profile tables
        """
        # Validate necessary constraints
        engine_config = self.executor.engine_config

        self.bm = block_manager
        self.pp = TablePerfPredictor(engine_config)

        self.pp.linr_S_list, self.pp.linr_T_list = self._profile_linr(self.pp.linr_S_list)
        self.pp.pref_S_list, self.pp.pref_T_list = self._profile_pref(self.pp.pref_S_list)
        self.pp.gdec_N_list, self.pp.gdec_T_list = self._profile_gdec(self.pp.gdec_N_list)
        self.pp.cdec_S_list, self.pp.cdec_N_lists, self.pp.cdec_T_lists = self._profile_cdec(self.pp.cdec_S_list, self.pp.cdec_N_lists)
      

    def _run_test_case_seq(
        self,
        pref_lens: list[int],
        gdec_lens: list[int],
        cdec_lens: list[int],
        nwarmup = 2,
        nrepeat = 3
    ):
        return self._run_test_case([pref_lens], [gdec_lens], [cdec_lens], nwarmup, nrepeat)
    

    def _run_test_case_pip_same(
        self,
        pref_lens: list[int],
        gdec_lens: list[int],
        cdec_lens: list[int],
        nwarmup = 2,
        nrepeat = 3
    ):
        return self._run_test_case([pref_lens] * 2, [gdec_lens] * 2, [cdec_lens] * 2, nwarmup, nrepeat)
    

    @torch.inference_mode()
    def _run_test_case(
      self,
      pref_lens: list[list[int]],
      gdec_lens: list[list[int]],
      cdec_lens: list[list[int]],
      nwarmup = 2,
      nrepeat = 3
    ) -> ModelPerfResult:
        """
        Run a artificial test case and return the performance results.
        """
        # print(f"Running test case with pref_lens={pref_lens}, gdec_lens={gdec_lens}, cdec_lens={cdec_lens}")

        nbatches = len(pref_lens)
        assert nbatches in (1, 2), "Only support 1 or 2 batches"

        batches = []
        offs = 0
        for i in range(nbatches):
            batch = SubBatch()
            npref = len(pref_lens[i])
            ngdec = len(gdec_lens[i])
            ncdec = len(cdec_lens[i])

            for j in range(npref):
                batch.add_pref(create_request([10] * pref_lens[i][j], offs + j, [], True), is_gpu=True)

            for j in range(ngdec):
                batch.add_gdec(create_request([10] * (gdec_lens[i][j] - 1), offs + npref + j, [10], True))

            for j in range(ncdec):
                batch.add_cdec(create_request([10] * (cdec_lens[i][j] - 1), offs + npref + ngdec + j, [10], True))

            offs += npref + ngdec + ncdec
            batches.append(batch)

        if self.bm is None:
            for batch in batches:
                batch.set_model_forward_args(self.model_config)
            args = (([], []), ([], [])), ([], [])      
        else: 
            args = self.bm.prepare(batches, [], [])
                  
        for i in range(-nwarmup, nrepeat):
            if i == 0:
                self.executor.turn_on_perf_monitor()
            # Directly call this since we already allocated the blocks
            output_tokens = self.executor.do_one_iteration(batches, *args)

        if self.bm is not None:
            # The requests would all finish due to quick-stop.
            self.bm.update_and_free(batches, output_tokens)

        res = self.executor.turn_off_perf_monitor_and_flush_results()
        return res
    
    def _profile_linr(
        self, 
        S_list: list[int]
    ) -> list[float]:
        """
        Profile model's linear part performance.
        """
        result_path = self.engine_config.profile_result_path + "linr.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                table = json.load(f)
            if table["S_list"][-1] >= S_list[-1]:
                return table["S_list"], table["T_list"]
        else:
            table = {
                "S_list": [],
                "T_list": []
            }
            
        print(f"Profiling linear part with S_list={S_list} ...")

        T_list = []
        # Use reversed order to reveal problem earlier
        for S in tqdm(list(reversed(S_list))):
            if S in table["S_list"]:
                T_list.append(table["T_list"][table["S_list"].index(S)])
                continue
            res = self._run_test_case_seq(
                pref_lens=[S],
                gdec_lens=[],
                cdec_lens=[]
            )
            T_list.append(ModelPerfResult.mean(res, "avg_linr_time"))
        T_list = list(reversed(T_list))

        with open(result_path, "w") as f:
            json.dump({
            "S_list": S_list,
            "T_list": T_list
            }, f, indent=2)

        plt.figure(figsize=(16, 12))
        plt.plot(S_list, T_list)
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel("S")
        plt.ylabel("T_l(ms)")
        plt.savefig(self.engine_config.profile_result_path + "linr.png")
        plt.close()

        return S_list, T_list

    def _profile_pref(
        self,
        S_list: list[int]
    ) -> list[list[float]]:
        """
        Profile model's GPU prefilling attention part performance.
        """
        result_path = self.engine_config.profile_result_path + "pref.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                table = json.load(f)
            if table["S_list"][-1] >= S_list[-1]:
                return table["S_list"], table["T_list"]
            
        print(f"Profiling prefill part with S_list={S_list}...")

        T_list = []
        for S in tqdm(S_list):
            res = self._run_test_case_seq(
                pref_lens=[S],
                gdec_lens=[],
                cdec_lens=[]
            )
            T_list.append(ModelPerfResult.mean(res, "avg_pref_time"))

        plt.figure(figsize=(16, 12))
        plt.plot(S_list, T_list)
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel("S")
        plt.ylabel("T(ms)")
        plt.savefig(self.engine_config.profile_result_path + "pref.png")
        plt.close()

        with open(result_path, "w") as f:
            json.dump({
            "S_list": S_list,
            "T_list": T_list
            }, f, indent=2)

        return S_list, T_list

    def _profile_gdec(
        self,
        N_list: list[int]
    ) -> list[float]:
        """
        Profile model's GPU attention part performance.
        """
        result_path = self.engine_config.profile_result_path + "gdec.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                res = json.load(f)
            if res["N_list"][-1] >= N_list[-1]:
                return res["N_list"], res["T_list"]
            
        print(f"Profiling GPU attention part with N_list={N_list} ...")

        T_list = []
        L = self.engine_config.max_seq_len
        for N in tqdm(N_list):
            res = self._run_test_case_seq(
                pref_lens=[],
                gdec_lens=[L] * ((N - 1) // L) + [(N - 1) % L + 1],
                cdec_lens=[]
            )
            T_list.append(ModelPerfResult.mean(res, "avg_gdec_time"))

        with open(result_path, "w") as f:
            json.dump({
            "N_list": N_list,
            "T_list": T_list
            }, f, indent=2)

        plt.figure(figsize=(16, 12))
        plt.plot(N_list, T_list)
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel("N")
        plt.ylabel("T(ms)")
        plt.savefig(self.engine_config.profile_result_path + "gdec.png")
        plt.close()

        return N_list, T_list   

    def _profile_cdec(
        self,
        S_list: list[int],
        N_lists: list[list[int]]
    ) -> list[list[float]]:
        """
        Profile model's CPU attention part performance.
        """
        result_path = self.engine_config.profile_result_path + "cdec.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                table = json.load(f)
            if table["S_list"][-1] >= S_list[-1] and table["N_lists"][-1][-1] >= N_lists[-1][-1]:
                return table["S_list"], table["N_lists"], table["T_lists"]
            
        print(f"Profiling CPU attention part with S_list={S_list}, N_lists={N_lists} ...")
            
        T_lists = []
        block_size = self.engine_config.block_size
        for i, S in enumerate(tqdm(S_list)):
            T_lists.append([])
            for N in self.pp.cdec_N_list_agg:
                if N < N_lists[i][0]:
                    T_lists[-1].append(0.0)
                    continue
                if N > N_lists[i][-1]:
                    T_lists[-1].append(float("inf"))
                    continue
                assert N % block_size == 0, "N must be divisible by block size"
                NB = N // block_size
                res = self._run_test_case_seq(
                    # Divide N into S segments as even as possible
                    pref_lens=[],
                    gdec_lens=[],
                    cdec_lens=[NB // S * block_size] * (S - NB % S) + [(NB // S + 1) * block_size] * (NB % S),
                )
                T_lists[-1].append(ModelPerfResult.mean(res, "avg_cdec_time"))

        nS = len(S_list)
        nN = len(self.pp.cdec_N_list_agg)
        for i in range(nS):
            for j in reversed(range(nN)):
                if T_lists[i][j] == 0.0:
                    assert i > 0 and j < nN - 1
                    T_lists[i][j] = T_lists[i - 1][j] + T_lists[i][j + 1] - T_lists[i - 1][j + 1]

        for i in reversed(range(nS)):
            for j in range(nN):
                if T_lists[i][j] == float("inf"):
                    assert i < nS - 1 and j > 0
                    T_lists[i][j] = T_lists[i + 1][j] + T_lists[i][j - 1] - T_lists[i + 1][j - 1]

        T_array = np.array(T_lists)

        plt.figure(figsize=(16, 12))
        ax = plt.axes(projection='3d')
        ax.plot_surface(
            np.outer(S_list, np.ones(nN)),
            np.outer(np.ones(nS), self.pp.cdec_N_list_agg),
            T_array,
            label = "CPU"
        )

        with open(result_path, "w") as f:
            json.dump({
            "S_list": S_list,
            "N_lists": N_lists,
            "T_lists": T_lists
            }, f, indent=2)

        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_xlabel("S_c")
        ax.set_ylabel("N_c")
        ax.set_zlabel("T(ms)")
        plt.savefig(self.engine_config.profile_result_path + "cdec.png")
        plt.close()

        return S_list, N_lists, T_lists

    def _profile_lnch(
        self,
        S_list: list[int]
    ) -> list[float]:
        """
        Profile model's kernel launch time.
        """
        result_path = self.engine_config.profile_result_path + "lnch.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                res = json.load(f)
            if res["S_list"] == S_list:
                return res["T_list"]
            
        print(f"Profiling kernel launch time with S_list={S_list} ...")

        T_list = []
        for S in tqdm(S_list):
            res = self._run_test_case_pip_same(
                pref_lens=[S // 2 - 2 * (S // 10)],
                gdec_lens=[10] * (S // 10),
                cdec_lens=[10] * (S // 10)
            )
            T_list.append(ModelPerfResult.mean(res, "avg_lnch_time"))

        with open(result_path, "w") as f:
            json.dump({
            "S_list": S_list,
            "T_list": T_list
            }, f, indent=2)

        plt.figure(figsize=(16, 12))
        plt.plot(S_list, T_list)
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel("S")
        plt.ylabel("T(ms)")
        plt.savefig(self.engine_config.profile_result_path + "lnch.png")
        plt.close()

        T_mean = np.array(T_list).mean()

        return T_mean

    @torch.inference_mode()
    def profile_num_blocks(self):
        """
        Profile the number of GPU blocks

        We run a forged prefill batch with the maximum number of tokens and
        sequences, record the peak memory usage, and infer the number of blocks
        that can be allocated.

        Finally, we set the number of GPU blocks in the engine configuration.
        """
        engine_config = self.engine_config
        model_config = self.model_config
        gpu_block_size_bytes = engine_config.block_size * model_config.get_kvslot_size(engine_config.extra_layer_for_cprf)
        cpu_block_size_bytes = engine_config.block_size * model_config.get_kvslot_size(False)
        engine_config.num_cpu_blocks = engine_config.swap_space * GB // cpu_block_size_bytes

        if self.engine_config.num_gpu_blocks_override == -1:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            ws = self.engine_config.tensor_parallel_degree

            # Synthesize a prefill batch
            N = engine_config.max_tokens_in_batch
            S = engine_config.max_batch_size
            self._run_test_case_seq(
                pref_lens=[N // S] * (S - N % S) + [N // S + 1] * (N % S),
                gdec_lens=[],
                cdec_lens=[],
                nrepeat=1, nwarmup=0
            )
            torch.cuda.synchronize()

            # peak_memory = torch.cuda.max_memory_allocated()
            # total_memory = torch.cuda.get_device_properties(0).total_memory
            peak_memory = 0
            for i in range(ws):
                free_memory, total_memory = torch.cuda.mem_get_info(i)
                single_peak_memory = total_memory - free_memory
                peak_memory = max(peak_memory, single_peak_memory)
            useable_memory = total_memory * engine_config.gpu_mem_utilization
            print(f"[Engine.profiler] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB")
            if useable_memory < peak_memory:
                raise RuntimeError(
                    f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB "
                    f"({total_memory/GB:.2f} GB * {engine_config.gpu_mem_utilization})"
                )
            
            torch.cuda.empty_cache()
            engine_config.num_gpu_blocks = math.floor((useable_memory - peak_memory) / gpu_block_size_bytes)
        else:
            engine_config.num_gpu_blocks = engine_config.num_gpu_blocks_override

        assert engine_config.num_gpu_blocks * self.engine_config.block_size >= self.engine_config.max_tokens_in_batch, \
            f"Number of GPU blocks {self.engine_config.num_gpu_blocks} is not enough to hold the maximum batch size"

        num_gpu_blocks = engine_config.num_gpu_blocks
        num_cpu_blocks = engine_config.num_cpu_blocks
        logger.info(f"[Engine.profiler] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks * gpu_block_size_bytes/GB:.2f} GB)")
        logger.info(f"[Engine.profiler] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks * cpu_block_size_bytes/GB:.2f} GB)")
