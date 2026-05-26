"""
Performance predictor for the SwiftLLM engine.
"""

from swiftllm.engine_config import EngineConfig 

class PerfPredictor:
    """
    Base class for performance predictors.
    """
    def __init__(
        self, *args
    ):
        raise NotImplementedError

    def get_linr_T(self, S: int) -> float:
        """
        Get the linear time for iteration width S
        """
        raise NotImplementedError

    def get_pref_T(self, S: int) -> float:
        """
        Get the GPU prefilling time for iteration width S
        """
        raise NotImplementedError

    def get_gdec_T(self, N: int) -> float:
        """
        Get the GPU decoding time for number of tokens N
        """
        raise NotImplementedError
    
    def get_cdec_T(self, S: int, N: int) -> float:
        """
        Get the CPU decoding time for iteration width S and number of tokens N
        """
        raise NotImplementedError
    
    def get_lnch_T(self) -> float:
        """
        Get the kernel launch time
        """
        raise NotImplementedError

class ZeroPerfPredictor(PerfPredictor):
    """
    A dummy performance predictor that always returns 0.
    """
    def __init__(
        self, *args
    ):
        pass

    def get_linr_T(self, S: int) -> float:
        return 0.0

    def get_pref_T(self, S: int) -> float:
        return 0.0

    def get_gdec_T(self, N: int) -> float:
        return 0.0

    def get_cdec_T(self, S: int, N: int) -> float:
        return 0.0

    def get_lnch_T(self) -> float:
        return 0.0

class TablePerfPredictor(PerfPredictor):
    """
    A perfomance predictor that uses a table to store the performance data.

    It uses linear interpolation to estimate the performance for unseen data.
    """
    def __init__(
        self,
        engine_config: EngineConfig
    ):
        # Linr
        self.linr_S_list = list(range(1, 512)) + [
            2 ** i for i in range(
                9,
                (engine_config.max_tokens_in_batch - 1).bit_length()
            )
        ] + [engine_config.max_tokens_in_batch]
        self.linr_T_list = None
        self.linr_S_lb_idx = self._get_lb_idx_list(self.linr_S_list)
        self.linr_S_threshold = 128 # NOTE: This is a heuristic value

        # Pref
        self.pref_S_list = sum([[2 ** (i-2) * 3, 2 ** i] for i in range(
            (engine_config.block_size - 1).bit_length(),
            (engine_config.max_tokens_in_batch - 1).bit_length()
        )], []) + [engine_config.max_tokens_in_batch]
        self.pref_T_list = None
        self.pref_S_lb_idx = self._get_lb_idx_list(self.pref_S_list)

        # Gdec
        self.gdec_N_list = sum([[2 ** (i-2) * 3, 2 ** i] for i in range(
            (engine_config.block_size - 1).bit_length(),
            (engine_config.max_gpu_tokens - 1).bit_length()
        )], []) + [engine_config.max_gpu_tokens]
        self.gdec_T_list = None
        self.gdec_N_lb_idx = self._get_lb_idx_list(self.gdec_N_list)

        # Cdec
        self.cdec_S_list = [2 ** i for i in range(
            0,
            (engine_config.max_batch_size - 1).bit_length()
        )] + [engine_config.max_batch_size]
        self.cdec_N_lists = [
            [S * engine_config.block_size] + 
            [2 ** i for i in range(
                (S * engine_config.block_size).bit_length(),
                (min(S * engine_config.max_seq_len, engine_config.max_cpu_tokens) - 1).bit_length()
            )] +
            [min(S * engine_config.max_seq_len, engine_config.max_cpu_tokens)]
            for S in self.cdec_S_list      
        ]
        self.cdec_N_list_agg = sorted(list(set(sum(self.cdec_N_lists, []))))

        self.cdec_T_lists = [None]
        self.cdec_S_lb_idx = self._get_lb_idx_list(self.cdec_S_list)
        self.cdec_N_lb_idx = self._get_lb_idx_list(self.cdec_N_list_agg)
        
        # Lnch
        self.lnch_T = 0.8
        # self.lnch_T = self._profile_lnch(lnch_S_list)

    def _get_lb_idx_list(self, input_list: list[int]) -> list[int]:
        """
        Get the lower bound index list of x in the input list.

        Given i, find the smallest j s.t. input_list[j] >= i.
        """
        return sum(
            [[i+1] * (input_list[i+1] - input_list[i]) for i in range(len(input_list) - 1)],
            [0] * (input_list[0] + 1)
        )
    
    def _interp(self, x: int, x0: int, x1: int, y0: float, y1: float) -> float:
        """
        Linear interpolation of 2 points (x0, y0) and (x1, y1) at x.
        """
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def _interp_1d(self, x, xs: list[int], ys: list[float], x_lb_idx: list[int]) -> float:
        """
        Linear interpolation of 1D points (x_list, y_list) at x. Assume x <= x_list[-1].
        """
        assert x <= xs[-1], f"x={x} exceeds the maximum {xs[-1]}"
        if x == 0:
            return 0.0
        idx = x_lb_idx[x]
        if idx == 0 or x == xs[idx]:
            return ys[idx]
        return self._interp(x, xs[idx-1], xs[idx], ys[idx-1], ys[idx])

    def get_linr_T(self, S: int) -> float:
        """
        Get the linear time for iteration width S, using linear interpolation
        """
        return self._interp_1d(S, self.linr_S_list, self.linr_T_list, self.linr_S_lb_idx)
    
    def get_pref_T(self, S: int) -> float:
        """
        Get the GPU prefilling time for iteration width S, using linear interpolation
        """
        return self._interp_1d(S, self.pref_S_list, self.pref_T_list, self.pref_S_lb_idx)

    def get_gdec_T(self, N: int) -> float:
        """
        Get the GPU decoding time for number of tokens N, using linear interpolation
        """
        return self._interp_1d(N, self.gdec_N_list, self.gdec_T_list, self.gdec_N_lb_idx)

    def get_cdec_T(self, S: int, N: int) -> float:
        """
        Get the CPU decoding time for iteration width S and number of tokens N,
        using bilinear interpolation
        """
        assert S < len(self.cdec_S_lb_idx), f"CPU batch size {S} exceeds the maximum {len(self.cdec_S_lb_idx)}"
        if S == 0:
            return 0.0
        s_idx = self.cdec_S_lb_idx[S]
        if s_idx == 0 or S == self.cdec_S_list[s_idx]:
            return self._interp_1d(N, self.cdec_N_list_agg, self.cdec_T_lists[s_idx], self.cdec_N_lb_idx)
        s1 = self.cdec_S_list[s_idx]
        s0 = self.cdec_S_list[s_idx - 1]
        ts1 = self._interp_1d(N, self.cdec_N_list_agg, self.cdec_T_lists[s_idx], self.cdec_N_lb_idx)
        ts0 = self._interp_1d(N, self.cdec_N_list_agg, self.cdec_T_lists[s_idx - 1], self.cdec_N_lb_idx)    
        return self._interp(S, s0, s1, ts0, ts1)

    def get_lnch_T(self) -> float:
        return self.lnch_T
