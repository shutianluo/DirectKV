"""
DirectKV startup hook — registers the directkv attention backend in SGLang.

Executed at Python startup via directkv_startup.pth.

Supports two SGLang API generations:
  * ≥ 0.5.x  — uses attention_registry (register_attention_backend decorator)
  * 0.4.x    — patches ModelRunner._get_attention_backend_from_str and
                ModelRunner.init_memory_pool directly

The hook probes which API is present and applies whichever patch fits.
"""

import sys
import os
import builtins

_EDITABLE_SGLANG = "/sgl-workspace/sglang/python/sglang"
_backend_registered = [False]


def _merge_pkg_path(pkg_name, mod=None):
    if mod is None:
        mod = sys.modules.get(pkg_name)
    if mod is None:
        return
    pkg_path = getattr(mod, "__path__", None)
    if pkg_path is None:
        return
    parts = pkg_name.split(".")
    subpath = os.path.join(*parts[1:]) if len(parts) > 1 else ""
    editable = os.path.join(_EDITABLE_SGLANG, subpath) if subpath else _EDITABLE_SGLANG
    if not os.path.isdir(editable):
        return
    current = list(pkg_path)
    if editable not in current:
        current.append(editable)
        mod.__path__ = current


def _ensure_parents_fixed(dotted_name):
    parts = dotted_name.split(".")
    if parts[0] != "sglang":
        return
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name not in sys.modules:
            try:
                _orig_import(pkg_name, None, None, (), 0)
            except ImportError:
                return
        _merge_pkg_path(pkg_name)


_orig_import = builtins.__import__


def _directkv_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and isinstance(name, str) and name.split(".")[0] == "sglang":
        _ensure_parents_fixed(name)

    result = _orig_import(name, globals, locals, fromlist, level)

    if isinstance(name, str) and name.split(".")[0] == "sglang":
        _merge_pkg_path(name, sys.modules.get(name))
        pkg = name.rsplit(".", 1)[0] if "." in name else ""
        if pkg:
            _merge_pkg_path(pkg)
        for attr in fromlist or ():
            full = f"{name}.{attr}"
            if full in sys.modules:
                _merge_pkg_path(full)

    # Lazily register once sglang internals are loaded enough.
    if not _backend_registered[0]:
        _try_register()

    return result


builtins.__import__ = _directkv_import


def _try_register():
    """Attempt backend registration; silently skip if not ready yet."""
    # --- Path A: sglang ≥ 0.5 with attention_registry ---
    _reg_mod = sys.modules.get("sglang.srt.layers.attention.attention_registry")
    if _reg_mod is not None and hasattr(_reg_mod, "ATTENTION_BACKENDS"):
        _backend_registered[0] = True
        try:
            _register_via_registry(_reg_mod)
        except Exception:
            pass
        return

    # --- Path B: sglang 0.4.x with ModelRunner._get_attention_backend_from_str ---
    _mr_mod = sys.modules.get("sglang.srt.model_executor.model_runner")
    if _mr_mod is not None and hasattr(_mr_mod, "ModelRunner"):
        _backend_registered[0] = True
        try:
            _register_via_model_runner(_mr_mod)
        except Exception:
            pass
        return


# ---------------------------------------------------------------------------
# Path A: registry-based registration (sglang ≥ 0.5)
# ---------------------------------------------------------------------------

def _register_via_registry(reg_mod):
    ATTENTION_BACKENDS = reg_mod.ATTENTION_BACKENDS
    register_attention_backend = reg_mod.register_attention_backend

    if "directkv" not in ATTENTION_BACKENDS:
        @register_attention_backend("directkv")
        def _create_directkv_backend(runner):
            sa = runner.server_args
            if not sa.disable_radix_cache:
                raise ValueError("directkv requires --disable-radix-cache")
            if not sa.disable_cuda_graph:
                raise ValueError("directkv requires --disable-cuda-graph")
            if getattr(runner, "use_mla_backend", False):
                raise ValueError("directkv does not support MLA models")
            from sglang.srt.layers.attention.directkv_backend import DirectKVBackend
            return DirectKVBackend(runner)

    if "directkv-smpv2" not in ATTENTION_BACKENDS:
        @register_attention_backend("directkv-smpv2")
        def _create_directkv_smpv2_backend(runner):
            sa = runner.server_args
            if not sa.disable_radix_cache:
                raise ValueError("directkv-smpv2 requires --disable-radix-cache")
            if getattr(runner, "use_mla_backend", False):
                raise ValueError("directkv-smpv2 does not support MLA models")
            from sglang.srt.layers.attention.directkv_smpv2_backend import DirectKVSmpV2Backend
            return DirectKVSmpV2Backend(runner)

    # Also add to ATTENTION_BACKEND_CHOICES for argparse
    try:
        import sglang.srt.server_args as _sa
        choices = getattr(_sa, "ATTENTION_BACKEND_CHOICES", None)
        if choices is not None:
            if "directkv" not in choices:
                choices.append("directkv")
            if "directkv-smpv2" not in choices:
                choices.append("directkv-smpv2")
    except ImportError:
        pass

    # Patch _init_pools if present (sglang ≥ 0.5 name)
    try:
        import sglang.srt.model_executor.model_runner_kv_cache_mixin as _mixin
        cls = _mixin.ModelRunnerKVCacheMixin
        if not getattr(cls._init_pools, "_directkv_patched", False):
            _orig = cls._init_pools

            def _patched(self, *args, **kwargs):
                backend = getattr(self.server_args, "attention_backend", None)
                if backend in ("directkv", "directkv-smpv2"):
                    _init_directkv_pools(self)
                else:
                    _orig(self, *args, **kwargs)

            _patched._directkv_patched = True
            cls._init_pools = _patched
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Path B: direct ModelRunner patching (sglang 0.4.x)
# ---------------------------------------------------------------------------

def _register_via_model_runner(mr_mod):
    ModelRunner = mr_mod.ModelRunner

    # 1. Patch _get_attention_backend_from_str
    orig_get = getattr(ModelRunner, "_get_attention_backend_from_str", None)
    if orig_get is not None and not getattr(orig_get, "_directkv_patched", False):
        def _patched_get_backend(self, backend_str):
            if backend_str == "directkv":
                sa = self.server_args
                if not sa.disable_radix_cache:
                    raise ValueError(
                        "directkv requires --disable-radix-cache. "
                        "Add this flag when launching the server."
                    )
                if not sa.disable_cuda_graph:
                    raise ValueError(
                        "directkv requires --disable-cuda-graph. "
                        "Add this flag when launching the server."
                    )
                if getattr(self, "use_mla_backend", False):
                    raise ValueError("directkv does not support MLA models.")
                from sglang.srt.layers.attention.directkv_backend import DirectKVBackend
                import logging
                logging.getLogger(__name__).info("Using DirectKV attention backend (CPU-pinned KV cache)")
                return DirectKVBackend(self)
            if backend_str == "directkv-smpv2":
                sa = self.server_args
                if not sa.disable_radix_cache:
                    raise ValueError(
                        "directkv-smpv2 requires --disable-radix-cache."
                    )
                if getattr(self, "use_mla_backend", False):
                    raise ValueError("directkv-smpv2 does not support MLA models.")
                from sglang.srt.layers.attention.directkv_smpv2_backend import DirectKVSmpV2Backend
                import logging
                logging.getLogger(__name__).info(
                    "Using DirectKV-SMPv2 backend (Track B — fused projection)"
                )
                return DirectKVSmpV2Backend(self)
            return orig_get(self, backend_str)

        _patched_get_backend._directkv_patched = True
        ModelRunner._get_attention_backend_from_str = _patched_get_backend

    # 2. Patch init_memory_pool to use DirectKVTokenToKVPool when backend == directkv
    orig_init_pool = getattr(ModelRunner, "init_memory_pool", None)
    if orig_init_pool is not None and not getattr(orig_init_pool, "_directkv_patched", False):
        def _patched_init_pool(self, total_gpu_memory, max_num_reqs=None, max_total_tokens=None):
            orig_init_pool(self, total_gpu_memory, max_num_reqs, max_total_tokens)
            backend = getattr(self.server_args, "attention_backend", None)
            if backend in ("directkv", "directkv-smpv2"):
                _replace_pool_with_directkv(self)

        _patched_init_pool._directkv_patched = True
        ModelRunner.init_memory_pool = _patched_init_pool


def _replace_pool_with_directkv(runner):
    """Replace the GPU KV pool with a CPU-pinned DirectKVTokenToKVPool."""
    import logging
    import torch
    from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.layers.dp_attention import get_attention_tp_size

    sa = runner.server_args
    tp = get_attention_tp_size()
    num_kv_heads = runner.model_config.get_num_kv_heads(tp)
    head_dim = runner.model_config.head_dim
    v_head_dim = getattr(runner.model_config, "v_head_dim", None) or head_dim

    runner.token_to_kv_pool = DirectKVTokenToKVPool(
        size=runner.max_total_num_tokens,
        page_size=runner.page_size,
        dtype=runner.kv_cache_dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        layer_num=runner.num_effective_layers,
        v_head_dim=v_head_dim,
        start_layer=runner.start_layer,
        end_layer=runner.end_layer,
    )

    runner.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        runner.max_total_num_tokens,
        dtype=runner.kv_cache_dtype,
        device=runner.device,
        kvcache=runner.token_to_kv_pool,
    )

    logging.getLogger(__name__).info(
        f"DirectKV pool: {runner.max_total_num_tokens} tokens in CPU-pinned memory"
    )


# ---------------------------------------------------------------------------
# Shared pool init (used by Path A's registry route if it needs it)
# ---------------------------------------------------------------------------

def _init_directkv_pools(runner):
    """Initialise memory pools for the directkv backend (sglang ≥ 0.5 path)."""
    import torch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.directkv_pool import DirectKVTokenToKVPool
    try:
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    except ImportError:
        TokenToKVPoolAllocator = None
    from sglang.srt.layers.dp_attention import get_attention_tp_size

    sa = runner.server_args
    max_num_reqs = runner.max_running_requests

    if runner.req_to_token_pool is None:
        runner.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=runner.model_config.context_len + 4,
            device=runner.device,
            enable_memory_saver=sa.enable_memory_saver,
        )

    tp = get_attention_tp_size()
    num_kv_heads = runner.model_config.get_num_kv_heads(tp)
    head_dim = runner.model_config.head_dim
    v_head_dim = getattr(runner.model_config, "v_head_dim", None) or head_dim

    runner.token_to_kv_pool = DirectKVTokenToKVPool(
        size=runner.max_total_num_tokens,
        page_size=runner.page_size,
        dtype=runner.kv_cache_dtype,
        head_num=num_kv_heads,
        head_dim=head_dim,
        layer_num=runner.num_effective_layers,
        v_head_dim=v_head_dim,
        start_layer=runner.start_layer,
        end_layer=runner.end_layer,
    )

    if TokenToKVPoolAllocator is not None:
        runner.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=runner.max_total_num_tokens,
            dtype=runner.kv_cache_dtype,
            device=runner.device,
            kvcache=runner.token_to_kv_pool,
            need_sort=False,
        )
