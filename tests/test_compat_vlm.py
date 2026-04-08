from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "omlx" / "compat" / "vlm.py"
SPEC = importlib.util.spec_from_file_location("omlx_compat_vlm_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
vlm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vlm)


def test_vlm_wrapper_delegates_to_mlx_vlm(monkeypatch):
    calls: dict[str, object] = {}

    def fake_load(*args, **kwargs):
        calls["load"] = (args, kwargs)
        return "model", "processor"

    def fake_stream_generate(*args, **kwargs):
        calls["stream_generate"] = (args, kwargs)
        return iter(["chunk"])

    def fake_make_prompt_cache(model, max_kv_size=None):
        calls["make_prompt_cache"] = (model, max_kv_size)
        return ["cache"]

    def fake_quantize(*args, **kwargs):
        calls["quantize_prompt_cache"] = (args, kwargs)

    def fake_apply_chat_template(*args, **kwargs):
        calls["apply_chat_template"] = (args, kwargs)
        return "prompt"

    modules = {
        "mlx_vlm": SimpleNamespace(load=fake_load),
        "mlx_vlm.generate": SimpleNamespace(
            stream_generate=fake_stream_generate,
            maybe_quantize_kv_cache=fake_quantize,
        ),
        "mlx_vlm.models.cache": SimpleNamespace(
            make_prompt_cache=fake_make_prompt_cache
        ),
        "mlx_vlm.prompt_utils": SimpleNamespace(
            apply_chat_template=fake_apply_chat_template
        ),
    }

    def fake_import_module(name):
        return modules[name]

    monkeypatch.setattr(vlm, "import_module", fake_import_module)
    vlm._modules.cache_clear()

    assert vlm.load("repo", adapter_path="adapter", revision="rev", trust_remote_code=True) == (
        "model",
        "processor",
    )
    assert list(
        vlm.stream_generate(
            "model",
            "processor",
            "prompt",
            image=["img.png"],
            max_tokens=32,
        )
    ) == ["chunk"]
    assert vlm.make_prompt_cache("language_model", max_kv_size=256) == ["cache"]

    prompt_cache = ["cache"]
    vlm.quantize_prompt_cache(
        prompt_cache,
        quantized_kv_start=0,
        kv_group_size=64,
        kv_bits=4,
        kv_quant_scheme="turboquant",
    )
    assert vlm.apply_chat_template(
        "processor",
        "config",
        [{"role": "user", "content": "hi"}],
        num_images=1,
        enable_thinking=False,
    ) == "prompt"

    assert calls["load"] == (("repo",), {"adapter_path": "adapter", "revision": "rev", "trust_remote_code": True})
    assert calls["stream_generate"] == (
        ("model", "processor", "prompt"),
        {"image": ["img.png"], "max_tokens": 32},
    )
    assert calls["make_prompt_cache"] == ("language_model", 256)
    assert calls["quantize_prompt_cache"] == (
        (prompt_cache,),
        {
            "quantized_kv_start": 0,
            "kv_group_size": 64,
            "kv_bits": 4,
            "kv_quant_scheme": "turboquant",
        },
    )
    assert calls["apply_chat_template"] == (
        (
            "processor",
            "config",
            [{"role": "user", "content": "hi"}],
        ),
        {"num_images": 1, "enable_thinking": False},
    )


def test_vlm_wrapper_raises_clear_import_error(monkeypatch):
    def fake_import_module(name):
        raise ImportError(name)

    monkeypatch.setattr(vlm, "import_module", fake_import_module)
    vlm._modules.cache_clear()

    with pytest.raises(ImportError, match="oMLX VLM compatibility requires"):
        vlm.load("repo")
