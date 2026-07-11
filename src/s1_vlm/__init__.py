"""Stage-1 VLM LoRA fine-tune using Qwen3-VL."""

__all__ = ["S1VLMConfig", "run_s1_vlm"]


def __getattr__(name: str):
    if name == "S1VLMConfig":
        from src.s1_vlm.config import S1VLMConfig
        return S1VLMConfig
    if name == "run_s1_vlm":
        from src.s1_vlm.trainer import run_s1_vlm
        return run_s1_vlm
    raise AttributeError(name)
