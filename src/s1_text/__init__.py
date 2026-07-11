"""Stage-1 full fine-tune text classifiers."""

__all__ = ["S1TextConfig", "TextHateClassifier", "run_s1_text"]


def __getattr__(name: str):
    if name == "S1TextConfig":
        from src.s1_text.config import S1TextConfig
        return S1TextConfig
    if name == "TextHateClassifier":
        from src.s1_text.model import TextHateClassifier
        return TextHateClassifier
    if name == "run_s1_text":
        from src.s1_text.trainer import run_s1_text
        return run_s1_text
    raise AttributeError(name)
