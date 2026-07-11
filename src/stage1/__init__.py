"""Shared Stage-1 (binary hate) utilities."""

__all__ = [
    "evaluate_stage1",
    "sweep_threshold",
    "save_stage1_results",
    "Stage1Dataset",
    "stage1_collate_text",
    "stage1_collate_multimodal",
]


def __getattr__(name: str):
    if name in ("evaluate_stage1", "sweep_threshold", "save_stage1_results"):
        from src.stage1 import eval as _eval
        return getattr(_eval, name)
    if name in ("Stage1Dataset", "stage1_collate_text", "stage1_collate_multimodal"):
        from src.stage1 import dataset as _ds
        return getattr(_ds, name)
    raise AttributeError(name)
