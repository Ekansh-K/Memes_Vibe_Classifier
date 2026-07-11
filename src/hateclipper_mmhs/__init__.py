"""Hate-CLIPper-style Stage-1 for MMHS150K."""

__all__ = ["HateCLIPperConfig", "HateCLIPperMMHS", "run_hateclipper"]


def __getattr__(name: str):
    if name == "HateCLIPperConfig":
        from src.hateclipper_mmhs.config import HateCLIPperConfig
        return HateCLIPperConfig
    if name == "HateCLIPperMMHS":
        from src.hateclipper_mmhs.model import HateCLIPperMMHS
        return HateCLIPperMMHS
    if name == "run_hateclipper":
        from src.hateclipper_mmhs.trainer import run_hateclipper
        return run_hateclipper
    raise AttributeError(name)
