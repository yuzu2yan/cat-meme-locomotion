"""Cat Meme Locomotion - Replicate cat movements on Unitree robot."""

__version__ = "0.1.0"

from .core.motion_extractor import CatMotionExtractor
from .unitree_3d import UnitreeRobotController

__all__ = ["CatMotionExtractor", "UnitreeRobotController"]