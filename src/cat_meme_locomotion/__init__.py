"""Cat Meme Locomotion - Replicate cat movements on Unitree robot."""

__version__ = "0.1.0"

from .core.motion_extractor import CatMotionExtractor
from .unitree_genesis_official import UnitreeOfficialController

__all__ = ["CatMotionExtractor", "UnitreeOfficialController"]