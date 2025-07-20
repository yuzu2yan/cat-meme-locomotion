#!/usr/bin/env python3
"""Main entry point for Cat-to-Unitree motion replication."""

import sys
from pathlib import Path
from typing import Optional

from cat_meme_locomotion.core.motion_extractor import CatMotionExtractor
from cat_meme_locomotion.unitree_3d import UnitreeRobotController


def main(
    gif_path: Optional[str] = None,
    urdf_path: Optional[str] = None,
) -> None:
    """
    Main function to run cat motion replication on Unitree robot.
    
    Args:
        gif_path: Path to the cat GIF file
        urdf_path: Path to Unitree URDF file
    """
    print("🐱 Cat Meme to Unitree Robot Motion Replicator")
    print("=" * 45)
    
    # Use default paths if not provided
    if not gif_path:
        gif_path = "assets/gifs/chipi-chipi-chapa-chapa.gif"
    if not urdf_path:
        urdf_path = "assets/models/unitree/go1.urdf"
    
    gif_path = Path(gif_path)
    
    # Check if files exist
    if not gif_path.exists():
        print(f"❌ Error: GIF file not found at {gif_path}")
        sys.exit(1)
    
    # Extract motion from GIF
    print("\n📊 Extracting cat motion from GIF...")
    extractor = CatMotionExtractor(str(gif_path))
    motion_data = extractor.extract_motion_pattern()
    
    if not motion_data:
        print("❌ Error: Could not extract motion data from GIF")
        sys.exit(1)
    
    print(f"✅ Extracted {motion_data['num_frames']} frames")
    print(f"✅ Detected {len(motion_data['peaks'])} bounce peaks")
    print(f"✅ Bounce frequency: {motion_data['frequency']:.2f} per frame")
    
    # Run 3D simulation
    print("\n🤖 Starting Unitree robot simulation...")
    
    try:
        controller = UnitreeRobotController()
        controller.create_scene()
        controller.load_unitree_robot()
        controller.apply_cat_motion(motion_data)
        
    except Exception as e:
        print(f"\n❌ Error running simulation: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a GPU with CUDA support")
        print("2. Check that genesis-world is properly installed")
        print("3. Try running with: CUDA_VISIBLE_DEVICES=0 uv run cat-unitree")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()