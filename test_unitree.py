#!/usr/bin/env python3
"""Test Unitree robot loading."""

import sys
sys.path.insert(0, 'src')

from cat_meme_locomotion.core.motion_extractor import CatMotionExtractor
from cat_meme_locomotion.unitree_3d import UnitreeRobotController

print("🐱 Testing Unitree Robot")
print("=" * 40)

# Extract motion
print("\n📊 Extracting cat motion...")
extractor = CatMotionExtractor('assets/gifs/chipi-chipi-chapa-chapa.gif')
motion_data = extractor.extract_motion_pattern()
print(f"✅ Extracted {motion_data['num_frames']} frames")

# Create controller
print("\n🤖 Creating controller...")
controller = UnitreeRobotController()
controller.create_scene()

print("\n📁 Loading Unitree robot...")
controller.load_unitree_robot()

print("\n✅ Robot loaded successfully!")
print(f"Robot DOFs: {controller.robot.n_dofs}")

print("\n🎮 Starting simulation...")
controller.apply_cat_motion(motion_data)