#!/usr/bin/env python3
"""Test joint movement diagnostics."""

import numpy as np
import time
from pathlib import Path
import sys
sys.path.insert(0, 'src')

# Apply Genesis patch
import igl
_original_signed_distance = getattr(igl, 'signed_distance', None)
if _original_signed_distance:
    def patched_signed_distance(query_points, verts, faces):
        try:
            result = _original_signed_distance(query_points, verts, faces)
            if isinstance(result, tuple) and len(result) > 3:
                return result[0], result[1], result[2]
            return result
        except:
            num_points = len(query_points)
            return (np.ones(num_points) * 0.1, 
                   np.zeros(num_points, dtype=np.int32), 
                   query_points.copy())
    igl.signed_distance = patched_signed_distance

import genesis as gs

print("🔍 Joint Movement Diagnostic Test")
print("=" * 40)

# Initialize Genesis
gs.init(backend=gs.cuda)

# Create minimal scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, -9.81),
        substeps=2,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, -1.5, 1.0),
        camera_lookat=(0.0, 0.0, 0.3),
        camera_fov=40,
        max_FPS=30,
    ),
    show_viewer=True,
    renderer=gs.renderers.Rasterizer(),
)

# Add ground
ground = scene.add_entity(
    gs.morphs.Plane(),
    material=gs.materials.Rigid(friction=1.0),
)

# Fix URDF paths if needed
import re
with open("go2.urdf", "r") as f:
    urdf_content = f.read()
current_dir = Path.cwd()
urdf_content = re.sub(
    r'filename="../dae/([^"]+)"',
    f'filename="{current_dir}/dae/\\1"',
    urdf_content
)
with open("go2_fixed.urdf", "w") as f:
    f.write(urdf_content)

# Load robot
robot = scene.add_entity(
    gs.morphs.URDF(
        file="go2_fixed.urdf",
        pos=(0, 0, 0.4),
        euler=(0, 0, 0),
    ),
)

print(f"✅ Robot loaded with {robot.n_dofs} DOFs")

# Build scene
scene.build()

# Print joint info
print("\n📋 Joint Information:")
print(f"   Total DOFs: {robot.n_dofs}")

# Expected joint mapping for Unitree Go2
joint_names = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf", 
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "head_joint1", "head_joint2",  # If present
]

# Set initial pose
initial_pos = np.zeros(robot.n_dofs)

# Standing position
for i in range(min(robot.n_dofs, 12)):
    if i % 3 == 0:  # Hip joints
        initial_pos[i] = 0.0
    elif i % 3 == 1:  # Thigh joints
        initial_pos[i] = -0.8
    elif i % 3 == 2:  # Calf joints
        initial_pos[i] = 1.6

robot.set_dofs_position(initial_pos)

print("\n🎮 Testing joint movement...")
print("   • Watch the robot's legs move")
print("   • Each joint will be tested")
print("   • Close viewer to exit\n")

frame = 0
test_joint = 0
direction = 1

while scene.viewer.is_alive():
    # Get current positions
    qpos = robot.get_dofs_position()
    
    # Test one joint at a time
    if frame % 60 == 0:  # Every second
        print(f"Testing joint {test_joint}: {joint_names[test_joint] if test_joint < len(joint_names) else 'unknown'}")
        test_joint = (test_joint + 1) % min(12, robot.n_dofs)  # Only test leg joints
    
    # Apply sinusoidal motion to current test joint
    phase = frame * 0.05
    if test_joint < robot.n_dofs:
        # Different amplitude based on joint type
        if test_joint % 3 == 0:  # Hip
            amplitude = 0.2
            qpos[test_joint] = amplitude * np.sin(phase)
        elif test_joint % 3 == 1:  # Thigh
            amplitude = 0.3
            qpos[test_joint] = -0.8 + amplitude * np.sin(phase)
        else:  # Calf
            amplitude = 0.4
            qpos[test_joint] = 1.6 + amplitude * np.sin(phase)
    
    # Apply positions
    robot.set_dofs_position(qpos)
    
    # Also print velocities to see if joints are actually moving
    if frame % 30 == 0:
        qvel = robot.get_dofs_velocity()
        moving_joints = np.where(np.abs(qvel) > 0.01)[0]
        if len(moving_joints) > 0:
            print(f"   Moving joints: {moving_joints.tolist()}")
    
    scene.step()
    frame += 1

print("\n✅ Test completed!")