#!/usr/bin/env python3
"""Unitree robot with cat motion using Genesis official locomotion approach."""

import numpy as np
import torch
import math
from typing import Dict
from pathlib import Path

# Apply igl patch before importing genesis
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


class UnitreeOfficialController:
    """Controller based on Genesis official locomotion example."""
    
    def __init__(self):
        # Initialize Genesis
        gs.init(backend=gs.cuda)
        
        self.scene = None
        self.robot = None
        self.motion_data = None
        self.frame_idx = 0
        
        # Control parameters from official example
        self.dt = 0.02  # 50Hz control frequency
        self.kp = 100.0  # Much higher position gain for stronger response
        self.kd = 5.0   # Higher velocity gain for better damping
        
        # Motion speed multiplier
        self.motion_speed = 3.0  # Speed up motion 3x to match GIF
        
        # Joint configuration - official Go2 naming and order
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",    # FR first
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",    # FL second
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",    # RR third
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",    # RL last
        ]
        
        # Default joint angles from official example (radians)
        self.default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,    # Front legs: 0.8
            "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,    # Front legs: 0.8
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 1.0,    # Rear legs: 1.0
            "RL_calf_joint": -1.5,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0,    # Rear legs: 1.0
            "RR_calf_joint": -1.5,
        }
        
        # Convert to tensor
        self.default_dof_pos = torch.tensor(
            [self.default_joint_angles[name] for name in self.joint_names],
            device=gs.device,
            dtype=gs.tc_float,
        )
        
    def create_scene(self):
        """Create scene following official example."""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
                gravity=(0, 0, -9.81),
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, -2.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=True,
            renderer=gs.renderers.Rasterizer(),
        )
        
        # Add ground plane
        self.ground = self.scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=1.0),
        )
        
    def load_unitree_robot(self):
        """Load robot following official example."""
        print("🤖 Loading Unitree Go2 (Genesis official style)...")
        
        # Base position and orientation from official
        base_init_pos = [0.0, 0.0, 0.42]
        base_init_quat = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
        
        # Load robot
        urdf_path = Path("go2.urdf")
        if urdf_path.exists():
            try:
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=str(urdf_path),
                        pos=base_init_pos,
                        quat=base_init_quat,
                    ),
                )
                print(f"✅ Robot loaded successfully!")
                
                # Build scene
                self.scene.build(n_envs=1)
                
                # Get motor indices
                self.motors_dof_idx = []
                for name in self.joint_names:
                    joint = self.robot.get_joint(name)
                    if joint is not None:
                        self.motors_dof_idx.append(joint.dof_start)
                        print(f"   ✓ Found joint '{name}' at DOF index {joint.dof_start}")
                    else:
                        # Try without _joint suffix
                        name_alt = name.replace("_joint", "")
                        joint = self.robot.get_joint(name_alt)
                        if joint is not None:
                            self.motors_dof_idx.append(joint.dof_start)
                            print(f"   ✓ Found joint '{name_alt}' at DOF index {joint.dof_start}")
                        else:
                            print(f"   ✗ Joint '{name}' not found!")
                
                print(f"✅ Found {len(self.motors_dof_idx)}/{len(self.joint_names)} motor joints")
                
                # Set PD gains
                self.robot.set_dofs_kp([self.kp] * len(self.motors_dof_idx), self.motors_dof_idx)
                self.robot.set_dofs_kv([self.kd] * len(self.motors_dof_idx), self.motors_dof_idx)
                
            except Exception as e:
                print(f"❌ Error loading robot: {e}")
                raise
        else:
            raise RuntimeError("URDF file not found")
    
    def apply_cat_motion(self, motion_data: Dict):
        """Apply cat motion using official control approach."""
        self.motion_data = motion_data
        y_motion = motion_data['y_normalized']
        amplitude = motion_data.get('amplitude', 0.2)
        frequency = motion_data.get('frequency', 0.15)
        
        print("\n🎮 Starting simulation (Genesis official style)...")
        print("   • PD control with proper gains")
        print("   • Official standing posture")
        print("   • Cat-like bouncing motion")
        print(f"   • Motion stats: amplitude={amplitude:.3f}, frequency={frequency:.3f}")
        print("   • Close viewer to exit\n")
        
        # Set initial pose
        self.robot.set_dofs_position(
            position=self.default_dof_pos,
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
        )
        
        # Let robot settle
        print("⏳ Letting robot settle...")
        for _ in range(50):
            self.scene.step()
        
        print("🏃 Starting cat motion...")
        
        # Control loop
        while self.scene.viewer.is_alive():
            # Get motion parameters with interpolation
            # Speed up frame progression
            motion_time = (self.frame_idx * self.motion_speed) % len(y_motion)
            motion_frame = int(motion_time)
            next_frame = (motion_frame + 1) % len(y_motion)
            
            # Linear interpolation between frames for smoother motion
            t_interp = motion_time - motion_frame
            bounce = y_motion[motion_frame] * (1 - t_interp) + y_motion[next_frame] * t_interp
            
            # Time for gait phase (also speed up)
            t = self.frame_idx * self.dt * self.motion_speed
            
            # Calculate target positions
            target_dof_pos = self.default_dof_pos.clone()
            
            # Apply cat-like motion to each leg
            # Order: FR(0), FL(1), RR(2), RL(3)
            for i in range(4):  # 4 legs
                hip_idx = i * 3
                thigh_idx = hip_idx + 1
                calf_idx = hip_idx + 2
                
                # Trotting gait phase (diagonal legs together)
                # FR(0) & RL(3) together, FL(1) & RR(2) together
                if i in [0, 3]:  # FR, RL
                    phase = 0
                else:  # FL, RR
                    phase = np.pi
                
                # Calculate motion phase
                leg_phase = t * 2 * np.pi + phase
                
                # Hip joint - minimal movement
                target_dof_pos[hip_idx] = self.default_dof_pos[hip_idx] + 0.1 * np.sin(leg_phase)
                
                # Thigh joint - main bounce driver
                # Scale motion based on detected amplitude - INCREASED
                thigh_motion = (0.5 + amplitude * 1.0) * bounce + 0.3
                target_dof_pos[thigh_idx] = self.default_dof_pos[thigh_idx] - thigh_motion * np.sin(leg_phase)
                
                # Calf joint - coordinate with thigh
                # Scale motion based on detected amplitude - INCREASED
                calf_motion = (0.6 + amplitude * 1.0) * bounce + 0.4
                target_dof_pos[calf_idx] = self.default_dof_pos[calf_idx] + calf_motion * np.sin(leg_phase - np.pi/4)
            
            # Apply position control
            self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
            
            # Step simulation
            self.scene.step()
            self.frame_idx += 1
            
            # Print progress
            if self.frame_idx % 50 == 0:
                pos = self.robot.get_pos()
                actual_pos = self.robot.get_dofs_position(self.motors_dof_idx)
                print(f"   Frame {self.frame_idx}, Height: {pos[0, 2]:.3f}m, Bounce: {bounce:.2f}")
                
                # Debug: print target vs actual positions for thigh joints
                if self.frame_idx % 200 == 0:
                    print(f"   Target thigh positions: {target_dof_pos[1]:.3f}, {target_dof_pos[4]:.3f}, {target_dof_pos[7]:.3f}, {target_dof_pos[10]:.3f}")
                    print(f"   Actual thigh positions: {actual_pos[0, 1]:.3f}, {actual_pos[0, 4]:.3f}, {actual_pos[0, 7]:.3f}, {actual_pos[0, 10]:.3f}")
                    print(f"   Difference: {abs(target_dof_pos[1] - actual_pos[0, 1]):.3f}, {abs(target_dof_pos[4] - actual_pos[0, 4]):.3f}")
        
        print("\n✨ Simulation completed!")


def run_official_simulation():
    """Run simulation using Genesis official approach."""
    import argparse
    from pathlib import Path
    from cat_meme_locomotion.core.motion_extractor import CatMotionExtractor
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unitree robot mimics cat motion from GIF")
    parser.add_argument(
        "--gif", 
        type=str, 
        default="assets/gifs/chipi-chipi-chapa-chapa.gif",
        help="Path to the GIF file (default: assets/gifs/chipi-chipi-chapa-chapa.gif)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=3.0,
        help="Motion speed multiplier (default: 3.0)"
    )
    args = parser.parse_args()
    
    # Validate GIF path
    gif_path = Path(args.gif)
    if not gif_path.exists():
        print(f"❌ Error: GIF file not found: {gif_path}")
        return
    
    print("🐱 Unitree Cat Motion (Genesis Official Style)")
    print("=" * 50)
    print(f"📁 GIF: {gif_path}")
    print(f"⚡ Speed: {args.speed}x")
    
    # Extract motion
    print("\n📊 Extracting cat motion...")
    extractor = CatMotionExtractor(str(gif_path))
    motion_data = extractor.extract_motion_pattern()
    
    if not motion_data:
        print("❌ Failed to extract motion")
        return
    
    print(f"✅ Extracted {motion_data['num_frames']} frames")
    print(f"✅ Found {len(motion_data['peaks'])} bounces")
    print(f"✅ Amplitude: {motion_data.get('amplitude', 0):.3f}")
    print(f"✅ Frequency: {motion_data.get('frequency', 0):.3f}")
    
    # Check if motion is too subtle
    if motion_data.get('amplitude', 0) < 0.05:
        print("⚠️  Warning: Very low amplitude detected. Motion might be subtle.")
    
    # Run simulation
    controller = UnitreeOfficialController()
    controller.motion_speed = args.speed  # Set speed from command line
    controller.create_scene()
    controller.load_unitree_robot()
    controller.apply_cat_motion(motion_data)


if __name__ == "__main__":
    run_official_simulation()