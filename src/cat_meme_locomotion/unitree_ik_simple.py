#!/usr/bin/env python3
"""Simplified Unitree robot controller using direct joint angle mapping."""

import numpy as np
import torch
import math
from typing import Dict, List
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
from cat_meme_locomotion.core.cv_animal_pose_extractor import CVAnimalPoseExtractor
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class KeypointTrajectory:
    """Simple keypoint trajectory data."""
    positions: List[Tuple[float, float]]
    confidences: List[float]


class UnitreeSimpleIKController:
    """Simplified controller mapping keypoints directly to joint angles."""
    
    def __init__(self):
        # Initialize Genesis
        gs.init(backend=gs.cuda)
        
        self.scene = None
        self.robot = None
        self.motion_data = None
        self.keypoint_trajectories = None
        self.frame_idx = 0
        
        # Control parameters
        self.dt = 0.01  # 100Hz control frequency
        self.kp = 200.0  # Very high gain for responsive control
        self.kd = 20.0   # High damping
        
        # Motion speed multiplier
        self.motion_speed = 1.0
        
        # Joint configuration
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        
        # Default joint angles
        self.default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }
        
        # Convert to tensor
        self.default_dof_pos = torch.tensor(
            [self.default_joint_angles[name] for name in self.joint_names],
            device=gs.device,
            dtype=gs.tc_float,
        )
        
    def create_scene(self):
        """Create scene."""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=4,
                gravity=(0, 0, -9.81),
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / self.dt),
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
        """Load robot."""
        print("🤖 Loading Unitree Go2 with simplified keypoint mapping...")
        
        # Fix URDF paths
        self._fix_urdf_paths()
        
        # Base position and orientation
        base_init_pos = [0.0, 0.0, 0.42]
        base_init_quat = [1.0, 0.0, 0.0, 0.0]
        
        # Load robot
        urdf_path = Path("go2_fixed.urdf")
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
                
                print(f"✅ Found {len(self.motors_dof_idx)} motor joints")
                
                # Set PD gains
                self.robot.set_dofs_kp([self.kp] * len(self.motors_dof_idx), self.motors_dof_idx)
                self.robot.set_dofs_kv([self.kd] * len(self.motors_dof_idx), self.motors_dof_idx)
                
            except Exception as e:
                print(f"❌ Error loading robot: {e}")
                raise
        else:
            raise RuntimeError("URDF file not found")
    
    def _fix_urdf_paths(self):
        """Fix mesh paths in URDF."""
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
    
    def apply_simple_motion(self, motion_data: Dict, keypoint_trajectories: Dict[str, KeypointTrajectory]):
        """Apply motion using simplified keypoint to joint mapping."""
        self.motion_data = motion_data
        self.keypoint_trajectories = keypoint_trajectories
        
        print("\n🎮 Starting simulation with simplified keypoint mapping...")
        print(f"   • Direct keypoint to joint angle mapping")
        print(f"   • Detected keypoints: {len(keypoint_trajectories)}")
        print("   • Close viewer to exit\n")
        
        # Set initial pose
        self.robot.set_dofs_position(
            position=self.default_dof_pos,
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
        )
        
        # Let robot settle
        print("⏳ Letting robot settle...")
        for _ in range(100):
            self.scene.step()
        
        print("🏃 Starting motion...")
        
        # Get max frames
        max_frames = max(len(traj.positions) for traj in keypoint_trajectories.values())
        
        # Pre-compute normalized trajectories
        normalized_trajectories = {}
        for name, traj in keypoint_trajectories.items():
            positions = np.array(traj.positions)
            # Normalize to [0, 1]
            min_vals = np.min(positions, axis=0)
            max_vals = np.max(positions, axis=0)
            normalized = (positions - min_vals) / (max_vals - min_vals + 1e-6)
            normalized_trajectories[name] = normalized
        
        # Control loop
        while self.scene.viewer.is_alive():
            # Get current frame index with wrapping
            motion_frame = int(self.frame_idx * self.motion_speed) % max_frames
            
            # Calculate target positions
            target_dof_pos = self.default_dof_pos.clone()
            
            # Simple mapping: use vertical motion of keypoints to drive joints
            
            # Front legs - use wrist keypoints
            if 'left_wrist' in normalized_trajectories and motion_frame < len(normalized_trajectories['left_wrist']):
                y_norm = normalized_trajectories['left_wrist'][motion_frame][1]  # Y coordinate
                # Map to thigh and calf joints
                target_dof_pos[4] = 0.3 + 1.0 * (1 - y_norm)  # FL_thigh: more bend when keypoint is up
                target_dof_pos[5] = -2.0 + 0.8 * y_norm      # FL_calf: extend when keypoint is down
            
            if 'right_wrist' in normalized_trajectories and motion_frame < len(normalized_trajectories['right_wrist']):
                y_norm = normalized_trajectories['right_wrist'][motion_frame][1]
                target_dof_pos[1] = 0.3 + 1.0 * (1 - y_norm)  # FR_thigh
                target_dof_pos[2] = -2.0 + 0.8 * y_norm       # FR_calf
            
            # Rear legs - use ankle keypoints
            if 'left_ankle' in normalized_trajectories and motion_frame < len(normalized_trajectories['left_ankle']):
                y_norm = normalized_trajectories['left_ankle'][motion_frame][1]
                target_dof_pos[10] = 0.5 + 1.2 * (1 - y_norm)  # RL_thigh
                target_dof_pos[11] = -2.0 + 0.8 * y_norm       # RL_calf
            
            if 'right_ankle' in normalized_trajectories and motion_frame < len(normalized_trajectories['right_ankle']):
                y_norm = normalized_trajectories['right_ankle'][motion_frame][1]
                target_dof_pos[7] = 0.5 + 1.2 * (1 - y_norm)   # RR_thigh
                target_dof_pos[8] = -2.0 + 0.8 * y_norm        # RR_calf
            
            # Add some hip motion based on body keypoints
            if 'left_shoulder' in normalized_trajectories and motion_frame < len(normalized_trajectories['left_shoulder']):
                x_norm = normalized_trajectories['left_shoulder'][motion_frame][0]
                target_dof_pos[3] = 0.2 * (x_norm - 0.5)  # FL_hip
                target_dof_pos[9] = 0.2 * (x_norm - 0.5)  # RL_hip
            
            if 'right_shoulder' in normalized_trajectories and motion_frame < len(normalized_trajectories['right_shoulder']):
                x_norm = normalized_trajectories['right_shoulder'][motion_frame][0]
                target_dof_pos[0] = 0.2 * (0.5 - x_norm)  # FR_hip (opposite)
                target_dof_pos[6] = 0.2 * (0.5 - x_norm)  # RR_hip
            
            # Apply smooth transition
            alpha = 0.5  # Smoothing factor
            current_pos = self.robot.get_dofs_position(self.motors_dof_idx)[0]
            smoothed_target = alpha * target_dof_pos + (1 - alpha) * current_pos
            
            # Apply position control
            self.robot.control_dofs_position(smoothed_target, self.motors_dof_idx)
            
            # Step simulation
            self.scene.step()
            self.frame_idx += 1
            
            # Print progress
            if self.frame_idx % 100 == 0:
                pos = self.robot.get_pos()
                print(f"   Frame {self.frame_idx}, Height: {pos[0, 2]:.3f}m")
        
        print("\n✨ Simulation completed!")


def run_simple_ik_simulation():
    """Run simulation with simplified keypoint mapping."""
    import argparse
    from pathlib import Path
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unitree robot with simplified keypoint mapping")
    parser.add_argument(
        "--gif", 
        type=str, 
        default="assets/gifs/chipi-chipi-chapa-chapa.gif",
        help="Path to the GIF file"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Motion speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save keypoint tracking visualization"
    )
    args = parser.parse_args()
    
    # Validate GIF path
    gif_path = Path(args.gif)
    if not gif_path.exists():
        print(f"❌ Error: GIF file not found: {gif_path}")
        return
    
    print("🐱 Unitree Cat Motion with Simplified Keypoint Mapping")
    print("=" * 50)
    print(f"📁 GIF: {gif_path}")
    print(f"⚡ Speed: {args.speed}x")
    
    # Extract motion with DeepLabCut
    print("\n📊 Extracting motion with CV-based keypoint detection...")
    extractor = CVAnimalPoseExtractor(args.gif)
    
    # Load GIF frames
    from cat_meme_locomotion.utils.gif_loader import GifLoader
    loader = GifLoader()
    frames = loader.load_gif(str(gif_path))
    
    if not frames:
        print("❌ Failed to load GIF")
        return
    
    # Extract keypoints from all frames
    all_keypoints = []
    for frame in frames:
        keypoints = extractor.extract_keypoints(frame)
        all_keypoints.append(keypoints)
    
    # Convert to trajectories
    keypoint_trajectories = {}
    if all_keypoints:
        # Get all unique keypoint names
        all_parts = set()
        for kps in all_keypoints:
            all_parts.update(kps.keys())
        
        # Build trajectories for each part
        for part in all_parts:
            positions = []
            confidences = []
            for kps in all_keypoints:
                if part in kps and kps[part]:
                    positions.append(kps[part][0])
                    confidences.append(1.0)
                else:
                    # Use last known position or center
                    if positions:
                        positions.append(positions[-1])
                    else:
                        positions.append((frames[0].shape[1]//2, frames[0].shape[0]//2))
                    confidences.append(0.0)
            keypoint_trajectories[part] = KeypointTrajectory(positions=positions, confidences=confidences)
    
    motion_data = {'frames': len(frames), 'fps': 10, 'num_frames': len(frames), 'detected_keypoints': list(keypoint_trajectories.keys())}
    
    print(f"✅ Analyzed {motion_data['num_frames']} frames")
    print(f"✅ Detected {len(motion_data['detected_keypoints'])} keypoints")
    
    # Visualize if requested
    if args.visualize:
        output_path = f"outputs/simple_keypoints_{gif_path.stem}.gif"
        Path("outputs").mkdir(exist_ok=True)
        extractor.create_labeled_gif(frames, all_keypoints, output_path)
        print(f"\n📊 Keypoint tracking visualization saved to {output_path}")
    
    # Run simulation
    controller = UnitreeSimpleIKController()
    controller.motion_speed = args.speed
    controller.create_scene()
    controller.load_unitree_robot()
    controller.apply_simple_motion(motion_data, keypoint_trajectories)


if __name__ == "__main__":
    run_simple_ik_simulation()