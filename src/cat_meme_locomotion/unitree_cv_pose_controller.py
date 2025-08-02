#!/usr/bin/env python3
"""Unitree robot controller using OpenCV-based animal pose estimation."""

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
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.cat_meme_locomotion.core.cv_animal_pose_extractor import CVAnimalPoseExtractor, KeypointTrajectory


class UnitreeCVPoseController:
    """Controller using OpenCV-based pose estimation for motion control."""
    
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
        self.kp = 300.0  # Very high position gain for responsive control
        self.kd = 30.0   # High damping
        
        # Motion parameters
        self.motion_speed = 1.0
        self.motion_amplitude = 1.5  # Amplify motion for better visibility
        
        # Joint configuration
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        
        # Default joint angles (standing pose)
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
        """Create simulation scene."""
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
        """Load Unitree robot."""
        print("🤖 Loading Unitree Go2 with CV-based pose control...")
        
        # Base position and orientation
        base_init_pos = [0.0, 0.0, 0.42]
        base_init_quat = [1.0, 0.0, 0.0, 0.0]
        
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
                
                print(f"✅ Found {len(self.motors_dof_idx)} motor joints")
                
                # Set PD gains
                self.robot.set_dofs_kp([self.kp] * len(self.motors_dof_idx), self.motors_dof_idx)
                self.robot.set_dofs_kv([self.kd] * len(self.motors_dof_idx), self.motors_dof_idx)
                
            except Exception as e:
                print(f"❌ Error loading robot: {e}")
                raise
        else:
            raise RuntimeError("URDF file not found")
    
    def apply_cv_motion(self, motion_data: Dict, keypoint_trajectories: Dict[str, KeypointTrajectory]):
        """Apply motion from CV-based pose estimation."""
        self.motion_data = motion_data
        self.keypoint_trajectories = keypoint_trajectories
        
        print("\n🎮 Starting simulation with CV-based pose control...")
        print(f"   • Method: OpenCV (SIFT + Contour Analysis)")
        print(f"   • Detected keypoints: {len(motion_data.get('detected_keypoints', []))}")
        print(f"   • Gait pattern: {motion_data.get('gait_pattern', 'unknown')}")
        print(f"   • Average confidence: {motion_data.get('avg_confidence', 0):.2f}")
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
        max_frames = max(len(traj.positions) for traj in keypoint_trajectories.values() if traj.positions)
        
        if max_frames == 0:
            print("❌ No valid trajectories found!")
            return
            
        print(f"   • Total frames: {max_frames}")
        print(f"   • Motion speed: {self.motion_speed}x")
        print(f"   • Motion amplitude: {self.motion_amplitude}x")
        
        # Pre-compute normalized trajectories
        normalized_trajectories = {}
        for name, traj in keypoint_trajectories.items():
            if traj.positions and len(traj.positions) > 0:
                positions = np.array(traj.positions)
                confidences = np.array(traj.confidences)
                
                # Normalize to [0, 1]
                min_vals = np.min(positions, axis=0)
                max_vals = np.max(positions, axis=0)
                
                # Avoid division by zero
                range_vals = max_vals - min_vals
                range_vals[range_vals < 1e-6] = 1.0
                
                normalized = (positions - min_vals) / range_vals
                
                # Store normalized positions without confidence weighting
                # Confidence will be used for filtering, not scaling
                normalized_trajectories[name] = normalized
        
        # Control loop
        last_target = self.default_dof_pos.clone()
        
        while self.scene.viewer.is_alive():
            # Get current frame index with wrapping
            motion_frame = int(self.frame_idx * self.motion_speed) % max_frames
            
            # Calculate target positions
            target_dof_pos = self.default_dof_pos.clone()
            
            # Front legs - use wrist/paw keypoints
            if 'left_wrist' in normalized_trajectories and motion_frame < len(normalized_trajectories['left_wrist']):
                traj = normalized_trajectories['left_wrist'][motion_frame]
                y_norm = traj[1]  # Y coordinate (vertical in image)
                # Amplified motion mapping
                target_dof_pos[4] = 0.0 + self.motion_amplitude * 1.2 * (1 - y_norm)  # FL_thigh
                target_dof_pos[5] = -2.5 + self.motion_amplitude * 0.8 * y_norm  # FL_calf
            
            if 'right_wrist' in normalized_trajectories and motion_frame < len(normalized_trajectories['right_wrist']):
                traj = normalized_trajectories['right_wrist'][motion_frame]
                y_norm = traj[1]
                target_dof_pos[1] = 0.0 + self.motion_amplitude * 1.2 * (1 - y_norm)  # FR_thigh
                target_dof_pos[2] = -2.5 + self.motion_amplitude * 0.8 * y_norm  # FR_calf
            
            # Rear legs - use ankle/paw keypoints
            if 'left_ankle' in normalized_trajectories and motion_frame < len(normalized_trajectories['left_ankle']):
                traj = normalized_trajectories['left_ankle'][motion_frame]
                y_norm = traj[1]
                # Rear legs typically have larger range of motion
                target_dof_pos[10] = 0.2 + self.motion_amplitude * 1.5 * (1 - y_norm)  # RL_thigh
                target_dof_pos[11] = -2.5 + self.motion_amplitude * 0.8 * y_norm  # RL_calf
            
            if 'right_ankle' in normalized_trajectories and motion_frame < len(normalized_trajectories['right_ankle']):
                traj = normalized_trajectories['right_ankle'][motion_frame]
                y_norm = traj[1]
                target_dof_pos[7] = 0.2 + self.motion_amplitude * 1.5 * (1 - y_norm)  # RR_thigh
                target_dof_pos[8] = -2.5 + self.motion_amplitude * 0.8 * y_norm  # RR_calf
            
            # Hip motion based on shoulder/body keypoints
            if 'left_shoulder' in normalized_trajectories and motion_frame < len(normalized_trajectories['left_shoulder']):
                traj = normalized_trajectories['left_shoulder'][motion_frame]
                x_norm = traj[0]  # X coordinate (lateral in image)
                # Add lateral hip motion
                target_dof_pos[3] = 0.4 * (x_norm - 0.5) * self.motion_amplitude  # FL_hip
                target_dof_pos[9] = 0.4 * (x_norm - 0.5) * self.motion_amplitude  # RL_hip
            
            if 'right_shoulder' in normalized_trajectories and motion_frame < len(normalized_trajectories['right_shoulder']):
                traj = normalized_trajectories['right_shoulder'][motion_frame]
                x_norm = traj[0]
                target_dof_pos[0] = 0.4 * (0.5 - x_norm) * self.motion_amplitude  # FR_hip (opposite)
                target_dof_pos[6] = 0.4 * (0.5 - x_norm) * self.motion_amplitude  # RR_hip
            
            # Add gait-specific patterns
            gait_pattern = self.motion_data.get('gait_pattern', 'walk')
            phase = motion_frame * 0.1  # Phase for cyclic motions
            
            if gait_pattern == 'trot':
                # Trot: diagonal pairs move together with phase shift
                target_dof_pos[1] += 0.3 * np.sin(phase) * self.motion_amplitude  # FR
                target_dof_pos[10] += 0.3 * np.sin(phase) * self.motion_amplitude  # RL
                target_dof_pos[4] += 0.3 * np.sin(phase + np.pi) * self.motion_amplitude  # FL
                target_dof_pos[7] += 0.3 * np.sin(phase + np.pi) * self.motion_amplitude  # RR
                
            elif gait_pattern == 'gallop':
                # Gallop: all legs move with similar timing
                gallop_offset = 0.4 * np.sin(phase * 1.5) * self.motion_amplitude
                for i in [1, 4, 7, 10]:  # All thigh joints
                    target_dof_pos[i] += gallop_offset
                    
            elif gait_pattern == 'pace':
                # Pace: same-side legs move together
                target_dof_pos[1] += 0.3 * np.sin(phase) * self.motion_amplitude  # FR
                target_dof_pos[7] += 0.3 * np.sin(phase) * self.motion_amplitude  # RR
                target_dof_pos[4] += 0.3 * np.sin(phase + np.pi) * self.motion_amplitude  # FL
                target_dof_pos[10] += 0.3 * np.sin(phase + np.pi) * self.motion_amplitude  # RL
            
            # Smooth transition with adaptive smoothing
            alpha = 0.8  # High alpha for smooth motion
            smoothed_target = alpha * target_dof_pos + (1 - alpha) * last_target
            last_target = smoothed_target.clone()
            
            # Apply position control
            self.robot.control_dofs_position(smoothed_target, self.motors_dof_idx)
            
            # Step simulation
            self.scene.step()
            self.frame_idx += 1
            
            # Print progress
            if self.frame_idx % 50 == 0:
                pos = self.robot.get_pos()
                vel = self.robot.get_vel()
                height = pos[0, 2].item() if hasattr(pos[0, 2], 'item') else pos[0, 2]
                vel_cpu = vel[0, :2].cpu().numpy() if hasattr(vel, 'cpu') else vel[0, :2]
                speed = np.linalg.norm(vel_cpu)
                print(f"   Frame {self.frame_idx}, Motion frame: {motion_frame}/{max_frames}, Height: {height:.3f}m, Speed: {speed:.3f}m/s")
        
        print("\n✨ Simulation completed!")


def run_cv_pose_simulation(gif_path=None, speed=None, amplitude=None, visualize=False):
    """Run simulation with CV-based pose estimation."""
    import argparse
    from pathlib import Path
    
    # If arguments are passed directly, use them
    if gif_path is not None:
        class Args:
            pass
        args = Args()
        args.gif = gif_path
        args.speed = speed if speed is not None else 1.0
        args.amplitude = amplitude if amplitude is not None else 1.5
        args.visualize = visualize
    else:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Unitree robot with OpenCV-based animal pose estimation")
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
            "--amplitude",
            type=float,
            default=1.5,
            help="Motion amplitude multiplier (default: 1.5)"
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
    
    print("🐱 Unitree Cat Motion with OpenCV Animal Pose Estimation")
    print("=" * 55)
    print(f"📁 GIF: {gif_path}")
    print(f"⚡ Speed: {args.speed}x")
    print(f"📏 Amplitude: {args.amplitude}x")
    print(f"🔧 Method: OpenCV (SIFT + Contour Analysis)")
    sys.stdout.flush()  # Force flush output
    
    # Extract motion with CV-based method
    print("\n📊 Extracting motion with CV-based pose detection...")
    sys.stdout.flush()
    
    try:
        extractor = CVAnimalPoseExtractor(str(gif_path))  # Works with both GIF and video files
    except Exception as e:
        print(f"❌ Error creating extractor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analyze motion
    motion_data = extractor.analyze_motion()
    
    if not motion_data:
        print("❌ Failed to extract motion")
        return
    
    print(f"✅ Analyzed {motion_data['num_frames']} frames")
    print(f"✅ Detected {len(motion_data['detected_keypoints'])} valid keypoints")
    sys.stdout.flush()
    
    # Always generate visualizations
    Path("outputs").mkdir(exist_ok=True)
    
    # Generate PNG visualization
    print("\n📈 Generating PNG visualization...")
    sys.stdout.flush()
    png_output_path = f"outputs/cv_pose_{gif_path.stem}.png"
    try:
        extractor.visualize_keypoints(png_output_path)
        print(f"📊 Keypoint tracking visualization saved to {png_output_path}")
    except Exception as e:
        print(f"❌ Error generating PNG: {e}")
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    
    # Generate tracking GIF
    print("\n🎬 Generating tracking GIF...")
    sys.stdout.flush()
    gif_output_path = f"outputs/cv_tracking_{gif_path.stem}.gif"
    try:
        extractor.create_tracking_gif(gif_output_path, fps=10, max_frames=100)
        print(f"✅ Tracking GIF saved to {gif_output_path}")
    except Exception as e:
        print(f"❌ Error generating GIF: {e}")
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    
    # Run simulation
    print("\n🤖 Starting simulation...")
    sys.stdout.flush()
    
    try:
        controller = UnitreeCVPoseController()
        controller.motion_speed = args.speed
        controller.motion_amplitude = args.amplitude
        
        print("📋 Creating scene...")
        sys.stdout.flush()
        controller.create_scene()
        
        print("🤖 Loading robot...")
        sys.stdout.flush()
        controller.load_unitree_robot()
        
        print("🎮 Applying motion...")
        sys.stdout.flush()
        controller.apply_cv_motion(motion_data, extractor.keypoint_trajectories)
    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    run_cv_pose_simulation()