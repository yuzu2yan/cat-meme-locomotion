#!/usr/bin/env python3
"""Unitree robot controller using YOLO pose estimation."""

import numpy as np
import torch
import math
from typing import Dict, List, Optional
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
from cat_meme_locomotion.core.simple_yolo_pose_extractor import SimpleYOLOPoseExtractor as ImprovedYOLOPoseExtractor, KeypointTrajectory


class UnitreeYOLOController:
    """Controller using YOLO pose estimation for motion control."""
    
    def __init__(self, model_name: str = 'yolov8x-pose.pt', use_animal_mapping: bool = True):
        # Initialize Genesis
        gs.init(backend=gs.cuda)
        
        self.scene = None
        self.robot = None
        self.motion_data = None
        self.keypoint_trajectories = None
        self.frame_idx = 0
        self.model_name = model_name
        self.use_animal_mapping = use_animal_mapping
        
        # Control parameters
        self.dt = 0.01  # 100Hz control frequency
        self.kp = 250.0  # High position gain
        self.kd = 25.0   # High damping
        
        # Motion parameters
        self.motion_speed = 1.0
        self.motion_amplitude = 1.2  # Amplify motion
        
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
        print("🤖 Loading Unitree Go2 with YOLO pose control...")
        
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
    
    def apply_yolo_motion(self, motion_data: Dict, keypoint_trajectories: Dict[str, KeypointTrajectory], 
                         animal_keypoint_trajectories: Optional[Dict[str, KeypointTrajectory]] = None):
        """Apply motion from YOLO pose estimation."""
        self.motion_data = motion_data
        self.keypoint_trajectories = keypoint_trajectories
        self.animal_keypoint_trajectories = animal_keypoint_trajectories or keypoint_trajectories
        
        print("\n🎮 Starting simulation with YOLO pose control...")
        print(f"   • Model: {self.model_name}")
        print(f"   • Detected keypoints: {len(animal_keypoint_trajectories or keypoint_trajectories)}")
        print(f"   • Using animal mapping: {self.use_animal_mapping}")
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
        trajectories_to_use = self.animal_keypoint_trajectories if self.use_animal_mapping else self.keypoint_trajectories
        max_frames = max(len(traj.positions) for traj in trajectories_to_use.values())
        
        # Pre-compute normalized trajectories with confidence weighting
        normalized_trajectories = {}
        for name, traj in trajectories_to_use.items():
            if traj.positions and len(traj.positions) > 0:
                positions = np.array(traj.positions)
                confidences = np.array(traj.confidences)
                
                # Only process keypoints with some valid detections
                valid_indices = confidences > 0.3
                if np.any(valid_indices):
                    # Find valid positions
                    valid_positions = positions[valid_indices]
                    
                    if len(valid_positions) > 0:
                        # Normalize to [0, 1] based on valid positions only
                        min_vals = np.min(valid_positions, axis=0)
                        max_vals = np.max(valid_positions, axis=0)
                        
                        # Avoid division by zero
                        range_vals = max_vals - min_vals
                        range_vals[range_vals < 1e-6] = 1.0
                        
                        # Normalize all positions
                        normalized = (positions - min_vals) / range_vals
                        
                        # Clip to [0, 1] range
                        normalized = np.clip(normalized, 0, 1)
                        
                        # Apply confidence weighting
                        normalized = normalized * confidences[:, np.newaxis]
                        
                        normalized_trajectories[name] = normalized
                        print(f"   ✓ {name}: {np.sum(valid_indices)} valid frames")
        
        # Control loop
        last_target = self.default_dof_pos.clone()
        
        while self.scene.viewer.is_alive():
            # Get current frame index with wrapping
            motion_frame = int(self.frame_idx * self.motion_speed) % max_frames
            
            # Calculate target positions
            target_dof_pos = self.default_dof_pos.clone()
            
            # Use a cyclic motion pattern as fallback
            phase = (motion_frame / max_frames) * 2 * np.pi
            
            if self.use_animal_mapping:
                # Use animal-specific keypoints
                self._apply_animal_motion(target_dof_pos, normalized_trajectories, motion_frame, phase)
            else:
                # Original human keypoint mapping
                self._apply_human_motion(target_dof_pos, normalized_trajectories, motion_frame, phase)
            
            # Add gait-specific adjustments
            gait_pattern = self.motion_data.get('gait_pattern', 'walk')
            if gait_pattern == 'trot':
                # Trot: diagonal pairs move together
                phase_offset = motion_frame * 0.1
                target_dof_pos[1] += 0.2 * np.sin(phase_offset)  # FR
                target_dof_pos[10] += 0.2 * np.sin(phase_offset)  # RL
                target_dof_pos[4] += 0.2 * np.sin(phase_offset + np.pi)  # FL
                target_dof_pos[7] += 0.2 * np.sin(phase_offset + np.pi)  # RR
            elif gait_pattern == 'hop':
                # Hop: all legs move together
                phase_offset = motion_frame * 0.15
                for i in [1, 4, 7, 10]:  # All thigh joints
                    target_dof_pos[i] += 0.3 * np.sin(phase_offset)
            
            # Smooth transition with momentum
            alpha = 0.7  # Higher alpha for smoother motion
            smoothed_target = alpha * target_dof_pos + (1 - alpha) * last_target
            last_target = smoothed_target.clone()
            
            # Apply position control
            self.robot.control_dofs_position(smoothed_target, self.motors_dof_idx)
            
            # Step simulation
            self.scene.step()
            self.frame_idx += 1
            
            # Print progress
            if self.frame_idx % 100 == 0:
                pos = self.robot.get_pos()
                vel = self.robot.get_vel()
                height = pos[0, 2].item() if hasattr(pos[0, 2], 'item') else pos[0, 2]
                vel_cpu = vel[0, :2].cpu().numpy() if hasattr(vel, 'cpu') else vel[0, :2]
                speed = np.linalg.norm(vel_cpu)
                print(f"   Frame {self.frame_idx}, Height: {height:.3f}m, Speed: {speed:.3f}m/s")
        
        print("\n✨ Simulation completed!")
    
    def _apply_animal_motion(self, target_dof_pos: torch.Tensor, normalized_trajectories: Dict, 
                           motion_frame: int, phase: float):
        """Apply motion using animal keypoint mapping."""
        # Front legs - use front paw keypoints
        if 'front_left_paw' in normalized_trajectories and motion_frame < len(normalized_trajectories['front_left_paw']):
            traj = normalized_trajectories['front_left_paw'][motion_frame]
            if np.sum(traj) > 0:
                y_norm = traj[1]
                # Natural quadruped motion
                target_dof_pos[4] = 0.3 + self.motion_amplitude * 0.8 * (1 - y_norm)  # FL_thigh
                target_dof_pos[5] = -2.0 + self.motion_amplitude * 0.6 * y_norm  # FL_calf
        else:
            # Fallback motion
            y_norm = 0.5 + 0.3 * np.sin(phase)
            target_dof_pos[4] = 0.3 + self.motion_amplitude * 0.5 * (1 - y_norm)
            target_dof_pos[5] = -1.8 + self.motion_amplitude * 0.3 * y_norm
        
        if 'front_right_paw' in normalized_trajectories and motion_frame < len(normalized_trajectories['front_right_paw']):
            traj = normalized_trajectories['front_right_paw'][motion_frame]
            if np.sum(traj) > 0:
                y_norm = traj[1]
                target_dof_pos[1] = 0.3 + self.motion_amplitude * 0.8 * (1 - y_norm)  # FR_thigh
                target_dof_pos[2] = -2.0 + self.motion_amplitude * 0.6 * y_norm  # FR_calf
        else:
            y_norm = 0.5 + 0.3 * np.sin(phase + np.pi)
            target_dof_pos[1] = 0.3 + self.motion_amplitude * 0.5 * (1 - y_norm)
            target_dof_pos[2] = -1.8 + self.motion_amplitude * 0.3 * y_norm
        
        # Back legs - use back paw keypoints
        if 'back_left_paw' in normalized_trajectories and motion_frame < len(normalized_trajectories['back_left_paw']):
            traj = normalized_trajectories['back_left_paw'][motion_frame]
            if np.sum(traj) > 0:
                y_norm = traj[1]
                target_dof_pos[10] = 0.5 + self.motion_amplitude * 1.0 * (1 - y_norm)  # RL_thigh
                target_dof_pos[11] = -2.0 + self.motion_amplitude * 0.6 * y_norm  # RL_calf
        else:
            y_norm = 0.5 + 0.3 * np.sin(phase + np.pi/2)
            target_dof_pos[10] = 0.5 + self.motion_amplitude * 0.7 * (1 - y_norm)
            target_dof_pos[11] = -1.8 + self.motion_amplitude * 0.3 * y_norm
        
        if 'back_right_paw' in normalized_trajectories and motion_frame < len(normalized_trajectories['back_right_paw']):
            traj = normalized_trajectories['back_right_paw'][motion_frame]
            if np.sum(traj) > 0:
                y_norm = traj[1]
                target_dof_pos[7] = 0.5 + self.motion_amplitude * 1.0 * (1 - y_norm)  # RR_thigh
                target_dof_pos[8] = -2.0 + self.motion_amplitude * 0.6 * y_norm  # RR_calf
        else:
            y_norm = 0.5 + 0.3 * np.sin(phase - np.pi/2)
            target_dof_pos[7] = 0.5 + self.motion_amplitude * 0.7 * (1 - y_norm)
            target_dof_pos[8] = -1.8 + self.motion_amplitude * 0.3 * y_norm
        
        # Hip motion based on spine/body keypoints
        if 'spine_center' in normalized_trajectories and motion_frame < len(normalized_trajectories['spine_center']):
            traj = normalized_trajectories['spine_center'][motion_frame]
            if np.sum(traj) > 0:
                x_norm = traj[0]
                # Subtle hip motion for balance
                hip_motion = 0.2 * (x_norm - 0.5)
                target_dof_pos[0] = hip_motion  # FR_hip
                target_dof_pos[3] = -hip_motion  # FL_hip
                target_dof_pos[6] = hip_motion  # RR_hip  
                target_dof_pos[9] = -hip_motion  # RL_hip
    
    def _apply_human_motion(self, target_dof_pos: torch.Tensor, normalized_trajectories: Dict, 
                          motion_frame: int, phase: float):
        """Apply motion using human keypoint mapping (original method)."""
        # Front legs - use wrist keypoints
        if 'left_wrist' in normalized_trajectories and motion_frame < len(normalized_trajectories['left_wrist']):
            traj = normalized_trajectories['left_wrist'][motion_frame]
            if np.sum(traj) > 0:
                y_norm = traj[1]
                target_dof_pos[4] = 0.2 + self.motion_amplitude * (1 - y_norm)  # FL_thigh
                target_dof_pos[5] = -2.2 + self.motion_amplitude * 0.5 * y_norm  # FL_calf
        else:
            y_norm = 0.5 + 0.3 * np.sin(phase)
            target_dof_pos[4] = 0.2 + self.motion_amplitude * 0.6 * (1 - y_norm)
            target_dof_pos[5] = -1.8 + self.motion_amplitude * 0.4 * y_norm
        
        if 'right_wrist' in normalized_trajectories and motion_frame < len(normalized_trajectories['right_wrist']):
            traj = normalized_trajectories['right_wrist'][motion_frame]
            if np.sum(traj) > 0:
                y_norm = traj[1]
                target_dof_pos[1] = 0.2 + self.motion_amplitude * (1 - y_norm)  # FR_thigh
                target_dof_pos[2] = -2.2 + self.motion_amplitude * 0.5 * y_norm  # FR_calf
        else:
            y_norm = 0.5 + 0.3 * np.sin(phase + np.pi)
            target_dof_pos[1] = 0.2 + self.motion_amplitude * 0.6 * (1 - y_norm)
            target_dof_pos[2] = -1.8 + self.motion_amplitude * 0.4 * y_norm
        
        # Rear legs - use ankle keypoints
        if 'left_ankle' in normalized_trajectories and motion_frame < len(normalized_trajectories['left_ankle']):
            traj = normalized_trajectories['left_ankle'][motion_frame]
            if np.sum(traj) > 0:
                y_norm = traj[1]
                target_dof_pos[10] = 0.4 + self.motion_amplitude * 1.2 * (1 - y_norm)  # RL_thigh
                target_dof_pos[11] = -2.2 + self.motion_amplitude * 0.5 * y_norm  # RL_calf
        else:
            y_norm = 0.5 + 0.3 * np.sin(phase + np.pi/2)
            target_dof_pos[10] = 0.4 + self.motion_amplitude * 0.8 * (1 - y_norm)
            target_dof_pos[11] = -1.8 + self.motion_amplitude * 0.4 * y_norm
        
        if 'right_ankle' in normalized_trajectories and motion_frame < len(normalized_trajectories['right_ankle']):
            traj = normalized_trajectories['right_ankle'][motion_frame]
            if np.sum(traj) > 0:
                y_norm = traj[1]
                target_dof_pos[7] = 0.4 + self.motion_amplitude * 1.2 * (1 - y_norm)  # RR_thigh
                target_dof_pos[8] = -2.2 + self.motion_amplitude * 0.5 * y_norm  # RR_calf
        else:
            y_norm = 0.5 + 0.3 * np.sin(phase - np.pi/2)
            target_dof_pos[7] = 0.4 + self.motion_amplitude * 0.8 * (1 - y_norm)
            target_dof_pos[8] = -1.8 + self.motion_amplitude * 0.4 * y_norm
        
        # Hip motion based on shoulder keypoints
        if 'left_shoulder' in normalized_trajectories and motion_frame < len(normalized_trajectories['left_shoulder']):
            traj = normalized_trajectories['left_shoulder'][motion_frame]
            if np.sum(traj) > 0:
                x_norm = traj[0]
                target_dof_pos[3] = 0.3 * (x_norm - 0.5)  # FL_hip
                target_dof_pos[9] = 0.3 * (x_norm - 0.5)  # RL_hip
        
        if 'right_shoulder' in normalized_trajectories and motion_frame < len(normalized_trajectories['right_shoulder']):
            traj = normalized_trajectories['right_shoulder'][motion_frame]
            if np.sum(traj) > 0:
                x_norm = traj[0]
                target_dof_pos[0] = 0.3 * (0.5 - x_norm)  # FR_hip (opposite)
                target_dof_pos[6] = 0.3 * (0.5 - x_norm)  # RR_hip


def run_yolo_simulation():
    """Run simulation with YOLO pose estimation."""
    import argparse
    from pathlib import Path
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unitree robot with YOLO pose estimation")
    parser.add_argument(
        "--gif", 
        type=str, 
        default="assets/gifs/chipi-chipi-chapa-chapa.gif",
        help="Path to the GIF or video file (mp4, avi, etc.)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x-pose.pt",
        help="YOLO model to use (yolov8n-pose.pt for faster, yolov8x-pose.pt for better)"
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
        default=1.2,
        help="Motion amplitude multiplier (default: 1.2)"
    )
    args = parser.parse_args()
    
    # Validate input file path
    input_path = Path(args.gif)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return
    
    # Check if it's a video file
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    is_video = input_path.suffix.lower() in video_extensions
    
    print("🐱 Unitree Cat Motion with YOLO Pose Estimation")
    print("=" * 50)
    print(f"📁 Input: {input_path}")
    print(f"📹 Type: {'Video' if is_video else 'GIF'}")
    print(f"🤖 Model: {args.model}")
    print(f"⚡ Speed: {args.speed}x")
    print(f"📏 Amplitude: {args.amplitude}x")
    
    # Extract motion with YOLO
    print("\n📊 Extracting motion with YOLO pose detection...")
    # For videos, limit frames to avoid memory issues
    if is_video:
        extractor = ImprovedYOLOPoseExtractor(str(input_path), model_name=args.model, max_frames=100, target_fps=10)
    else:
        extractor = ImprovedYOLOPoseExtractor(str(input_path), model_name=args.model)
    
    # Analyze motion
    motion_data = extractor.analyze_motion()
    
    if not motion_data:
        print("❌ Failed to extract motion")
        return
    
    print(f"✅ Analyzed {motion_data['num_frames']} frames")
    print(f"✅ Detected {len(motion_data['detected_keypoints'])} keypoints")
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Always generate tracking results
    print("\n📸 Generating motion capture and tracking results...")
    
    try:
        # 1. Static keypoint visualization
        output_path = f"outputs/yolo_keypoints_{input_path.stem}.png"
        extractor.visualize_keypoints(output_path)
        print(f"   ✓ Keypoint visualization saved")
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")
    
    try:
        # 2. Motion capture tracking GIF (always GIF output)
        gif_output_path = f"outputs/yolo_tracking_{input_path.stem}.gif"
        # For videos, limit frames to avoid huge GIF files
        if is_video:
            extractor.create_tracking_gif(gif_output_path, fps=10, max_frames=100)
        else:
            extractor.create_tracking_gif(gif_output_path)
        print(f"   ✓ Motion capture GIF saved: {gif_output_path}")
    except Exception as e:
        print(f"   ⚠️  Tracking GIF failed: {e}")
    
    # Skip metrics for now as method doesn't exist in improved extractor
    # try:
    #     # 3. Learning metrics and performance
    #     metrics_dir = f"outputs/yolo_metrics_{input_path.stem}"
    #     extractor.save_tracking_metrics(metrics_dir)
    #     print(f"   ✓ Learning metrics saved")
    # except Exception as e:
    #     print(f"   ⚠️  Metrics failed: {e}")
    
    print(f"\n📁 Results saved to: outputs/")
    
    # Run simulation
    controller = UnitreeYOLOController(model_name=args.model)
    controller.motion_speed = args.speed
    controller.motion_amplitude = args.amplitude
    controller.create_scene()
    controller.load_unitree_robot()
    # Pass both human and animal keypoint trajectories
    animal_trajectories = extractor.animal_keypoint_trajectories if hasattr(extractor, 'animal_keypoint_trajectories') else None
    controller.apply_yolo_motion(motion_data, extractor.keypoint_trajectories, animal_trajectories)


if __name__ == "__main__":
    run_yolo_simulation()