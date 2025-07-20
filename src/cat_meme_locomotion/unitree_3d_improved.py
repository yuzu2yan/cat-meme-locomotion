#!/usr/bin/env python3
"""Improved Unitree robot simulation with proper PD control and cat motion."""

import numpy as np
import sys
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


class ImprovedUnitreeController:
    """Improved controller with PD control for Unitree robot."""
    
    def __init__(self):
        # Initialize Genesis
        gs.init(backend=gs.cuda)
        
        self.scene = None
        self.robot = None
        self.motion_data = None
        self.frame_idx = 0
        
        # PD control parameters
        self.kp = 50.0  # Proportional gain
        self.kd = 1.0   # Derivative gain
        
        # Joint limits and defaults for Unitree Go2
        self.joint_defaults = {
            # Hip joints (abduction/adduction)
            0: 0.0, 3: 0.0, 6: 0.0, 9: 0.0,
            # Thigh joints 
            1: -0.8, 4: -0.8, 7: -0.8, 10: -0.8,
            # Calf joints
            2: 1.6, 5: 1.6, 8: 1.6, 11: 1.6,
        }
        
        # Joint limits (rad)
        self.joint_limits = {
            # Hip: [-0.863, 0.863]
            # Thigh: [-1.686, 4.501]  
            # Calf: [-2.818, -0.888]
        }
        
    def create_scene(self):
        """Create Genesis scene with physics settings."""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.005,  # Smaller timestep for stability
                gravity=(0, 0, -9.81),
                substeps=4,  # More substeps
                requires_grad=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, -2.0, 1.2),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=40,
                max_FPS=60,
            ),
            show_viewer=True,
            renderer=gs.renderers.Rasterizer(),
        )
        
        # Add ground with good friction
        self.ground = self.scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(
                friction=1.5,
            ),
        )
        
    def load_unitree_robot(self):
        """Load Unitree Go2 with proper settings."""
        print("🤖 Loading Unitree Go2 robot...")
        
        # Fix URDF paths
        self._fix_urdf_paths()
        
        # Load robot with initial height
        urdf_path = Path("go2_fixed.urdf")
        if urdf_path.exists():
            try:
                print(f"📁 Loading URDF from: {urdf_path}")
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=str(urdf_path),
                        pos=(0, 0, 0.4),  # Start higher
                        euler=(0, 0, 0),
                    ),
                )
                print(f"✅ Robot loaded with {self.robot.n_dofs} DOFs")
                
                # Store joint info
                self.n_dofs = self.robot.n_dofs
                
            except Exception as e:
                print(f"❌ Error loading URDF: {e}")
                raise
        else:
            raise RuntimeError("Fixed URDF not found")
    
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
    
    def apply_cat_motion(self, motion_data: Dict):
        """Apply cat motion with improved control."""
        self.motion_data = motion_data
        y_motion = motion_data['y_normalized']
        
        # Build scene
        self.scene.build()
        
        print("\n🎮 Starting improved simulation...")
        print("   • PD control for stable motion")
        print("   • Cat-like bouncing gait")
        print("   • Close viewer to exit\n")
        
        # Set initial pose
        self._set_initial_pose()
        
        # Store previous joint positions for derivative
        prev_qpos = self.robot.get_dofs_position()
        if hasattr(prev_qpos, 'cpu'):
            prev_qpos = prev_qpos.cpu().numpy().copy()
        else:
            prev_qpos = np.array(prev_qpos).copy()
        
        # Main loop
        while self.scene.viewer.is_alive():
            # Get motion parameters
            motion_idx = self.frame_idx % len(y_motion)
            bounce = y_motion[motion_idx]
            phase = self.frame_idx * 0.02  # Slower phase
            
            # Calculate target positions
            target_qpos = self._calculate_target_positions(bounce, phase)
            
            # Get current state
            current_qpos = self.robot.get_dofs_position()
            current_qvel = self.robot.get_dofs_velocity()
            
            # Convert tensors to numpy if needed
            if hasattr(current_qpos, 'cpu'):
                current_qpos = current_qpos.cpu().numpy()
            if hasattr(current_qvel, 'cpu'):
                current_qvel = current_qvel.cpu().numpy()
            
            # PD control
            torques = self._pd_control(target_qpos, current_qpos, current_qvel)
            
            # Apply torques
            self.robot.set_dofs_force(torques)
            
            # Also apply some position control for stability
            self.robot.set_dofs_position(target_qpos)
            
            # Step simulation
            self.scene.step()
            self.frame_idx += 1
            
        print("\n✨ Simulation completed!")
    
    def _set_initial_pose(self):
        """Set stable initial standing pose."""
        initial_qpos = np.zeros(self.n_dofs)
        
        # Apply default positions
        for joint_idx, value in self.joint_defaults.items():
            if joint_idx < self.n_dofs:
                initial_qpos[joint_idx] = value
        
        # Convert to appropriate type for Genesis
        self.robot.set_dofs_position(initial_qpos)
        self.robot.set_dofs_velocity(np.zeros(self.n_dofs))
        
        # Let robot settle
        print("⏳ Letting robot settle...")
        for _ in range(100):
            self.scene.step()
            
    def _calculate_target_positions(self, bounce: float, phase: float) -> np.ndarray:
        """Calculate target joint positions for cat motion."""
        target = np.zeros(self.n_dofs)
        
        # Start with default positions
        for joint_idx, value in self.joint_defaults.items():
            if joint_idx < self.n_dofs:
                target[joint_idx] = value
        
        # Apply cat-like motion pattern
        for leg_idx in range(4):
            hip_idx = leg_idx * 3
            thigh_idx = hip_idx + 1
            calf_idx = hip_idx + 2
            
            if calf_idx >= self.n_dofs:
                break
            
            # Trotting gait: diagonal legs move together
            # FL & RR together, FR & RL together
            if leg_idx in [0, 3]:  # FL, RR
                leg_phase = phase
            else:  # FR, RL
                leg_phase = phase + np.pi
            
            # Hip movement (minimal)
            target[hip_idx] = 0.05 * np.sin(leg_phase * 2)
            
            # Thigh movement - main driver of bounce
            thigh_amp = 0.3 * bounce + 0.1
            target[thigh_idx] = self.joint_defaults[thigh_idx] + thigh_amp * np.sin(leg_phase)
            
            # Calf movement - coordinate with thigh
            calf_amp = 0.4 * bounce + 0.1
            target[calf_idx] = self.joint_defaults[calf_idx] - calf_amp * np.sin(leg_phase - np.pi/4)
            
        return target
    
    def _pd_control(self, target_pos: np.ndarray, current_pos: np.ndarray, 
                    current_vel: np.ndarray) -> np.ndarray:
        """PD controller for joint torques."""
        # Position error
        pos_error = target_pos - current_pos
        
        # Velocity error (target velocity is 0)
        vel_error = -current_vel
        
        # PD control law
        torques = self.kp * pos_error + self.kd * vel_error
        
        # Torque limits (prevent damage)
        max_torque = 20.0
        torques = np.clip(torques, -max_torque, max_torque)
        
        return torques


def run_improved_simulation():
    """Run improved Unitree simulation."""
    from cat_meme_locomotion.core.motion_extractor import CatMotionExtractor
    
    print("🐱 Improved Unitree Cat Motion")
    print("=" * 40)
    
    # Extract motion
    print("\n📊 Extracting cat motion...")
    extractor = CatMotionExtractor('assets/gifs/chipi-chipi-chapa-chapa.gif')
    motion_data = extractor.extract_motion_pattern()
    
    if not motion_data:
        print("❌ Failed to extract motion")
        return
    
    print(f"✅ Extracted {motion_data['num_frames']} frames")
    print(f"✅ Found {len(motion_data['peaks'])} bounces")
    
    # Run simulation
    controller = ImprovedUnitreeController()
    controller.create_scene()
    controller.load_unitree_robot()
    controller.apply_cat_motion(motion_data)


if __name__ == "__main__":
    run_improved_simulation()