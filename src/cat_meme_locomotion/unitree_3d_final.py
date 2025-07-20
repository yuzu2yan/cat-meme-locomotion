#!/usr/bin/env python3
"""Final Unitree robot simulation with proper joint control."""

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


class FinalUnitreeController:
    """Final controller for Unitree robot with cat motion."""
    
    def __init__(self):
        # Initialize Genesis
        gs.init(backend=gs.cuda)
        
        self.scene = None
        self.robot = None
        self.motion_data = None
        self.frame_idx = 0
        
        # Joint configuration for Go2
        # 12 leg joints + 6 other joints (head/tail if present)
        self.leg_joint_indices = list(range(12))  # First 12 are leg joints
        
        # Default standing position (radians)
        self.standing_pose = {
            # Hip joints (abduction/adduction) - neutral
            0: 0.0, 3: 0.0, 6: 0.0, 9: 0.0,
            # Thigh joints - bent for standing
            1: -0.9, 4: -0.9, 7: -0.9, 10: -0.9,
            # Calf joints - bent for standing  
            2: 1.8, 5: 1.8, 8: 1.8, 11: 1.8,
        }
        
    def create_scene(self):
        """Create Genesis scene."""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0, 0, -9.81),
                substeps=2,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, -2.0, 1.0),
                camera_lookat=(0.0, 0.0, 0.35),
                camera_fov=40,
                max_FPS=60,
            ),
            show_viewer=True,
            renderer=gs.renderers.Rasterizer(),
        )
        
        # Add ground
        self.ground = self.scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=1.0),
        )
        
    def load_unitree_robot(self):
        """Load Unitree Go2 robot."""
        print("🤖 Loading Unitree Go2 robot...")
        
        # Fix URDF paths
        self._fix_urdf_paths()
        
        # Load robot
        urdf_path = Path("go2_fixed.urdf")
        if urdf_path.exists():
            try:
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=str(urdf_path),
                        pos=(0, 0, 0.35),
                        euler=(0, 0, 0),
                    ),
                )
                print(f"✅ Robot loaded with {self.robot.n_dofs} DOFs")
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
        """Apply cat motion to robot."""
        self.motion_data = motion_data
        y_motion = motion_data['y_normalized']
        
        # Build scene
        self.scene.build()
        
        print("\n🎮 Starting final simulation...")
        print("   • Direct position control")
        print("   • Cat-like bouncing motion")
        print("   • Close viewer to exit\n")
        
        # Set initial pose
        qpos = np.zeros(self.n_dofs)
        for joint_idx, angle in self.standing_pose.items():
            if joint_idx < self.n_dofs:
                qpos[joint_idx] = angle
        self.robot.set_dofs_position(qpos)
        
        # Let robot settle
        print("⏳ Letting robot settle...")
        for _ in range(50):
            self.scene.step()
        
        print("🏃 Starting cat motion...")
        
        # Main animation loop
        while self.scene.viewer.is_alive():
            # Get motion parameters
            motion_idx = self.frame_idx % len(y_motion)
            bounce = y_motion[motion_idx]
            
            # Time-based phase for leg coordination
            t = self.frame_idx * 0.02  # Slow down animation
            
            # Get current positions
            qpos = self.robot.get_dofs_position()
            if hasattr(qpos, 'cpu'):
                qpos = qpos.cpu().numpy()
            
            # Update leg joints with cat-like motion
            for leg_idx in range(4):
                # Joint indices for this leg
                hip_idx = leg_idx * 3
                thigh_idx = hip_idx + 1
                calf_idx = hip_idx + 2
                
                if calf_idx >= self.n_dofs:
                    break
                
                # Trotting gait pattern:
                # FL & RR move together (phase 0)
                # FR & RL move together (phase π)
                if leg_idx in [0, 3]:  # FL, RR
                    phase = 0
                else:  # FR, RL
                    phase = np.pi
                
                # Calculate leg motion
                leg_phase = t * 2 * np.pi + phase
                
                # Hip joint - slight lateral movement
                hip_amp = 0.1
                qpos[hip_idx] = hip_amp * np.sin(leg_phase)
                
                # Thigh joint - main vertical movement
                thigh_base = self.standing_pose[thigh_idx]
                thigh_amp = 0.4 * bounce + 0.1  # Scale with bounce
                qpos[thigh_idx] = thigh_base + thigh_amp * np.sin(leg_phase)
                
                # Calf joint - coordinated with thigh
                calf_base = self.standing_pose[calf_idx]
                calf_amp = 0.6 * bounce + 0.15  # Larger movement
                # Offset phase for natural motion
                qpos[calf_idx] = calf_base - calf_amp * np.sin(leg_phase - np.pi/3)
            
            # Apply positions
            self.robot.set_dofs_position(qpos)
            
            # Step simulation
            self.scene.step()
            self.frame_idx += 1
            
            # Print progress occasionally
            if self.frame_idx % 100 == 0:
                print(f"   Frame {self.frame_idx}, Bounce: {bounce:.2f}")
        
        print("\n✨ Simulation completed!")


def run_final_simulation():
    """Run final Unitree simulation."""
    from cat_meme_locomotion.core.motion_extractor import CatMotionExtractor
    
    print("🐱 Final Unitree Cat Motion")
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
    controller = FinalUnitreeController()
    controller.create_scene()
    controller.load_unitree_robot()
    controller.apply_cat_motion(motion_data)


if __name__ == "__main__":
    run_final_simulation()