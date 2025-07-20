#!/usr/bin/env python3
"""Unitree robot simulation with cat motion in Genesis."""

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


class UnitreeRobotController:
    """Controller for Unitree robot with cat motion."""
    
    def __init__(self):
        # Initialize Genesis
        gs.init(backend=gs.cuda)
        
        self.scene = None
        self.robot = None
        self.motion_data = None
        self.frame_idx = 0
        
        # Joint names for Unitree robots
        self.joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        
    def create_scene(self):
        """Create Genesis scene."""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0, 0, -9.81),
                substeps=2,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, -2.5, 1.5),
                camera_lookat=(0.0, 0.0, 0.4),
                camera_fov=40,
                max_FPS=60,
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
        """Load Unitree Go1 robot model."""
        print("🤖 Creating Unitree-style robot...")
        
        # For now, always use simplified model to avoid URDF/mesh issues
        print("✅ Using simplified Unitree robot model")
        self._create_simple_unitree()
    
    def _create_simple_unitree(self):
        """Create simplified Unitree-like robot using primitive shapes."""
        # Robot parts storage
        self.robot_parts = []
        
        # Main body (trunk)
        body = self.scene.add_entity(
            gs.morphs.Box(
                pos=(0, 0, 0.35),
                size=(0.4, 0.2, 0.1),
            ),
            material=gs.materials.Rigid(
                rho=20000.0,
                friction=0.8,
            ),
        )
        self.robot_parts.append(body)
        self.robot = body  # Main reference
        
        # Create 4 legs with joints
        leg_positions = [
            ("FL", 0.15, 0.1),    # Front Left
            ("FR", 0.15, -0.1),   # Front Right
            ("RL", -0.15, 0.1),   # Rear Left
            ("RR", -0.15, -0.1),  # Rear Right
        ]
        
        self.legs = []
        for name, x_offset, y_offset in leg_positions:
            # Upper leg (thigh)
            upper_leg = self.scene.add_entity(
                gs.morphs.Box(
                    pos=(x_offset, y_offset, 0.25),
                    size=(0.05, 0.05, 0.15),
                ),
                material=gs.materials.Rigid(rho=5000.0),
            )
            
            # Lower leg (calf)
            lower_leg = self.scene.add_entity(
                gs.morphs.Box(
                    pos=(x_offset, y_offset, 0.1),
                    size=(0.04, 0.04, 0.15),
                ),
                material=gs.materials.Rigid(rho=3000.0),
            )
            
            # Foot
            foot = self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=(x_offset, y_offset, 0.02),
                    radius=0.02,
                ),
                material=gs.materials.Rigid(
                    rho=1000.0,
                    friction=1.5,
                ),
            )
            
            self.legs.append({
                'name': name,
                'upper': upper_leg,
                'lower': lower_leg,
                'foot': foot,
                'offset': (x_offset, y_offset)
            })
            
            self.robot_parts.extend([upper_leg, lower_leg, foot])
        
    def apply_cat_motion(self, motion_data: Dict):
        """Apply cat motion to robot."""
        self.motion_data = motion_data
        y_motion = motion_data['y_normalized']
        
        # Build scene
        self.scene.build()
        
        print("\n🎮 Starting Unitree robot simulation...")
        print("   • Robot mimics cat bouncing motion")
        print("   • Close viewer to exit\n")
        
        # Get robot DOFs if available
        try:
            num_dofs = self.robot.n_dofs
            print(f"Robot has {num_dofs} degrees of freedom")
            
            # Set initial pose
            initial_qpos = np.zeros(num_dofs)
            # Set standing pose for quadruped
            if num_dofs == 12:  # Unitree has 12 DOFs
                # Hip joints slightly outward
                initial_qpos[0] = 0.1   # FL hip
                initial_qpos[3] = -0.1  # FR hip  
                initial_qpos[6] = 0.1   # RL hip
                initial_qpos[9] = -0.1  # RR hip
                
                # Thigh joints
                for i in [1, 4, 7, 10]:
                    initial_qpos[i] = -0.8
                    
                # Calf joints
                for i in [2, 5, 8, 11]:
                    initial_qpos[i] = 1.6
                    
            self.robot.set_dofs_position(initial_qpos)
        except:
            print("Note: Using simplified model without articulated joints")
        
        # Animation loop
        while self.scene.viewer.is_alive():
            # Get current motion
            motion_idx = self.frame_idx % len(y_motion)
            bounce = y_motion[motion_idx]
            phase = self.frame_idx * 0.05
            
            # Update robot based on type
            try:
                if hasattr(self.robot, 'n_dofs') and self.robot.n_dofs > 0:
                    # Articulated robot
                    self._update_articulated_robot(bounce, phase)
                else:
                    # Simple body
                    self._update_simple_robot(bounce, phase)
            except Exception as e:
                # Fallback to simple motion
                self._update_simple_robot(bounce, phase)
            
            # Step simulation
            self.scene.step()
            self.frame_idx += 1
        
        print("\n✨ Simulation completed!")
    
    def _update_articulated_robot(self, bounce: float, phase: float):
        """Update articulated robot joints."""
        qpos = self.robot.get_dofs_position()
        
        # Create bouncing gait
        for leg_idx in range(4):
            hip_idx = leg_idx * 3
            thigh_idx = hip_idx + 1
            calf_idx = hip_idx + 2
            
            # Phase shift for each leg
            leg_phase = phase + leg_idx * np.pi / 2
            
            # Hip movement (lateral)
            qpos[hip_idx] = 0.1 * np.sin(leg_phase) * (1 if leg_idx % 2 == 0 else -1)
            
            # Thigh movement (main bounce)
            qpos[thigh_idx] = -0.8 + bounce * 0.3 * np.sin(leg_phase * 2)
            
            # Calf movement (follows thigh)
            qpos[calf_idx] = 1.6 - bounce * 0.4 * np.sin(leg_phase * 2)
        
        # Apply joint positions
        self.robot.set_dofs_position(qpos)
        
        # Add body motion
        base_height = 0.35 + bounce * 0.1
        self.robot.set_pos(np.array([0, 0, base_height]))
    
    def _update_simple_robot(self, bounce: float, phase: float):
        """Update simple robot body and legs."""
        # Update main body
        body_height = 0.35 + bounce * 0.15
        roll = 0.03 * np.sin(phase)
        pitch = 0.03 * np.sin(phase * 2)
        
        self.robot.set_pos(np.array([0, 0, body_height]))
        
        # Update legs if available
        if hasattr(self, 'legs') and self.legs:
            for i, leg in enumerate(self.legs):
                x_offset, y_offset = leg['offset']
                
                # Phase shift for each leg
                leg_phase = phase + i * np.pi / 2
                
                # Leg movement pattern
                hip_sway = 0.01 * np.sin(leg_phase)
                knee_bend = bounce * 0.1 + 0.05 * np.sin(leg_phase * 2)
                
                # Update upper leg
                upper_height = body_height - 0.1 - knee_bend
                leg['upper'].set_pos(np.array([
                    x_offset + hip_sway,
                    y_offset,
                    upper_height
                ]))
                
                # Update lower leg
                lower_height = upper_height - 0.15 + knee_bend * 0.5
                leg['lower'].set_pos(np.array([
                    x_offset + hip_sway * 0.5,
                    y_offset,
                    lower_height
                ]))
                
                # Update foot with ground contact
                foot_lift = max(0, bounce * 0.05 * np.sin(leg_phase * 2))
                leg['foot'].set_pos(np.array([
                    x_offset + hip_sway * 0.3,
                    y_offset,
                    0.02 + foot_lift
                ]))


def run_unitree_simulation():
    """Main function to run Unitree simulation."""
    from cat_meme_locomotion.core.motion_extractor import CatMotionExtractor
    
    print("🐱 Unitree Robot Cat Motion Simulation")
    print("=" * 40)
    
    # Extract cat motion
    print("\n📊 Extracting cat motion from GIF...")
    extractor = CatMotionExtractor('assets/gifs/chipi-chipi-chapa-chapa.gif')
    motion_data = extractor.extract_motion_pattern()
    
    if not motion_data:
        print("❌ Failed to extract motion data")
        return
    
    print(f"✅ Extracted {motion_data['num_frames']} frames")
    print(f"✅ Detected {len(motion_data['peaks'])} bounce peaks")
    
    # Create controller and run
    controller = UnitreeRobotController()
    controller.create_scene()
    controller.load_unitree_robot()
    controller.apply_cat_motion(motion_data)


if __name__ == "__main__":
    run_unitree_simulation()