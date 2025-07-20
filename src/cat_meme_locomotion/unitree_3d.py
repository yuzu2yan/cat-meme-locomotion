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
        
        # Joint names for Unitree Go2 robots (from URDF)
        self.joint_names = []
        self.joint_indices = {}
        
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
        """Load Unitree Go2 robot model."""
        print("🤖 Loading Unitree Go2 robot...")
        
        # First, copy the URDF to fix the mesh paths
        self._fix_urdf_paths()
        
        # Load the fixed URDF
        urdf_path = Path("go2_fixed.urdf")
        if urdf_path.exists():
            try:
                print(f"📁 Loading URDF from: {urdf_path}")
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=str(urdf_path),
                        pos=(0, 0, 0.35),
                        euler=(0, 0, 0),
                    ),
                )
                print("✅ Successfully loaded Unitree Go2 robot from URDF!")
                # Do NOT use simplified model - we have the real one
                return
            except Exception as e:
                print(f"❌ Error loading URDF: {e}")
                raise  # Don't fall back to simplified model
        else:
            print("❌ Fixed URDF file not found")
            raise RuntimeError("Cannot load Unitree Go2 URDF")
    
    def _fix_urdf_paths(self):
        """Fix mesh paths in URDF to use absolute paths."""
        import re
        
        # Read original URDF
        with open("go2.urdf", "r") as f:
            urdf_content = f.read()
        
        # Get current directory
        current_dir = Path.cwd()
        
        # Replace relative mesh paths with absolute paths
        urdf_content = re.sub(
            r'filename="../dae/([^"]+)"',
            f'filename="{current_dir}/dae/\\1"',
            urdf_content
        )
        
        # Write fixed URDF
        with open("go2_fixed.urdf", "w") as f:
            f.write(urdf_content)
        
        print("✅ Fixed URDF mesh paths")
    
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
        
        # Get robot DOFs and joint information
        try:
            num_dofs = self.robot.n_dofs
            print(f"Robot has {num_dofs} degrees of freedom")
            
            # Get joint names from URDF
            joint_names = []
            joint_limits = []
            
            # For Go2, we expect specific joint naming pattern
            # Try to detect joint names based on common patterns
            expected_joints = [
                # Front Left
                'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                # Front Right  
                'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                # Rear Left
                'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                # Rear Right
                'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
            ]
            
            # Map joint indices
            self.joint_indices = {}
            for i, name in enumerate(expected_joints):
                if i < num_dofs:
                    self.joint_indices[name] = i
            
            print(f"Mapped {len(self.joint_indices)} joints")
            
            # Set initial standing pose
            initial_qpos = np.zeros(num_dofs)
            
            # Standing position for Go2
            # Start with all joints at zero, then adjust
            for i in range(min(num_dofs, 18)):
                initial_qpos[i] = 0.0
            
            # Set specific joint angles for standing
            # These values are typical for Unitree robots
            if num_dofs >= 12:
                # Hip joints (keep at 0)
                # Thigh joints - negative for proper stance
                for i in [1, 4, 7, 10]:
                    if i < num_dofs:
                        initial_qpos[i] = -0.7
                
                # Calf joints - positive for proper stance  
                for i in [2, 5, 8, 11]:
                    if i < num_dofs:
                        initial_qpos[i] = 1.4
                        
            self.robot.set_dofs_position(initial_qpos)
            print("✅ Set initial standing pose")
            
            # Store for later use
            self.initial_pose = initial_qpos.copy()
            
        except Exception as e:
            print(f"Warning: Could not set initial pose: {e}")
        
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
        # Get current joint positions
        qpos = self.robot.get_dofs_position()
        num_dofs = len(qpos)
        
        # Only update the first 12 DOFs (leg joints)
        # Skip head/tail joints if present
        for leg_idx in range(4):
            # Each leg has 3 joints
            hip_idx = leg_idx * 3
            thigh_idx = hip_idx + 1
            calf_idx = hip_idx + 2
            
            # Check bounds
            if calf_idx >= num_dofs:
                break
            
            # Phase shift for each leg to create trotting pattern
            leg_phase = phase + leg_idx * np.pi / 2
            
            # Hip joint movement (abduction/adduction) - minimal
            qpos[hip_idx] = 0.05 * np.sin(leg_phase)
            
            # Thigh joint - main movement for cat-like bounce
            # Base angle + dynamic movement
            base_thigh = -0.7  # Base standing angle
            bounce_motion = bounce * 0.2 * np.sin(leg_phase * 2)
            qpos[thigh_idx] = base_thigh + bounce_motion
            
            # Calf joint - follows thigh with opposite motion
            base_calf = 1.4  # Base standing angle
            calf_motion = -bounce * 0.3 * np.sin(leg_phase * 2)
            qpos[calf_idx] = base_calf + calf_motion
        
        # Apply joint positions
        self.robot.set_dofs_position(qpos)
        
        # Don't modify robot base position - let physics handle it
        # The bounce should come from leg movement, not teleporting the body
    
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