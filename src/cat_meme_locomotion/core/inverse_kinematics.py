"""Inverse kinematics solver for quadruped robot using scipy optimization."""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


class QuadrupedIK:
    """Inverse kinematics solver for quadruped robot."""
    
    def __init__(self):
        # Robot dimensions (Unitree Go2 approximate dimensions in meters)
        self.body_length = 0.4  # Front to back
        self.body_width = 0.2   # Left to right
        self.body_height = 0.05  # Thickness
        
        # Leg segment lengths (corrected for actual robot)
        self.l_hip = 0.08      # Hip offset from body
        self.l_thigh = 0.213   # Upper leg length
        self.l_calf = 0.213    # Lower leg length
        
        # Joint limits (radians)
        self.joint_limits = {
            'hip': (-0.8, 0.8),      # Abduction/adduction
            'thigh': (-1.0, 3.14),   # Flexion/extension
            'calf': (-2.8, -0.5)     # Flexion only (knee)
        }
        
        # Leg attachment points relative to body center
        self.leg_origins = {
            'FR': np.array([self.body_length/2, -self.body_width/2, 0]),
            'FL': np.array([self.body_length/2, self.body_width/2, 0]),
            'RR': np.array([-self.body_length/2, -self.body_width/2, 0]),
            'RL': np.array([-self.body_length/2, self.body_width/2, 0])
        }
        
        # Default standing position for feet (relative to body center)
        # More realistic positions within reachable workspace
        self.default_foot_positions = {
            'FR': np.array([self.body_length/2, -self.body_width/2 - 0.08, -0.35]),
            'FL': np.array([self.body_length/2, self.body_width/2 + 0.08, -0.35]),
            'RR': np.array([-self.body_length/2, -self.body_width/2 - 0.08, -0.35]),
            'RL': np.array([-self.body_length/2, self.body_width/2 + 0.08, -0.35])
        }
        
    def forward_kinematics(self, leg_name: str, joint_angles: np.ndarray) -> np.ndarray:
        """
        Calculate foot position given joint angles.
        
        Args:
            leg_name: 'FR', 'FL', 'RR', or 'RL'
            joint_angles: [hip_angle, thigh_angle, calf_angle]
            
        Returns:
            Foot position in body frame
        """
        hip_angle, thigh_angle, calf_angle = joint_angles
        
        # Get leg origin
        origin = self.leg_origins[leg_name]
        
        # Hip joint rotation (around X axis for FR/FL, around -X for RR/RL)
        hip_sign = 1 if leg_name[1] == 'R' else -1  # Right legs rotate opposite
        
        # Position after hip joint
        hip_pos = origin + np.array([
            0,
            hip_sign * self.l_hip * np.cos(hip_angle),
            -self.l_hip * np.sin(hip_angle)
        ])
        
        # Position after thigh joint (rotation around Y axis)
        thigh_pos = hip_pos + np.array([
            self.l_thigh * np.sin(thigh_angle),
            0,
            -self.l_thigh * np.cos(thigh_angle)
        ])
        
        # Position after calf joint (knee)
        # Calf angle is relative to thigh
        total_angle = thigh_angle + calf_angle
        foot_pos = thigh_pos + np.array([
            self.l_calf * np.sin(total_angle),
            0,
            -self.l_calf * np.cos(total_angle)
        ])
        
        return foot_pos
    
    def inverse_kinematics(self, leg_name: str, target_pos: np.ndarray, 
                          initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate joint angles to reach target foot position.
        
        Args:
            leg_name: 'FR', 'FL', 'RR', or 'RL'
            target_pos: Target foot position in body frame
            initial_guess: Initial joint angles (optional)
            
        Returns:
            Joint angles [hip, thigh, calf]
        """
        # First check if target is reachable and adjust if needed
        origin = self.leg_origins[leg_name]
        adjusted_target = self._ensure_reachable(leg_name, target_pos)
        
        # Objective function: minimize distance to target
        def objective(angles):
            current_pos = self.forward_kinematics(leg_name, angles)
            return np.sum((current_pos - adjusted_target) ** 2)
        
        # Constraints
        bounds = [
            self.joint_limits['hip'],
            self.joint_limits['thigh'],
            self.joint_limits['calf']
        ]
        
        # Better initial guess based on target position
        if initial_guess is None:
            initial_guess = self._compute_analytical_guess(leg_name, adjusted_target)
        
        # Try scipy minimize first (faster)
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # If not converged well, try differential evolution (more robust)
        if result.fun > 1e-4:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=15
            )
        
        return result.x
    
    def _ensure_reachable(self, leg_name: str, target_pos: np.ndarray) -> np.ndarray:
        """Ensure target position is within reachable workspace."""
        origin = self.leg_origins[leg_name]
        vec = target_pos - origin
        
        # Check hip reachability first
        hip_sign = 1 if leg_name[1] == 'R' else -1
        y_offset = vec[1] * hip_sign
        z_offset = vec[2]
        hip_reach_required = np.sqrt(y_offset**2 + z_offset**2)
        
        # If hip can't reach, scale down the Y and Z components
        if hip_reach_required > self.l_hip * 0.95:  # 95% of max to avoid singularities
            scale = (self.l_hip * 0.95) / hip_reach_required
            vec[1] *= scale
            vec[2] *= scale
        
        # Check overall reach
        max_reach = self.l_hip + self.l_thigh + self.l_calf
        distance = np.linalg.norm(vec)
        
        if distance > max_reach * 0.95:  # 95% of max reach
            vec = vec * (max_reach * 0.95 / distance)
        
        # Ensure minimum Z height
        adjusted_target = origin + vec
        if adjusted_target[2] > -0.1:  # Too high
            adjusted_target[2] = -0.1
        
        return adjusted_target
    
    def _compute_analytical_guess(self, leg_name: str, target_pos: np.ndarray) -> np.ndarray:
        """Compute analytical initial guess based on target position."""
        origin = self.leg_origins[leg_name]
        vec = target_pos - origin
        
        # Estimate hip angle based on Y offset
        hip_sign = 1 if leg_name[1] == 'R' else -1
        y_offset = vec[1] * hip_sign
        hip_angle = np.arcsin(np.clip(y_offset / self.l_hip, -1, 1))
        
        # Project to XZ plane for thigh and calf angles
        x_offset = vec[0]
        z_offset = vec[2] + self.l_hip * np.cos(hip_angle)  # Adjust for hip
        
        # Use law of cosines to estimate thigh and calf angles
        distance_xz = np.sqrt(x_offset**2 + z_offset**2)
        
        if distance_xz < self.l_thigh + self.l_calf:
            # Reachable - compute angles
            cos_thigh = (distance_xz**2 + self.l_thigh**2 - self.l_calf**2) / (2 * distance_xz * self.l_thigh)
            cos_thigh = np.clip(cos_thigh, -1, 1)
            
            thigh_angle = np.arctan2(x_offset, -z_offset) - np.arccos(cos_thigh)
            
            cos_calf = (self.l_thigh**2 + self.l_calf**2 - distance_xz**2) / (2 * self.l_thigh * self.l_calf)
            cos_calf = np.clip(cos_calf, -1, 1)
            calf_angle = -(np.pi - np.arccos(cos_calf))  # Negative for knee bend
        else:
            # Not directly reachable - use default angles
            thigh_angle = 0.8
            calf_angle = -1.5
        
        return np.array([hip_angle, thigh_angle, calf_angle])
    
    def keypoints_to_foot_positions(self, keypoints: Dict[str, Tuple[float, float]], 
                                   frame_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Convert 2D keypoints to 3D foot positions.
        
        Args:
            keypoints: Dictionary of keypoint names to (x, y) positions
            frame_shape: (height, width) of the image frame
            
        Returns:
            Dictionary of leg names to 3D foot positions
        """
        h, w = frame_shape
        foot_positions = {}
        
        # Map keypoints to legs
        keypoint_mapping = {
            'FR': 'right_wrist',
            'FL': 'left_wrist',
            'RR': 'right_ankle',
            'RL': 'left_ankle'
        }
        
        # Reference points for scaling
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            left_shoulder = np.array(keypoints['left_shoulder'])
            right_shoulder = np.array(keypoints['right_shoulder'])
            shoulder_width_pixels = np.linalg.norm(right_shoulder - left_shoulder)
            scale = self.body_width / shoulder_width_pixels if shoulder_width_pixels > 0 else 0.001
        else:
            # Fallback scaling
            scale = self.body_width / (w * 0.3)  # Assume shoulders are ~30% of image width
        
        # Image center
        center_x, center_y = w / 2, h / 2
        
        for leg_name, keypoint_name in keypoint_mapping.items():
            if keypoint_name in keypoints:
                # Get 2D position
                kp_x, kp_y = keypoints[keypoint_name]
                
                # Convert to 3D position relative to body center
                # X: forward/backward (based on leg position)
                # Y: left/right (from image x)
                # Z: up/down (from image y)
                
                # Default foot position as base
                default_pos = self.default_foot_positions[leg_name].copy()
                
                # Update Y position (lateral movement)
                y_offset = (kp_x - center_x) * scale
                default_pos[1] = self.leg_origins[leg_name][1] + y_offset
                
                # Update Z position (vertical movement)
                # Invert Y axis (image Y increases downward)
                z_offset = -(kp_y - center_y) * scale
                default_pos[2] = -0.3 + z_offset  # Base height + offset
                
                # Add some forward/backward motion based on vertical position
                x_offset = z_offset * 0.3  # Legs move forward when lifting
                default_pos[0] += x_offset
                
                foot_positions[leg_name] = default_pos
            else:
                # Use default position if keypoint not detected
                foot_positions[leg_name] = self.default_foot_positions[leg_name].copy()
        
        return foot_positions
    
    def solve_all_legs(self, foot_positions: Dict[str, np.ndarray], 
                      current_angles: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        Solve IK for all legs.
        
        Args:
            foot_positions: Target foot positions for each leg
            current_angles: Current joint angles (for initial guess)
            
        Returns:
            Joint angles for each leg
        """
        joint_angles = {}
        
        for leg_name, target_pos in foot_positions.items():
            # Use current angles as initial guess if available
            if current_angles and leg_name in current_angles:
                initial_guess = current_angles[leg_name]
            else:
                initial_guess = None
            
            # Solve IK
            angles = self.inverse_kinematics(leg_name, target_pos, initial_guess)
            joint_angles[leg_name] = angles
            
            # Verify solution
            actual_pos = self.forward_kinematics(leg_name, angles)
            error = np.linalg.norm(actual_pos - target_pos)
            if error > 0.01:  # 1cm error threshold
                print(f"⚠️  IK warning for {leg_name}: error = {error:.3f}m")
        
        return joint_angles
    
    def visualize_pose(self, joint_angles: Dict[str, np.ndarray], 
                      target_positions: Optional[Dict[str, np.ndarray]] = None,
                      save_path: Optional[str] = None):
        """Visualize robot pose in 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw body
        body_corners = [
            [-self.body_length/2, -self.body_width/2, 0],
            [self.body_length/2, -self.body_width/2, 0],
            [self.body_length/2, self.body_width/2, 0],
            [-self.body_length/2, self.body_width/2, 0],
            [-self.body_length/2, -self.body_width/2, 0]
        ]
        body_corners = np.array(body_corners)
        ax.plot(body_corners[:, 0], body_corners[:, 1], body_corners[:, 2], 'k-', linewidth=3)
        
        # Draw legs
        colors = {'FR': 'red', 'FL': 'blue', 'RR': 'orange', 'RL': 'green'}
        
        for leg_name, angles in joint_angles.items():
            origin = self.leg_origins[leg_name]
            
            # Calculate joint positions
            hip_angle, thigh_angle, calf_angle = angles
            
            # Hip position
            hip_sign = 1 if leg_name[1] == 'R' else -1
            hip_pos = origin + np.array([
                0,
                hip_sign * self.l_hip * np.cos(hip_angle),
                -self.l_hip * np.sin(hip_angle)
            ])
            
            # Knee position
            knee_pos = hip_pos + np.array([
                self.l_thigh * np.sin(thigh_angle),
                0,
                -self.l_thigh * np.cos(thigh_angle)
            ])
            
            # Foot position
            foot_pos = self.forward_kinematics(leg_name, angles)
            
            # Draw leg segments
            color = colors[leg_name]
            ax.plot([origin[0], hip_pos[0]], [origin[1], hip_pos[1]], 
                   [origin[2], hip_pos[2]], color=color, linewidth=2)
            ax.plot([hip_pos[0], knee_pos[0]], [hip_pos[1], knee_pos[1]], 
                   [hip_pos[2], knee_pos[2]], color=color, linewidth=2)
            ax.plot([knee_pos[0], foot_pos[0]], [knee_pos[1], foot_pos[1]], 
                   [knee_pos[2], foot_pos[2]], color=color, linewidth=2)
            
            # Mark joints
            ax.scatter(*origin, color=color, s=50)
            ax.scatter(*hip_pos, color=color, s=30)
            ax.scatter(*knee_pos, color=color, s=30)
            ax.scatter(*foot_pos, color=color, s=80, marker='o')
            
            # Draw target positions if provided
            if target_positions and leg_name in target_positions:
                target = target_positions[leg_name]
                ax.scatter(*target, color=color, s=100, marker='x', alpha=0.5)
                ax.plot([foot_pos[0], target[0]], [foot_pos[1], target[1]], 
                       [foot_pos[2], target[2]], 'k--', alpha=0.3)
        
        # Set labels and limits
        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y (left)')
        ax.set_zlabel('Z (up)')
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.2])
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
        
        plt.title('Quadruped Robot Pose (IK Solution)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()


def test_ik_solver():
    """Test the IK solver with sample positions."""
    ik = QuadrupedIK()
    
    # Test forward kinematics
    test_angles = np.array([0.0, 0.8, -1.5])
    foot_pos = ik.forward_kinematics('FR', test_angles)
    print(f"Forward kinematics test:")
    print(f"  Angles: {test_angles}")
    print(f"  Foot position: {foot_pos}")
    
    # Test inverse kinematics
    target_pos = foot_pos + np.array([0.05, 0.0, 0.05])  # Move foot slightly
    solved_angles = ik.inverse_kinematics('FR', target_pos)
    solved_pos = ik.forward_kinematics('FR', solved_angles)
    
    print(f"\nInverse kinematics test:")
    print(f"  Target position: {target_pos}")
    print(f"  Solved angles: {solved_angles}")
    print(f"  Actual position: {solved_pos}")
    print(f"  Error: {np.linalg.norm(solved_pos - target_pos):.6f}m")
    
    # Test all legs
    foot_positions = {
        'FR': np.array([0.3, -0.25, -0.25]),
        'FL': np.array([0.3, 0.25, -0.25]),
        'RR': np.array([-0.3, -0.25, -0.3]),
        'RL': np.array([-0.3, 0.25, -0.3])
    }
    
    joint_angles = ik.solve_all_legs(foot_positions)
    print(f"\nAll legs IK solution:")
    for leg, angles in joint_angles.items():
        print(f"  {leg}: {angles}")
    
    # Visualize
    Path("outputs").mkdir(exist_ok=True)
    ik.visualize_pose(joint_angles, foot_positions, save_path="outputs/ik_test.png")
    print(f"\nVisualization saved to outputs/ik_test.png")


if __name__ == "__main__":
    test_ik_solver()