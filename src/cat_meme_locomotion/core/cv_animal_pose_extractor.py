"""OpenCV-based animal pose estimation without external dependencies."""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.signal import savgol_filter
from pathlib import Path


@dataclass
class AnimalKeypoint:
    """Represents a detected keypoint for animals."""
    name: str
    x: float
    y: float
    confidence: float


@dataclass 
class KeypointTrajectory:
    """Trajectory of a keypoint over time."""
    name: str
    positions: List[Tuple[float, float]]
    confidences: List[float]
    
    def smooth_trajectory(self, window_length: int = 5, polyorder: int = 2):
        """Smooth the trajectory using Savitzky-Golay filter."""
        if len(self.positions) < window_length:
            return self.positions
            
        positions = np.array(self.positions)
        x_smooth = savgol_filter(positions[:, 0], window_length, polyorder)
        y_smooth = savgol_filter(positions[:, 1], window_length, polyorder)
        
        return [(x, y) for x, y in zip(x_smooth, y_smooth)]


class CVAnimalPoseExtractor:
    """Extract animal poses using advanced OpenCV techniques."""
    
    # Animal keypoints (17 points similar to COCO)
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Skeleton connections
    SKELETON_CONNECTIONS = [
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
        ('left_ear', 'left_shoulder'), ('right_ear', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip')
    ]
    
    def __init__(self, gif_path: str):
        """Initialize CV-based pose extractor."""
        self.gif_path = gif_path
        self.frames: List[np.ndarray] = []
        self.keypoint_trajectories: Dict[str, KeypointTrajectory] = {}
        self.motion_data: Dict = {}
        
        # Initialize feature detectors
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        print(f"🔧 Initialized CV-based animal pose extractor")
        
    def extract_frames(self) -> List[np.ndarray]:
        """Extract all frames from the GIF file."""
        gif = Image.open(self.gif_path)
        self.frames = []
        
        try:
            while True:
                frame = np.array(gif.convert('RGB'))
                self.frames.append(frame)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
            
        print(f"📽️ Extracted {len(self.frames)} frames from GIF")
        return self.frames
    
    def detect_animal_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the main animal contour in the frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find largest contour (assume it's the animal)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter small contours
        if cv2.contourArea(largest_contour) < 100:
            return None
            
        return largest_contour
    
    def detect_keypoints(self, frame: np.ndarray) -> List[AnimalKeypoint]:
        """Detect keypoints using advanced CV techniques."""
        # Get animal contour
        contour = self.detect_animal_contour(frame)
        if contour is None:
            return []
            
        # Get bounding box and moments
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return []
            
        cx = int(M["m10"] / M["m00"])  # Centroid x
        cy = int(M["m01"] / M["m00"])  # Centroid y
        
        # Create mask from contour
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Detect SIFT keypoints within mask
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp_sift, desc_sift = self.sift.detectAndCompute(gray, mask)
        
        # Sort keypoints by response (strength)
        kp_sift = sorted(kp_sift, key=lambda x: x.response, reverse=True)
        
        # Estimate keypoints based on contour analysis
        detected_keypoints = []
        
        # Find extrema points
        topmost = tuple(contour[contour[:,:,1].argmin()][0])
        bottommost = tuple(contour[contour[:,:,1].argmax()][0])
        leftmost = tuple(contour[contour[:,:,0].argmin()][0])
        rightmost = tuple(contour[contour[:,:,0].argmax()][0])
        
        # Approximate convex hull for body shape
        hull = cv2.convexHull(contour)
        
        # Estimate head region (top part of contour)
        head_mask = contour[:, 0, 1] < cy - h*0.2
        head_region = contour[head_mask]
        if len(head_region) > 0:
            # Nose - topmost point in head region
            nose_idx = head_region[:, 0, 1].argmin()
            nose_pos = head_region[nose_idx][0]
            detected_keypoints.append(AnimalKeypoint('nose', float(nose_pos[0]), float(nose_pos[1]), 0.8))
            
            # Eyes - left and right of nose
            eye_offset = w * 0.1
            detected_keypoints.append(AnimalKeypoint('left_eye', nose_pos[0] - eye_offset, nose_pos[1] + h*0.05, 0.7))
            detected_keypoints.append(AnimalKeypoint('right_eye', nose_pos[0] + eye_offset, nose_pos[1] + h*0.05, 0.7))
            
            # Ears - further out from eyes
            detected_keypoints.append(AnimalKeypoint('left_ear', nose_pos[0] - eye_offset*1.5, nose_pos[1], 0.6))
            detected_keypoints.append(AnimalKeypoint('right_ear', nose_pos[0] + eye_offset*1.5, nose_pos[1], 0.6))
        
        # Body keypoints based on contour segmentation
        # Divide contour into regions
        front_mask = contour[:, 0, 0] < cx
        rear_mask = contour[:, 0, 0] >= cx
        front_region = contour[front_mask]
        rear_region = contour[rear_mask]
        
        # Shoulders (front body)
        if len(front_region) > 0:
            front_top_idx = front_region[:, 0, 1].argmin()
            front_bottom_idx = front_region[:, 0, 1].argmax()
            front_top = front_region[front_top_idx][0]
            front_bottom = front_region[front_bottom_idx][0]
            
            detected_keypoints.append(AnimalKeypoint('left_shoulder', float(front_top[0]), float(front_top[1] + h*0.2), 0.7))
            detected_keypoints.append(AnimalKeypoint('right_shoulder', float(front_bottom[0]), float(front_bottom[1] - h*0.2), 0.7))
            
            # Front legs (elbows and wrists)
            detected_keypoints.append(AnimalKeypoint('left_elbow', float(front_top[0]), float(front_top[1] + h*0.4), 0.6))
            detected_keypoints.append(AnimalKeypoint('right_elbow', float(front_bottom[0]), float(front_bottom[1] - h*0.1), 0.6))
            
            detected_keypoints.append(AnimalKeypoint('left_wrist', float(leftmost[0]), float(leftmost[1]), 0.5))
            detected_keypoints.append(AnimalKeypoint('right_wrist', float(leftmost[0]), float(bottommost[1]), 0.5))
        
        # Hips (rear body)
        if len(rear_region) > 0:
            rear_top_idx = rear_region[:, 0, 1].argmin()
            rear_bottom_idx = rear_region[:, 0, 1].argmax()
            rear_top = rear_region[rear_top_idx][0]
            rear_bottom = rear_region[rear_bottom_idx][0]
            
            detected_keypoints.append(AnimalKeypoint('left_hip', float(rear_top[0]), float(rear_top[1] + h*0.2), 0.7))
            detected_keypoints.append(AnimalKeypoint('right_hip', float(rear_bottom[0]), float(rear_bottom[1] - h*0.2), 0.7))
            
            # Rear legs (knees and ankles)
            detected_keypoints.append(AnimalKeypoint('left_knee', float(rear_top[0]), float(rear_top[1] + h*0.4), 0.6))
            detected_keypoints.append(AnimalKeypoint('right_knee', float(rear_bottom[0]), float(rear_bottom[1] - h*0.1), 0.6))
            
            detected_keypoints.append(AnimalKeypoint('left_ankle', float(rightmost[0]), float(rightmost[1]), 0.5))
            detected_keypoints.append(AnimalKeypoint('right_ankle', float(rightmost[0]), float(bottommost[1]), 0.5))
        
        # Enhance with SIFT keypoints if available
        for i, kp in enumerate(kp_sift[:20]):  # Use top 20 SIFT keypoints
            kp_x, kp_y = kp.pt
            
            # Find closest anatomical keypoint and update confidence
            min_dist = float('inf')
            closest_idx = -1
            
            for j, akp in enumerate(detected_keypoints):
                dist = np.sqrt((akp.x - kp_x)**2 + (akp.y - kp_y)**2)
                if dist < min_dist and dist < w * 0.1:  # Within 10% of width
                    min_dist = dist
                    closest_idx = j
                    
            if closest_idx >= 0:
                # Update position with weighted average
                detected_keypoints[closest_idx].confidence = min(1.0, detected_keypoints[closest_idx].confidence + 0.1)
        
        return detected_keypoints
    
    def analyze_motion(self) -> Dict[str, Any]:
        """Analyze motion from all frames."""
        print("\n🔍 Analyzing motion with CV-based pose detection...")
        
        # Extract frames if not already done
        if not self.frames:
            self.extract_frames()
            
        # Initialize keypoint trajectories
        for name in self.KEYPOINT_NAMES:
            self.keypoint_trajectories[name] = KeypointTrajectory(
                name=name,
                positions=[],
                confidences=[]
            )
        
        # Process each frame
        for i, frame in enumerate(self.frames):
            if i % 10 == 0:
                print(f"   Processing frame {i+1}/{len(self.frames)}...")
                
            keypoints = self.detect_keypoints(frame)
            
            # Update trajectories
            # First, create a dict of detected keypoints
            kp_dict = {kp.name: kp for kp in keypoints}
            
            # Then update all trajectories (with interpolation for missing keypoints)
            for name in self.KEYPOINT_NAMES:
                if name in kp_dict:
                    kp = kp_dict[name]
                    self.keypoint_trajectories[name].positions.append((kp.x, kp.y))
                    self.keypoint_trajectories[name].confidences.append(kp.confidence)
                elif len(self.keypoint_trajectories[name].positions) > 0:
                    # Interpolate from last known position
                    last_pos = self.keypoint_trajectories[name].positions[-1]
                    self.keypoint_trajectories[name].positions.append(last_pos)
                    self.keypoint_trajectories[name].confidences.append(0.3)  # Low confidence
        
        # Smooth trajectories
        for name, traj in self.keypoint_trajectories.items():
            if len(traj.positions) > 5:
                traj.positions = traj.smooth_trajectory()
        
        # Analyze gait patterns
        gait_info = self._analyze_gait_pattern()
        
        # Create motion data summary
        valid_keypoints = [name for name, traj in self.keypoint_trajectories.items() 
                          if len(traj.positions) > 0 and np.mean(traj.confidences) > 0.3]
        
        self.motion_data = {
            'num_frames': len(self.frames),
            'detected_keypoints': valid_keypoints,
            'gait_pattern': gait_info['pattern'],
            'gait_confidence': gait_info['confidence'],
            'avg_confidence': np.mean([
                np.mean(traj.confidences) 
                for traj in self.keypoint_trajectories.values() 
                if traj.confidences
            ])
        }
        
        print(f"\n✅ Motion analysis complete!")
        print(f"   • Detected keypoints: {len(self.motion_data['detected_keypoints'])}")
        print(f"   • Gait pattern: {self.motion_data['gait_pattern']}")
        print(f"   • Average confidence: {self.motion_data['avg_confidence']:.2f}")
        
        return self.motion_data
    
    def _analyze_gait_pattern(self) -> Dict[str, Any]:
        """Analyze gait pattern from paw trajectories."""
        # Get paw trajectories
        paw_names = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']
        paw_trajectories = {
            name: self.keypoint_trajectories[name] 
            for name in paw_names 
            if name in self.keypoint_trajectories and len(self.keypoint_trajectories[name].positions) > 0
        }
        
        if len(paw_trajectories) < 2:
            return {'pattern': 'unknown', 'confidence': 0.0}
        
        # Analyze vertical motion of paws
        paw_heights = {}
        for name, traj in paw_trajectories.items():
            if traj.positions:
                positions = np.array(traj.positions)
                paw_heights[name] = positions[:, 1]  # Y coordinates
        
        # Simple gait classification based on phase relationships
        if self._is_trot_gait(paw_heights):
            return {'pattern': 'trot', 'confidence': 0.8}
        elif self._is_pace_gait(paw_heights):
            return {'pattern': 'pace', 'confidence': 0.7}
        elif self._is_gallop_gait(paw_heights):
            return {'pattern': 'gallop', 'confidence': 0.7}
        else:
            return {'pattern': 'walk', 'confidence': 0.6}
    
    def _is_trot_gait(self, paw_heights: Dict[str, np.ndarray]) -> bool:
        """Check if gait is trot (diagonal pairs move together)."""
        if len(paw_heights) < 4:
            return False
            
        # Check correlation between diagonal pairs
        try:
            # Left front with right rear
            if 'left_wrist' in paw_heights and 'right_ankle' in paw_heights:
                corr1 = np.corrcoef(paw_heights['left_wrist'], paw_heights['right_ankle'])[0, 1]
                if corr1 > 0.7:
                    return True
                    
            # Right front with left rear
            if 'right_wrist' in paw_heights and 'left_ankle' in paw_heights:
                corr2 = np.corrcoef(paw_heights['right_wrist'], paw_heights['left_ankle'])[0, 1]
                if corr2 > 0.7:
                    return True
                    
        except:
            pass
            
        return False
    
    def _is_pace_gait(self, paw_heights: Dict[str, np.ndarray]) -> bool:
        """Check if gait is pace (lateral pairs move together)."""
        if len(paw_heights) < 2:
            return False
            
        try:
            # Check same-side correlation
            correlations = []
            
            # Left side
            if 'left_wrist' in paw_heights and 'left_ankle' in paw_heights:
                corr = np.corrcoef(paw_heights['left_wrist'], paw_heights['left_ankle'])[0, 1]
                correlations.append(corr)
                
            # Right side
            if 'right_wrist' in paw_heights and 'right_ankle' in paw_heights:
                corr = np.corrcoef(paw_heights['right_wrist'], paw_heights['right_ankle'])[0, 1]
                correlations.append(corr)
                
            return len(correlations) > 0 and np.mean(correlations) > 0.7
        except:
            return False
    
    def _is_gallop_gait(self, paw_heights: Dict[str, np.ndarray]) -> bool:
        """Check if gait is gallop (all paws have similar phase)."""
        if len(paw_heights) < 3:
            return False
            
        try:
            # Check if all paws move in similar pattern
            all_heights = list(paw_heights.values())
            correlations = []
            
            for i in range(len(all_heights)-1):
                for j in range(i+1, len(all_heights)):
                    if len(all_heights[i]) == len(all_heights[j]):
                        corr = np.corrcoef(all_heights[i], all_heights[j])[0, 1]
                        correlations.append(corr)
                        
            return len(correlations) > 0 and np.mean(correlations) > 0.6
        except:
            return False
    
    def visualize_keypoints(self, output_path: str, frame_idx: Optional[int] = None):
        """Visualize detected keypoints on frames."""
        if not self.frames or not self.keypoint_trajectories:
            print("No data to visualize")
            return
            
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Select frame to visualize
        if frame_idx is None:
            frame_idx = len(self.frames) // 2  # Middle frame
            
        frame = self.frames[frame_idx]
        
        # Plot 1: Keypoints on frame
        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(frame)
        ax1.set_title(f'CV-based Animal Pose Detection (Frame {frame_idx})')
        
        # Draw keypoints
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.KEYPOINT_NAMES)))
        
        for i, name in enumerate(self.KEYPOINT_NAMES):
            if name in self.keypoint_trajectories and frame_idx < len(self.keypoint_trajectories[name].positions):
                if self.keypoint_trajectories[name].positions:
                    x, y = self.keypoint_trajectories[name].positions[frame_idx]
                    conf = self.keypoint_trajectories[name].confidences[frame_idx]
                    
                    if conf > 0.3:  # Only show confident keypoints
                        ax1.scatter(x, y, c=[colors[i]], s=100, alpha=conf, label=name)
                        ax1.text(x+5, y+5, name[:3], fontsize=8, color='white',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
        
        # Draw skeleton
        for conn in self.SKELETON_CONNECTIONS:
            kp1, kp2 = conn
            if (kp1 in self.keypoint_trajectories and kp2 in self.keypoint_trajectories and
                frame_idx < len(self.keypoint_trajectories[kp1].positions) and
                frame_idx < len(self.keypoint_trajectories[kp2].positions) and
                self.keypoint_trajectories[kp1].positions and
                self.keypoint_trajectories[kp2].positions):
                
                x1, y1 = self.keypoint_trajectories[kp1].positions[frame_idx]
                x2, y2 = self.keypoint_trajectories[kp2].positions[frame_idx]
                conf1 = self.keypoint_trajectories[kp1].confidences[frame_idx]
                conf2 = self.keypoint_trajectories[kp2].confidences[frame_idx]
                
                if conf1 > 0.3 and conf2 > 0.3:
                    ax1.plot([x1, x2], [y1, y2], 'g-', alpha=min(conf1, conf2), linewidth=2)
        
        ax1.axis('off')
        
        # Plot 2: Keypoint trajectories
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title('Keypoint Trajectories')
        
        for i, (name, traj) in enumerate(self.keypoint_trajectories.items()):
            if traj.positions and np.mean(traj.confidences) > 0.3:
                positions = np.array(traj.positions)
                ax2.plot(positions[:, 0], positions[:, 1], '-', 
                        color=colors[i % len(colors)], alpha=0.7, label=name)
                
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.invert_yaxis()
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Confidence over time
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_title('Detection Confidence Over Time')
        
        for name, traj in self.keypoint_trajectories.items():
            if traj.confidences and np.mean(traj.confidences) > 0.1:
                ax3.plot(traj.confidences, label=name, alpha=0.7)
                
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Confidence')
        ax3.set_ylim([0, 1])
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: Motion analysis summary
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        summary_text = f"""Motion Analysis Summary:
        
Frames: {self.motion_data.get('num_frames', 0)}
Detected Keypoints: {len(self.motion_data.get('detected_keypoints', []))}
Gait Pattern: {self.motion_data.get('gait_pattern', 'unknown')}
Gait Confidence: {self.motion_data.get('gait_confidence', 0):.2f}
Avg Detection Confidence: {self.motion_data.get('avg_confidence', 0):.2f}

Method: OpenCV-based (SIFT + Contour Analysis)
"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to {output_path}")