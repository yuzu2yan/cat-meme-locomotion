"""Improved YOLO-based pose estimation with better animal motion mapping."""

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.signal import savgol_filter
from pathlib import Path
import os
import time
import imageio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnimalKeypoint:
    """Represents a detected keypoint."""
    name: str
    x: float
    y: float
    confidence: float


@dataclass  
class KeypointTrajectory:
    """Trajectory of a keypoint across frames."""
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


class ImprovedYOLOPoseExtractor:
    """Extract poses using YOLO with improved animal motion mapping."""
    
    # COCO keypoints (17 points) - we'll map these to animal joints
    COCO_KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Animal keypoint mapping from human pose
    ANIMAL_KEYPOINT_MAPPING = {
        # Head/neck region
        'head': ['nose'],
        'neck': ['nose', 'left_shoulder', 'right_shoulder'],  # Average of these
        
        # Front legs (map from arms)
        'front_left_shoulder': ['left_shoulder'],
        'front_left_elbow': ['left_elbow'],
        'front_left_paw': ['left_wrist'],
        'front_right_shoulder': ['right_shoulder'],
        'front_right_elbow': ['right_elbow'], 
        'front_right_paw': ['right_wrist'],
        
        # Back legs (map from legs)
        'back_left_hip': ['left_hip'],
        'back_left_knee': ['left_knee'],
        'back_left_paw': ['left_ankle'],
        'back_right_hip': ['right_hip'],
        'back_right_knee': ['right_knee'],
        'back_right_paw': ['right_ankle'],
        
        # Body center
        'spine_center': ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder'],
        'spine_front': ['left_shoulder', 'right_shoulder'],
        'spine_rear': ['left_hip', 'right_hip']
    }
    
    # Skeleton connections for visualization
    ANIMAL_SKELETON_CONNECTIONS = [
        # Head to body
        ('head', 'neck'),
        ('neck', 'spine_front'),
        
        # Spine
        ('spine_front', 'spine_center'),
        ('spine_center', 'spine_rear'),
        
        # Front legs
        ('spine_front', 'front_left_shoulder'),
        ('front_left_shoulder', 'front_left_elbow'),
        ('front_left_elbow', 'front_left_paw'),
        ('spine_front', 'front_right_shoulder'),
        ('front_right_shoulder', 'front_right_elbow'),
        ('front_right_elbow', 'front_right_paw'),
        
        # Back legs
        ('spine_rear', 'back_left_hip'),
        ('back_left_hip', 'back_left_knee'),
        ('back_left_knee', 'back_left_paw'),
        ('spine_rear', 'back_right_hip'),
        ('back_right_hip', 'back_right_knee'),
        ('back_right_knee', 'back_right_paw'),
    ]
    
    def __init__(self, input_path: str, model_name: str = 'yolov8x-pose.pt', 
                 max_frames: Optional[int] = None, target_fps: Optional[int] = None,
                 use_object_detection: bool = True):
        """Initialize improved YOLO pose extractor.
        
        Args:
            input_path: Path to input GIF or video file
            model_name: YOLO model name
            max_frames: Maximum number of frames to process
            target_fps: Target FPS for frame extraction
            use_object_detection: Whether to use object detection to find animals first
        """
        self.input_path = input_path
        self.frames: List[np.ndarray] = []
        self.keypoint_trajectories: Dict[str, KeypointTrajectory] = {}
        self.animal_keypoint_trajectories: Dict[str, KeypointTrajectory] = {}
        self.motion_data: Dict = {}
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.use_object_detection = use_object_detection
        
        # Initialize YOLO models
        logger.info(f"🔧 Loading YOLO pose model: {model_name}")
        self.pose_model = YOLO(model_name)
        
        if self.use_object_detection:
            logger.info("🔧 Loading YOLO object detection model for animal detection")
            self.detect_model = YOLO('yolov8x.pt')
        
        # Check if GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"📍 Using device: {self.device}")
        
    def extract_frames(self) -> List[np.ndarray]:
        """Extract all frames from the input file."""
        file_ext = Path(self.input_path).suffix.lower()
        
        if file_ext == '.gif':
            # Extract from GIF
            gif = Image.open(self.input_path)
            self.frames = []
            
            try:
                frame_count = 0
                while True:
                    frame = np.array(gif.convert('RGB'))
                    self.frames.append(frame)
                    frame_count += 1
                    
                    if self.max_frames and frame_count >= self.max_frames:
                        break
                        
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
                
            logger.info(f"📽️ Extracted {len(self.frames)} frames from GIF")
            
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            # Extract from video
            self.frames = self._extract_frames_from_video(
                self.input_path, 
                max_frames=self.max_frames or 300,
                target_fps=self.target_fps or 15
            )
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        return self.frames
    
    def _extract_frames_from_video(self, video_path: str, 
                                  max_frames: Optional[int] = None,
                                  target_fps: Optional[int] = None) -> List[np.ndarray]:
        """Extract frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video info: {total_frames} frames @ {fps:.1f} FPS")
        
        frame_skip = 1
        if target_fps and target_fps < fps:
            frame_skip = int(fps / target_fps)
            
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
            
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def detect_animal_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect animal bounding box in frame."""
        if not self.use_object_detection:
            return None
            
        # Run object detection
        results = self.detect_model(frame, verbose=False, conf=0.25)
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            classes = boxes.cls.cpu().numpy()
            
            # Look for animal classes (COCO classes)
            # 15: bird, 16: cat, 17: dog, 18: horse, 19: sheep, 20: cow, 
            # 21: elephant, 22: bear, 23: zebra, 24: giraffe
            animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            
            for i, cls in enumerate(classes):
                if int(cls) in animal_classes:
                    # Get bounding box
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    return (int(x1), int(y1), int(x2), int(y2))
        
        return None
        
    def detect_keypoints(self, frame: np.ndarray) -> List[AnimalKeypoint]:
        """Detect keypoints in a single frame using YOLO."""
        # First try to detect animal bbox
        bbox = self.detect_animal_bbox(frame)
        
        if bbox:
            # Crop to animal region with padding
            x1, y1, x2, y2 = bbox
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Run pose detection on cropped region
            results = self.pose_model(cropped_frame, verbose=False, conf=0.2)
            
            # Adjust coordinates back to full frame
            offset_x, offset_y = x1, y1
        else:
            # Run pose detection on full frame
            results = self.pose_model(frame, verbose=False, conf=0.2)
            offset_x, offset_y = 0, 0
        
        detected_keypoints = []
        
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints
            
            if keypoints.xy.shape[0] > 0:
                kp_xy = keypoints.xy[0].cpu().numpy()
                kp_conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(17)
                
                # Convert to AnimalKeypoint objects with offset
                for i, name in enumerate(self.COCO_KEYPOINT_NAMES):
                    if i < len(kp_xy):
                        detected_keypoints.append(AnimalKeypoint(
                            name=name,
                            x=float(kp_xy[i, 0] + offset_x),
                            y=float(kp_xy[i, 1] + offset_y),
                            confidence=float(kp_conf[i])
                        ))
        
        return detected_keypoints
    
    def map_to_animal_keypoints(self, human_keypoints: List[AnimalKeypoint]) -> Dict[str, AnimalKeypoint]:
        """Map human pose keypoints to animal keypoints."""
        # Create lookup dictionary
        human_kp_dict = {kp.name: kp for kp in human_keypoints}
        animal_keypoints = {}
        
        for animal_name, human_names in self.ANIMAL_KEYPOINT_MAPPING.items():
            # Get all relevant human keypoints
            relevant_kps = [human_kp_dict.get(name) for name in human_names if name in human_kp_dict]
            relevant_kps = [kp for kp in relevant_kps if kp and kp.confidence > 0.1]
            
            if relevant_kps:
                # Average position and confidence
                avg_x = np.mean([kp.x for kp in relevant_kps])
                avg_y = np.mean([kp.y for kp in relevant_kps])
                avg_conf = np.mean([kp.confidence for kp in relevant_kps])
                
                animal_keypoints[animal_name] = AnimalKeypoint(
                    name=animal_name,
                    x=avg_x,
                    y=avg_y,
                    confidence=avg_conf
                )
        
        return animal_keypoints
    
    def analyze_motion(self) -> Dict:
        """Analyze motion from all frames."""
        logger.info("\n🔍 Analyzing motion with improved YOLO pose detection...")
        
        if not self.frames:
            self.extract_frames()
        
        # Initialize trajectories for both human and animal keypoints
        for name in self.COCO_KEYPOINT_NAMES:
            self.keypoint_trajectories[name] = KeypointTrajectory(
                name=name, positions=[], confidences=[]
            )
        
        for name in self.ANIMAL_KEYPOINT_MAPPING.keys():
            self.animal_keypoint_trajectories[name] = KeypointTrajectory(
                name=name, positions=[], confidences=[]
            )
        
        # Process each frame
        animal_detected_count = 0
        
        for i, frame in enumerate(self.frames):
            if i % 10 == 0:
                logger.info(f"   Processing frame {i+1}/{len(self.frames)}...")
            
            # Detect keypoints
            human_keypoints = self.detect_keypoints(frame)
            
            # Update human keypoint trajectories
            detected_names = set()
            for kp in human_keypoints:
                if kp.name in self.keypoint_trajectories:
                    self.keypoint_trajectories[kp.name].positions.append((kp.x, kp.y))
                    self.keypoint_trajectories[kp.name].confidences.append(kp.confidence)
                    detected_names.add(kp.name)
            
            # Add None for missing keypoints
            for name in self.COCO_KEYPOINT_NAMES:
                if name not in detected_names:
                    self.keypoint_trajectories[name].positions.append((0, 0))
                    self.keypoint_trajectories[name].confidences.append(0.0)
            
            # Map to animal keypoints
            animal_keypoints = self.map_to_animal_keypoints(human_keypoints)
            if animal_keypoints:
                animal_detected_count += 1
            
            # Update animal keypoint trajectories
            detected_animal_names = set()
            for name, kp in animal_keypoints.items():
                self.animal_keypoint_trajectories[name].positions.append((kp.x, kp.y))
                self.animal_keypoint_trajectories[name].confidences.append(kp.confidence)
                detected_animal_names.add(name)
            
            # Add None for missing animal keypoints
            for name in self.ANIMAL_KEYPOINT_MAPPING.keys():
                if name not in detected_animal_names:
                    self.animal_keypoint_trajectories[name].positions.append((0, 0))
                    self.animal_keypoint_trajectories[name].confidences.append(0.0)
        
        # Analyze gait pattern
        gait_pattern = self._analyze_gait_pattern()
        
        # Calculate statistics
        detected_keypoints = [name for name, traj in self.animal_keypoint_trajectories.items() 
                            if any(c > 0.3 for c in traj.confidences)]
        
        self.motion_data = {
            'num_frames': len(self.frames),
            'detected_keypoints': detected_keypoints,
            'animal_detection_rate': animal_detected_count / len(self.frames),
            'gait_pattern': gait_pattern,
            'keypoint_trajectories': self.keypoint_trajectories,
            'animal_keypoint_trajectories': self.animal_keypoint_trajectories
        }
        
        logger.info(f"\n✅ Motion analysis complete!")
        logger.info(f"   • Animal detection rate: {self.motion_data['animal_detection_rate']:.1%}")
        logger.info(f"   • Detected animal keypoints: {len(detected_keypoints)}")
        logger.info(f"   • Gait pattern: {gait_pattern}")
        
        return self.motion_data
    
    def _analyze_gait_pattern(self) -> str:
        """Analyze gait pattern from animal paw movements."""
        # Use animal paw trajectories
        paw_names = ['front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw']
        paw_data = {}
        
        for name in paw_names:
            if name in self.animal_keypoint_trajectories:
                traj = self.animal_keypoint_trajectories[name]
                if traj.positions:
                    # Calculate vertical movement
                    y_positions = [p[1] for p, c in zip(traj.positions, traj.confidences) 
                                 if p[1] > 0 and c > 0.2]
                    if y_positions:
                        paw_data[name] = np.array(y_positions)
        
        if len(paw_data) < 2:
            return "unknown"
        
        # Analyze phase relationships between paws
        if len(paw_data) >= 4:
            # Check diagonal pairs (trot pattern)
            if 'front_left_paw' in paw_data and 'back_right_paw' in paw_data:
                fl = paw_data['front_left_paw']
                br = paw_data['back_right_paw']
                
                if len(fl) > 10 and len(br) > 10:
                    min_len = min(len(fl), len(br))
                    correlation = np.corrcoef(fl[:min_len], br[:min_len])[0, 1]
                    
                    if correlation > 0.7:
                        return "trot"  # Diagonal pairs move together
            
            # Check lateral pairs (pace pattern)
            if 'front_left_paw' in paw_data and 'back_left_paw' in paw_data:
                fl = paw_data['front_left_paw']
                bl = paw_data['back_left_paw']
                
                if len(fl) > 10 and len(bl) > 10:
                    min_len = min(len(fl), len(bl))
                    correlation = np.corrcoef(fl[:min_len], bl[:min_len])[0, 1]
                    
                    if correlation > 0.7:
                        return "pace"  # Same side moves together
            
            # Check all paws together (bound/gallop)
            if len(paw_data) == 4:
                all_paws = list(paw_data.values())
                if all(len(p) > 10 for p in all_paws):
                    min_len = min(len(p) for p in all_paws)
                    all_paws_trimmed = [p[:min_len] for p in all_paws]
                    
                    # Check if all move together
                    correlations = []
                    for i in range(len(all_paws_trimmed)):
                        for j in range(i+1, len(all_paws_trimmed)):
                            corr = np.corrcoef(all_paws_trimmed[i], all_paws_trimmed[j])[0, 1]
                            correlations.append(corr)
                    
                    avg_correlation = np.mean(correlations)
                    if avg_correlation > 0.6:
                        return "bound"  # All paws move together
        
        return "walk"  # Default walking pattern
    
    def visualize_keypoints(self, output_path: str):
        """Visualize detected keypoints with animal mapping."""
        logger.info(f"\n📊 Creating visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot 1: Human keypoints on first frame
        ax1 = axes[0]
        if self.frames:
            ax1.imshow(self.frames[0])
            
            # Plot first frame human keypoints
            first_frame_kps = self.detect_keypoints(self.frames[0])
            for kp in first_frame_kps:
                if kp.confidence > 0.3:
                    ax1.scatter(kp.x, kp.y, s=100, c='red', alpha=kp.confidence)
                    ax1.text(kp.x + 5, kp.y - 5, kp.name[:3], color='white', 
                           fontsize=8, weight='bold')
        
        ax1.set_title('Human Pose Detection (YOLO)')
        ax1.axis('off')
        
        # Plot 2: Animal keypoints mapped
        ax2 = axes[1]
        if self.frames:
            ax2.imshow(self.frames[0])
            
            # Map to animal keypoints
            animal_kps = self.map_to_animal_keypoints(first_frame_kps)
            
            # Plot animal keypoints
            for name, kp in animal_kps.items():
                if kp.confidence > 0.3:
                    color = 'green' if 'paw' in name else 'blue'
                    ax2.scatter(kp.x, kp.y, s=120, c=color, alpha=kp.confidence)
                    ax2.text(kp.x + 5, kp.y - 5, name.replace('_', ' '), 
                           color='white', fontsize=8, weight='bold')
            
            # Draw animal skeleton
            for conn in self.ANIMAL_SKELETON_CONNECTIONS:
                kp1 = animal_kps.get(conn[0])
                kp2 = animal_kps.get(conn[1])
                
                if kp1 and kp2 and kp1.confidence > 0.3 and kp2.confidence > 0.3:
                    ax2.plot([kp1.x, kp2.x], [kp1.y, kp2.y], 'g-', alpha=0.5, linewidth=2)
        
        ax2.set_title('Animal Keypoint Mapping')
        ax2.axis('off')
        
        # Plot 3: Paw trajectories
        ax3 = axes[2]
        if self.frames:
            ax3.imshow(self.frames[0], alpha=0.3)
        
        # Plot paw trajectories with different colors
        paw_colors = {
            'front_left_paw': 'red',
            'front_right_paw': 'blue',
            'back_left_paw': 'orange',
            'back_right_paw': 'purple'
        }
        
        for name, color in paw_colors.items():
            if name in self.animal_keypoint_trajectories:
                traj = self.animal_keypoint_trajectories[name]
                valid_positions = [(p[0], p[1]) for p, c in zip(traj.positions, traj.confidences) 
                                 if c > 0.3 and p[0] > 0]
                
                if valid_positions:
                    xs, ys = zip(*valid_positions)
                    ax3.plot(xs, ys, '-', color=color, alpha=0.7, linewidth=2, label=name.replace('_', ' '))
        
        ax3.set_title('Paw Movement Trajectories')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.axis('off')
        
        # Plot 4: Gait analysis
        ax4 = axes[3]
        
        # Plot paw height over time
        for name, color in paw_colors.items():
            if name in self.animal_keypoint_trajectories:
                traj = self.animal_keypoint_trajectories[name]
                y_positions = [p[1] for p, c in zip(traj.positions, traj.confidences) 
                             if c > 0.3 and p[1] > 0]
                
                if y_positions:
                    ax4.plot(y_positions, '-', color=color, alpha=0.7, label=name.replace('_', ' '))
        
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Y Position (pixels)')
        ax4.set_title('Paw Height Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.invert_yaxis()  # Invert Y axis since Y increases downward in images
        
        # Plot 5: Detection confidence
        ax5 = axes[4]
        
        # Plot confidence for key animal parts
        key_parts = ['spine_center', 'head', 'front_left_paw', 'back_right_paw']
        for part in key_parts:
            if part in self.animal_keypoint_trajectories:
                traj = self.animal_keypoint_trajectories[part]
                if traj.confidences:
                    ax5.plot(traj.confidences, label=part.replace('_', ' '), alpha=0.7)
        
        ax5.set_xlabel('Frame')
        ax5.set_ylabel('Confidence')
        ax5.set_title('Animal Keypoint Detection Confidence')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])
        
        # Plot 6: Summary
        ax6 = axes[5]
        ax6.axis('off')
        
        summary_text = f"""Improved YOLO Motion Analysis Summary
        
Frames: {self.motion_data.get('num_frames', 0)}
Animal Detection Rate: {self.motion_data.get('animal_detection_rate', 0):.1%}
Detected Animal Keypoints: {len(self.motion_data.get('detected_keypoints', []))}
Gait Pattern: {self.motion_data.get('gait_pattern', 'unknown')}

Key Animal Parts Detected:
"""
        
        # Add per-keypoint quality
        for name, traj in self.animal_keypoint_trajectories.items():
            if traj.confidences:
                avg_conf = np.mean([c for c in traj.confidences if c > 0])
                if avg_conf > 0.1:
                    summary_text += f"  {name.replace('_', ' ')}: {avg_conf:.2f}\n"
                    
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Visualization saved to {output_path}")
    
    def create_tracking_gif(self, output_path: str, fps: int = 10, max_frames: Optional[int] = None):
        """Create an animated GIF showing tracking results with animal keypoints."""
        logger.info(f"\n🎬 Creating tracking GIF with animal pose...")
        
        frames_to_use = self.frames
        if max_frames and len(self.frames) > max_frames:
            indices = np.linspace(0, len(self.frames) - 1, max_frames, dtype=int)
            frames_to_use = [self.frames[i] for i in indices]
        
        annotated_frames = []
        
        for i, frame in enumerate(frames_to_use):
            vis_frame = frame.copy()
            
            # Detect keypoints and map to animal
            human_keypoints = self.detect_keypoints(frame)
            animal_keypoints = self.map_to_animal_keypoints(human_keypoints)
            
            # Draw animal keypoints
            for name, kp in animal_keypoints.items():
                if kp.confidence > 0.3:
                    # Different colors for different body parts
                    if 'paw' in name:
                        color = (255, 0, 0)  # Red for paws
                    elif 'head' in name or 'neck' in name:
                        color = (0, 255, 0)  # Green for head/neck
                    else:
                        color = (0, 0, 255)  # Blue for body
                    
                    cv2.circle(vis_frame, (int(kp.x), int(kp.y)), 8, color, -1)
                    cv2.putText(vis_frame, name.split('_')[-1], (int(kp.x) + 10, int(kp.y) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw animal skeleton
            for conn in self.ANIMAL_SKELETON_CONNECTIONS:
                kp1 = animal_keypoints.get(conn[0])
                kp2 = animal_keypoints.get(conn[1])
                
                if kp1 and kp2 and kp1.confidence > 0.3 and kp2.confidence > 0.3:
                    cv2.line(vis_frame, 
                           (int(kp1.x), int(kp1.y)),
                           (int(kp2.x), int(kp2.y)),
                           (0, 255, 255), 3)  # Yellow skeleton
            
            # Add frame info
            info_text = f"Frame: {i + 1}/{len(frames_to_use)} | Animal Pose Estimation"
            cv2.putText(vis_frame, info_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add gait pattern if detected
            if i > len(frames_to_use) // 4:  # Show after some frames
                gait = self.motion_data.get('gait_pattern', 'unknown')
                cv2.putText(vis_frame, f"Gait: {gait}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            annotated_frames.append(vis_frame)
        
        # Save as GIF
        imageio.mimsave(output_path, annotated_frames, fps=fps, loop=0)
        logger.info(f"✅ Tracking GIF saved to: {output_path}")


# Test function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test improved YOLO pose extraction")
    parser.add_argument("--gif", type=str, default="assets/gifs/chipi-chipi-chapa-chapa.gif",
                       help="Path to GIF file")
    parser.add_argument("--model", type=str, default="yolov8x-pose.pt",
                       help="YOLO model to use")
    parser.add_argument("--no-object-detection", action="store_true",
                       help="Disable object detection for animal localization")
    args = parser.parse_args()
    
    # Test extraction
    extractor = ImprovedYOLOPoseExtractor(
        args.gif, 
        model_name=args.model,
        use_object_detection=not args.no_object_detection
    )
    motion_data = extractor.analyze_motion()
    
    # Save visualization
    extractor.visualize_keypoints("outputs/improved_yolo_pose_test.png")
    extractor.create_tracking_gif("outputs/improved_yolo_tracking.gif")