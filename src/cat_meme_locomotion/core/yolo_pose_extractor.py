"""YOLO-based animal pose estimation for motion extraction."""

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
from .tracking_visualizer import TrackingVisualizer, TrackingMetrics, VideoProcessor


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


class YOLOPoseExtractor:
    """Extract animal poses using YOLO v8 pose model."""
    
    # COCO animal keypoints (17 points)
    COCO_KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Animal-specific keypoint mapping for quadrupeds
    QUADRUPED_MAPPING = {
        'nose': 'nose',
        'left_eye': 'left_eye', 
        'right_eye': 'right_eye',
        'left_ear': 'left_ear',
        'right_ear': 'right_ear',
        'left_shoulder': 'left_front_shoulder',
        'right_shoulder': 'right_front_shoulder',
        'left_elbow': 'left_front_knee', 
        'right_elbow': 'right_front_knee',
        'left_wrist': 'left_front_paw',
        'right_wrist': 'right_front_paw', 
        'left_hip': 'left_rear_hip',
        'right_hip': 'right_rear_hip',
        'left_knee': 'left_rear_knee',
        'right_knee': 'right_rear_knee',
        'left_ankle': 'left_rear_paw',
        'right_ankle': 'right_rear_paw'
    }
    
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
    
    def __init__(self, input_path: str, model_name: str = 'yolov8x-pose.pt'):
        """Initialize YOLO pose extractor.
        
        Args:
            input_path: Path to input GIF or video file
            model_name: YOLO model name (default: yolov8x-pose.pt for best accuracy)
        """
        self.input_path = input_path
        self.frames: List[np.ndarray] = []
        self.keypoint_trajectories: Dict[str, KeypointTrajectory] = {}
        self.motion_data: Dict = {}
        
        # Initialize tracking visualizer
        self.visualizer = TrackingVisualizer()
        
        # Initialize YOLO model
        print(f"🔧 Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        
        # Check if GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📍 Using device: {self.device}")
        
    def extract_frames(self) -> List[np.ndarray]:
        """Extract all frames from the input file (GIF or video)."""
        file_ext = Path(self.input_path).suffix.lower()
        
        if file_ext == '.gif':
            # Extract from GIF
            gif = Image.open(self.input_path)
            self.frames = []
            
            try:
                while True:
                    frame = np.array(gif.convert('RGB'))
                    self.frames.append(frame)
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
                
            print(f"📽️ Extracted {len(self.frames)} frames from GIF")
            
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            # Extract from video
            self.frames = VideoProcessor.extract_frames_from_video(
                self.input_path, 
                max_frames=300,  # Limit for memory
                target_fps=15    # Reduce FPS for faster processing
            )
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        return self.frames
    
    def detect_keypoints(self, frame: np.ndarray) -> List[AnimalKeypoint]:
        """Detect keypoints in a single frame using YOLO."""
        # Run YOLO pose detection
        results = self.model(frame, verbose=False)
        
        detected_keypoints = []
        
        if len(results) > 0 and results[0].keypoints is not None:
            # Get keypoints from first detection (assuming one animal)
            keypoints = results[0].keypoints
            
            if keypoints.xy.shape[0] > 0:
                # Extract keypoint coordinates and confidence
                kp_xy = keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
                kp_conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(17)
                
                # Convert to AnimalKeypoint objects
                for i, name in enumerate(self.COCO_KEYPOINT_NAMES):
                    if i < len(kp_xy) and kp_conf[i] > 0.3:  # Confidence threshold
                        detected_keypoints.append(AnimalKeypoint(
                            name=name,
                            x=float(kp_xy[i, 0]),
                            y=float(kp_xy[i, 1]),
                            confidence=float(kp_conf[i])
                        ))
        
        # If YOLO doesn't detect enough keypoints, use fallback
        if len(detected_keypoints) < 10:
            detected_keypoints.extend(self._detect_keypoints_fallback(frame))
            
        return detected_keypoints
    
    def _detect_keypoints_fallback(self, frame: np.ndarray) -> List[AnimalKeypoint]:
        """Fallback keypoint detection using contour analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find edges and contours
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
            
        # Find largest contour (assume it's the cat)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Estimate keypoints based on bounding box
        fallback_keypoints = []
        
        # Head region (nose, eyes, ears)
        fallback_keypoints.extend([
            AnimalKeypoint('nose', x + w*0.1, y + h*0.3, 0.5),
            AnimalKeypoint('left_eye', x + w*0.2, y + h*0.2, 0.5),
            AnimalKeypoint('right_eye', x + w*0.2, y + h*0.4, 0.5),
        ])
        
        # Body keypoints
        fallback_keypoints.extend([
            AnimalKeypoint('left_shoulder', x + w*0.3, y + h*0.3, 0.5),
            AnimalKeypoint('right_shoulder', x + w*0.3, y + h*0.7, 0.5),
            AnimalKeypoint('left_hip', x + w*0.7, y + h*0.3, 0.5),
            AnimalKeypoint('right_hip', x + w*0.7, y + h*0.7, 0.5),
        ])
        
        # Paws
        fallback_keypoints.extend([
            AnimalKeypoint('left_wrist', x + w*0.2, y + h*0.2, 0.4),
            AnimalKeypoint('right_wrist', x + w*0.2, y + h*0.8, 0.4),
            AnimalKeypoint('left_ankle', x + w*0.8, y + h*0.2, 0.4),
            AnimalKeypoint('right_ankle', x + w*0.8, y + h*0.8, 0.4),
        ])
        
        return fallback_keypoints
    
    def analyze_motion(self) -> Dict[str, Any]:
        """Analyze motion from all frames."""
        print("\n🔍 Analyzing motion with YOLO pose detection...")
        
        # Extract frames if not already done
        if not self.frames:
            self.extract_frames()
            
        # Initialize keypoint trajectories
        for name in self.COCO_KEYPOINT_NAMES:
            self.keypoint_trajectories[name] = KeypointTrajectory(
                name=name,
                positions=[],
                confidences=[]
            )
        
        # Process each frame and collect metrics
        for i, frame in enumerate(self.frames):
            if i % 10 == 0:
                print(f"   Processing frame {i+1}/{len(self.frames)}...")
            
            start_time = time.time()
            keypoints = self.detect_keypoints(frame)
            comp_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Update trajectories
            detected_count = 0
            conf_sum = 0
            for kp in keypoints:
                self.keypoint_trajectories[kp.name].positions.append((kp.x, kp.y))
                self.keypoint_trajectories[kp.name].confidences.append(kp.confidence)
                detected_count += 1
                conf_sum += kp.confidence
            
            # Pad missing keypoints
            for name in self.COCO_KEYPOINT_NAMES:
                if name not in [kp.name for kp in keypoints]:
                    # Use last known position or center
                    if self.keypoint_trajectories[name].positions:
                        last_pos = self.keypoint_trajectories[name].positions[-1]
                        self.keypoint_trajectories[name].positions.append(last_pos)
                    else:
                        self.keypoint_trajectories[name].positions.append((frame.shape[1]//2, frame.shape[0]//2))
                    self.keypoint_trajectories[name].confidences.append(0.0)
            
            # Collect metrics
            avg_conf = conf_sum / detected_count if detected_count > 0 else 0
            metric = TrackingMetrics(
                frame_idx=i,
                num_detected=detected_count,
                avg_confidence=avg_conf,
                computation_time=comp_time
            )
            self.visualizer.add_metric(metric)
        
        # Analyze gait patterns
        gait_info = self._analyze_gait_pattern()
        
        # Create motion data summary
        self.motion_data = {
            'num_frames': len(self.frames),
            'detected_keypoints': list(self.keypoint_trajectories.keys()),
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
            if name in self.keypoint_trajectories
        }
        
        if len(paw_trajectories) < 4:
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
            corr1 = np.corrcoef(
                paw_heights.get('left_wrist', [0]),
                paw_heights.get('right_ankle', [0])
            )[0, 1]
            
            # Right front with left rear
            corr2 = np.corrcoef(
                paw_heights.get('right_wrist', [0]),
                paw_heights.get('left_ankle', [0])
            )[0, 1]
            
            return corr1 > 0.7 and corr2 > 0.7
        except:
            return False
    
    def _is_pace_gait(self, paw_heights: Dict[str, np.ndarray]) -> bool:
        """Check if gait is pace (lateral pairs move together)."""
        if len(paw_heights) < 4:
            return False
            
        try:
            # Left side correlation
            corr1 = np.corrcoef(
                paw_heights.get('left_wrist', [0]),
                paw_heights.get('left_ankle', [0])
            )[0, 1]
            
            # Right side correlation
            corr2 = np.corrcoef(
                paw_heights.get('right_wrist', [0]),
                paw_heights.get('right_ankle', [0])
            )[0, 1]
            
            return corr1 > 0.7 and corr2 > 0.7
        except:
            return False
    
    def _is_gallop_gait(self, paw_heights: Dict[str, np.ndarray]) -> bool:
        """Check if gait is gallop (all paws have similar phase)."""
        if len(paw_heights) < 4:
            return False
            
        try:
            # Check if all paws move in similar pattern
            all_heights = list(paw_heights.values())
            correlations = []
            
            for i in range(len(all_heights)-1):
                for j in range(i+1, len(all_heights)):
                    corr = np.corrcoef(all_heights[i], all_heights[j])[0, 1]
                    correlations.append(corr)
                    
            return np.mean(correlations) > 0.6
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
        ax1.set_title(f'YOLO Pose Detection (Frame {frame_idx})')
        
        # Draw keypoints
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.COCO_KEYPOINT_NAMES)))
        
        for i, name in enumerate(self.COCO_KEYPOINT_NAMES):
            if name in self.keypoint_trajectories and frame_idx < len(self.keypoint_trajectories[name].positions):
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
                frame_idx < len(self.keypoint_trajectories[kp2].positions)):
                
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

Keypoint Quality:
"""
        
        # Add per-keypoint quality
        for name, traj in self.keypoint_trajectories.items():
            if traj.confidences:
                avg_conf = np.mean(traj.confidences)
                if avg_conf > 0.1:
                    summary_text += f"  {name}: {avg_conf:.2f}\n"
                    
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to {output_path}")
    
    def create_tracking_gif(self, output_path: str, fps: int = 10):
        """Create an animated GIF showing tracking results.
        
        Args:
            output_path: Path to save the tracking GIF
            fps: Frames per second for the GIF
        """
        self.visualizer.create_tracking_gif(
            self.frames,
            self.keypoint_trajectories,
            output_path,
            fps=fps,
            show_trajectory=True,
            trajectory_length=10
        )
    
    def save_tracking_metrics(self, output_dir: str):
        """Save tracking metrics and visualizations.
        
        Args:
            output_dir: Directory to save the outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save tracking metrics plot
        metrics_path = output_dir / "tracking_metrics.png"
        self.visualizer.plot_tracking_metrics(str(metrics_path))
        
        # Save confidence heatmap
        heatmap_path = output_dir / "confidence_heatmap.png"
        self.visualizer.create_confidence_heatmap(
            self.keypoint_trajectories,
            str(heatmap_path)
        )
        
        print(f"📊 Tracking metrics saved to {output_dir}")


def test_yolo_pose():
    """Test YOLO pose extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test YOLO pose extraction")
    parser.add_argument("--gif", type=str, 
                       default="assets/gifs/happy-cat.gif",
                       help="Path to GIF file")
    parser.add_argument("--model", type=str,
                       default="yolov8x-pose.pt",
                       help="YOLO model to use")
    args = parser.parse_args()
    
    # Create extractor
    extractor = YOLOPoseExtractor(args.gif, model_name=args.model)
    
    # Analyze motion
    motion_data = extractor.analyze_motion()
    
    # Visualize results
    output_path = f"outputs/yolo_pose_{Path(args.gif).stem}.png"
    Path("outputs").mkdir(exist_ok=True)
    extractor.visualize_keypoints(output_path)
    
    return extractor, motion_data


if __name__ == "__main__":
    test_yolo_pose()