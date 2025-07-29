"""YOLO-based pose estimation for motion extraction."""

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


class YOLOPoseExtractor:
    """Extract poses using YOLO v8 pose model."""
    
    # COCO keypoints (17 points)
    COCO_KEYPOINT_NAMES = [
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
    
    def __init__(self, input_path: str, model_name: str = 'yolov8x-pose.pt', 
                 max_frames: Optional[int] = None, target_fps: Optional[int] = None):
        """Initialize YOLO pose extractor.
        
        Args:
            input_path: Path to input GIF or video file
            model_name: YOLO model name (default: yolov8x-pose.pt for best accuracy)
            max_frames: Maximum number of frames to process (useful for videos)
            target_fps: Target FPS for frame extraction (None = use original)
        """
        self.input_path = input_path
        self.frames: List[np.ndarray] = []
        self.keypoint_trajectories: Dict[str, KeypointTrajectory] = {}
        self.motion_data: Dict = {}
        self.max_frames = max_frames
        self.target_fps = target_fps
        
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
                frame_count = 0
                while True:
                    frame = np.array(gif.convert('RGB'))
                    self.frames.append(frame)
                    frame_count += 1
                    
                    # Check max_frames limit
                    if self.max_frames and frame_count >= self.max_frames:
                        break
                        
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
                
            print(f"📽️ Extracted {len(self.frames)} frames from GIF")
            
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            # Extract from video
            self.frames = self._extract_frames_from_video(
                self.input_path, 
                max_frames=self.max_frames or 300,  # Use provided or default limit
                target_fps=self.target_fps or 15    # Use provided or default FPS
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
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {total_frames} frames @ {fps:.1f} FPS")
        
        # Calculate frame skip for target FPS
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
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
            
        cap.release()
        
        print(f"Extracted {len(frames)} frames from video")
        return frames
        
    def detect_keypoints(self, frame: np.ndarray) -> List[AnimalKeypoint]:
        """Detect keypoints in a single frame using YOLO."""
        # Run YOLO pose detection with adjusted parameters
        results = self.model(frame, 
                           verbose=False,
                           conf=0.25,  # Lower confidence threshold
                           iou=0.45,   # IoU threshold for NMS
                           imgsz=640)  # Standard image size
        
        detected_keypoints = []
        
        if len(results) > 0 and results[0].keypoints is not None:
            # Get keypoints from first detection (assuming one subject)
            keypoints = results[0].keypoints
            
            if keypoints.xy.shape[0] > 0:
                # Extract keypoint coordinates and confidence
                kp_xy = keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
                kp_conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(17)
                
                # Convert to AnimalKeypoint objects
                for i, name in enumerate(self.COCO_KEYPOINT_NAMES):
                    if i < len(kp_xy):
                        # Include all keypoints, even with low confidence
                        detected_keypoints.append(AnimalKeypoint(
                            name=name,
                            x=float(kp_xy[i, 0]),
                            y=float(kp_xy[i, 1]),
                            confidence=float(kp_conf[i])
                        ))
        
        return detected_keypoints
    
    def analyze_motion(self) -> Dict:
        """Analyze motion from all frames."""
        print("\n🔍 Analyzing motion with YOLO pose detection...")
        
        # Extract frames if not already done
        if not self.frames:
            self.extract_frames()
        
        # Initialize trajectories
        for name in self.COCO_KEYPOINT_NAMES:
            self.keypoint_trajectories[name] = KeypointTrajectory(
                name=name,
                positions=[],
                confidences=[]
            )
        
        # Process each frame
        total_confidence = 0
        detected_counts = []
        
        for i, frame in enumerate(self.frames):
            if i % 10 == 0:
                print(f"   Processing frame {i+1}/{len(self.frames)}...")
            
            # Detect keypoints
            keypoints = self.detect_keypoints(frame)
            detected_counts.append(len([kp for kp in keypoints if kp.confidence > 0.3]))
            
            # Update trajectories
            detected_names = set()
            for kp in keypoints:
                if kp.name in self.keypoint_trajectories:
                    self.keypoint_trajectories[kp.name].positions.append((kp.x, kp.y))
                    self.keypoint_trajectories[kp.name].confidences.append(kp.confidence)
                    detected_names.add(kp.name)
                    if kp.confidence > 0.3:
                        total_confidence += kp.confidence
            
            # Add None for missing keypoints
            for name in self.COCO_KEYPOINT_NAMES:
                if name not in detected_names:
                    self.keypoint_trajectories[name].positions.append((0, 0))
                    self.keypoint_trajectories[name].confidences.append(0.0)
        
        # Analyze gait pattern
        gait_pattern = self._analyze_gait_pattern()
        
        # Calculate statistics
        avg_confidence = total_confidence / (len(self.frames) * len(self.COCO_KEYPOINT_NAMES))
        detected_keypoints = [name for name, traj in self.keypoint_trajectories.items() 
                            if any(c > 0.3 for c in traj.confidences)]
        
        self.motion_data = {
            'num_frames': len(self.frames),
            'detected_keypoints': detected_keypoints,
            'avg_confidence': avg_confidence,
            'gait_pattern': gait_pattern,
            'keypoint_trajectories': self.keypoint_trajectories
        }
        
        print(f"\n✅ Motion analysis complete!")
        print(f"   • Detected keypoints: {len(detected_keypoints)}")
        print(f"   • Gait pattern: {gait_pattern}")
        print(f"   • Average confidence: {avg_confidence:.2f}")
        
        return self.motion_data
    
    def _analyze_gait_pattern(self) -> str:
        """Analyze gait pattern from keypoint movements."""
        # Simple gait analysis based on ankle movements
        ankle_names = ['left_ankle', 'right_ankle']
        ankle_data = {}
        
        for name in ankle_names:
            if name in self.keypoint_trajectories:
                traj = self.keypoint_trajectories[name]
                if traj.positions:
                    # Calculate vertical movement
                    y_positions = [p[1] for p in traj.positions if p[1] > 0]
                    if y_positions:
                        ankle_data[name] = np.array(y_positions)
        
        if len(ankle_data) < 2:
            return "unknown"
        
        # Check phase relationship
        if 'left_ankle' in ankle_data and 'right_ankle' in ankle_data:
            left = ankle_data['left_ankle']
            right = ankle_data['right_ankle']
            
            if len(left) > 10 and len(right) > 10:
                # Simple correlation check
                min_len = min(len(left), len(right))
                correlation = np.corrcoef(left[:min_len], right[:min_len])[0, 1]
                
                if correlation > 0.6:
                    return "hop"  # Synchronized movement
                elif correlation < -0.6:
                    return "walk"  # Alternating movement
                else:
                    return "trot"  # Mixed pattern
        
        return "walk"
    
    def visualize_keypoints(self, output_path: str):
        """Visualize detected keypoints."""
        print(f"\n📊 Creating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: All keypoints on first frame
        ax1 = axes[0]
        if self.frames:
            ax1.imshow(self.frames[0])
            
            # Plot first frame keypoints
            first_frame_kps = self.detect_keypoints(self.frames[0])
            for kp in first_frame_kps:
                if kp.confidence > 0.3:
                    ax1.scatter(kp.x, kp.y, s=100, c='red', alpha=kp.confidence)
                    ax1.text(kp.x + 5, kp.y - 5, kp.name[:3], color='white', 
                           fontsize=8, weight='bold')
            
            # Draw skeleton
            for conn in self.SKELETON_CONNECTIONS:
                kp1 = next((kp for kp in first_frame_kps if kp.name == conn[0]), None)
                kp2 = next((kp for kp in first_frame_kps if kp.name == conn[1]), None)
                
                if kp1 and kp2 and kp1.confidence > 0.3 and kp2.confidence > 0.3:
                    ax1.plot([kp1.x, kp2.x], [kp1.y, kp2.y], 'g-', alpha=0.5, linewidth=2)
        
        ax1.set_title('YOLO Detected Keypoints (First Frame)')
        ax1.axis('off')
        
        # Plot 2: Trajectories
        ax2 = axes[1]
        if self.frames:
            ax2.imshow(self.frames[0], alpha=0.3)
        
        # Plot trajectories
        for name, traj in self.keypoint_trajectories.items():
            if traj.positions and any(c > 0.3 for c in traj.confidences):
                valid_positions = [(p[0], p[1]) for p, c in zip(traj.positions, traj.confidences) 
                                 if c > 0.3 and p[0] > 0]
                
                if valid_positions:
                    xs, ys = zip(*valid_positions)
                    ax2.plot(xs, ys, '-', alpha=0.7, linewidth=2, label=name)
        
        ax2.set_title('Keypoint Trajectories')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.axis('off')
        
        # Plot 3: Confidence over time
        ax3 = axes[2]
        for name, traj in self.keypoint_trajectories.items():
            if traj.confidences and max(traj.confidences) > 0.3:
                ax3.plot(traj.confidences, label=name, alpha=0.7)
        
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Confidence')
        ax3.set_title('Detection Confidence Over Time')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary text
        ax4 = axes[3]
        ax4.axis('off')
        
        summary_text = f"""YOLO Motion Analysis Summary
        
Frames: {self.motion_data.get('num_frames', 0)}
Detected Keypoints: {len(self.motion_data.get('detected_keypoints', []))}
Gait Pattern: {self.motion_data.get('gait_pattern', 'unknown')}
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
    
    def create_tracking_gif(self, output_path: str, fps: int = 10, max_frames: Optional[int] = None):
        """Create an animated GIF showing tracking results."""
        print(f"\n🎬 Creating tracking GIF...")
        
        # Limit frames if specified
        frames_to_use = self.frames
        if max_frames and len(self.frames) > max_frames:
            # Sample frames evenly
            indices = np.linspace(0, len(self.frames) - 1, max_frames, dtype=int)
            frames_to_use = [self.frames[i] for i in indices]
        
        annotated_frames = []
        
        for i, frame in enumerate(frames_to_use):
            # Copy frame
            vis_frame = frame.copy()
            
            # Detect keypoints for this frame
            keypoints = self.detect_keypoints(frame)
            
            # Draw keypoints
            for kp in keypoints:
                if kp.confidence > 0.3:
                    color = (255, 0, 0)  # Red
                    cv2.circle(vis_frame, (int(kp.x), int(kp.y)), 5, color, -1)
                    cv2.putText(vis_frame, kp.name[:3], (int(kp.x) + 5, int(kp.y) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw skeleton
            for conn in self.SKELETON_CONNECTIONS:
                kp1 = next((kp for kp in keypoints if kp.name == conn[0]), None)
                kp2 = next((kp for kp in keypoints if kp.name == conn[1]), None)
                
                if kp1 and kp2 and kp1.confidence > 0.3 and kp2.confidence > 0.3:
                    cv2.line(vis_frame, 
                           (int(kp1.x), int(kp1.y)),
                           (int(kp2.x), int(kp2.y)),
                           (0, 255, 0), 2)
            
            # Add frame info
            info_text = f"Frame: {i + 1}/{len(frames_to_use)} | YOLO Detection"
            cv2.putText(vis_frame, info_text, (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            annotated_frames.append(vis_frame)
        
        # Save as GIF
        imageio.mimsave(output_path, annotated_frames, fps=fps, loop=0)
        print(f"✅ Tracking GIF saved to: {output_path}")
    
    def save_tracking_metrics(self, output_dir: str):
        """Save tracking metrics and visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create confidence heatmap
        self._create_confidence_heatmap(f"{output_dir}/confidence_heatmap.png")
        
        # Create tracking metrics plot
        self._plot_tracking_metrics(f"{output_dir}/tracking_metrics.png")
        
        print(f"📊 Tracking metrics saved to {output_dir}")
    
    def _create_confidence_heatmap(self, output_path: str):
        """Create confidence heatmap."""
        keypoint_names = list(self.keypoint_trajectories.keys())
        if not keypoint_names:
            return
            
        max_frames = max(len(traj.confidences) for traj in self.keypoint_trajectories.values())
        
        # Create confidence matrix
        confidence_matrix = np.zeros((len(keypoint_names), max_frames))
        
        for i, kp_name in enumerate(keypoint_names):
            traj = self.keypoint_trajectories[kp_name]
            confidences = traj.confidences[:max_frames]
            confidence_matrix[i, :len(confidences)] = confidences
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(confidence_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_yticks(range(len(keypoint_names)))
        ax.set_yticklabels(keypoint_names)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Keypoint')
        ax.set_title('YOLO Keypoint Detection Confidence Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Confidence Score')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_tracking_metrics(self, output_path: str):
        """Plot tracking metrics over time."""
        # Calculate metrics per frame
        num_detected = []
        avg_confidence = []
        
        for i in range(len(self.frames)):
            detected = 0
            total_conf = 0
            
            for traj in self.keypoint_trajectories.values():
                if i < len(traj.confidences) and traj.confidences[i] > 0.3:
                    detected += 1
                    total_conf += traj.confidences[i]
            
            num_detected.append(detected)
            avg_confidence.append(total_conf / len(self.keypoint_trajectories) if detected > 0 else 0)
        
        # Create plots
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        
        # Plot 1: Number of keypoints detected
        axes[0].plot(num_detected, 'b-', linewidth=2)
        axes[0].set_ylabel('Keypoints Detected')
        axes[0].set_ylim([0, len(self.keypoint_trajectories)])
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(range(len(num_detected)), num_detected, alpha=0.3)
        
        # Plot 2: Average confidence
        axes[1].plot(avg_confidence, 'g-', linewidth=2)
        axes[1].set_ylabel('Average Confidence')
        axes[1].set_xlabel('Frame Number')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
        axes[1].fill_between(range(len(avg_confidence)), avg_confidence, alpha=0.3)
        axes[1].legend()
        
        plt.suptitle('YOLO Pose Tracking Performance Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# Test function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test YOLO pose extraction")
    parser.add_argument("--gif", type=str, default="assets/gifs/chipi-chipi-chapa-chapa.gif",
                       help="Path to GIF file")
    parser.add_argument("--model", type=str, default="yolov8x-pose.pt",
                       help="YOLO model to use")
    args = parser.parse_args()
    
    # Test extraction
    extractor = YOLOPoseExtractor(args.gif, model_name=args.model)
    motion_data = extractor.analyze_motion()
    
    # Save visualization
    extractor.visualize_keypoints("outputs/yolo_pose_test.png")