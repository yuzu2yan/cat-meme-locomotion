"""Visualizer for tracking results and training progress."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import imageio
from dataclasses import dataclass


@dataclass
class TrackingMetrics:
    """Metrics for tracking performance over time."""
    frame_idx: int
    num_detected: int
    avg_confidence: float
    computation_time: float
    gait_phase: Optional[str] = None


class TrackingVisualizer:
    """Create visualizations for pose tracking results."""
    
    # Color palette for keypoints (RGB)
    KEYPOINT_COLORS = {
        'nose': (255, 0, 0),        # Red
        'left_eye': (255, 128, 0),  # Orange
        'right_eye': (255, 128, 0),
        'left_ear': (255, 255, 0),  # Yellow
        'right_ear': (255, 255, 0),
        'left_shoulder': (0, 255, 0),   # Green
        'right_shoulder': (0, 255, 0),
        'left_elbow': (0, 255, 128),    # Teal
        'right_elbow': (0, 255, 128),
        'left_wrist': (0, 255, 255),    # Cyan
        'right_wrist': (0, 255, 255),
        'left_hip': (0, 128, 255),      # Light Blue
        'right_hip': (0, 128, 255),
        'left_knee': (0, 0, 255),       # Blue
        'right_knee': (0, 0, 255),
        'left_ankle': (128, 0, 255),    # Purple
        'right_ankle': (128, 0, 255),
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
    
    def __init__(self):
        """Initialize the visualizer."""
        self.tracking_metrics: List[TrackingMetrics] = []
        
    def add_metric(self, metric: TrackingMetrics):
        """Add a tracking metric."""
        self.tracking_metrics.append(metric)
        
    def create_tracking_gif(self, 
                           frames: List[np.ndarray],
                           keypoint_trajectories: Dict[str, Any],
                           output_path: str,
                           fps: int = 10,
                           show_trajectory: bool = True,
                           trajectory_length: int = 10):
        """Create an animated GIF showing tracking results.
        
        Args:
            frames: List of video frames
            keypoint_trajectories: Keypoint trajectories from tracking
            output_path: Path to save the GIF
            fps: Frames per second for the GIF
            show_trajectory: Whether to show trailing trajectories
            trajectory_length: Number of past frames to show in trajectory
        """
        print(f"\nCreating tracking visualization GIF...")
        
        annotated_frames = []
        
        for frame_idx, frame in enumerate(frames):
            # Copy frame to avoid modifying original
            vis_frame = frame.copy()
            
            # Draw current keypoints
            for kp_name, trajectory in keypoint_trajectories.items():
                if frame_idx < len(trajectory.positions) and trajectory.positions:
                    x, y = trajectory.positions[frame_idx]
                    confidence = trajectory.confidences[frame_idx] if frame_idx < len(trajectory.confidences) else 1.0
                    
                    if confidence > 0.3:  # Only show confident detections
                        color = self.KEYPOINT_COLORS.get(kp_name, (255, 255, 255))
                        # Draw keypoint
                        cv2.circle(vis_frame, (int(x), int(y)), 5, color, -1)
                        cv2.circle(vis_frame, (int(x), int(y)), 7, color, 2)
                        
                        # Add keypoint label
                        cv2.putText(vis_frame, kp_name[:3], (int(x) + 10, int(y) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw skeleton
            for connection in self.SKELETON_CONNECTIONS:
                kp1, kp2 = connection
                if (kp1 in keypoint_trajectories and kp2 in keypoint_trajectories and
                    frame_idx < len(keypoint_trajectories[kp1].positions) and
                    frame_idx < len(keypoint_trajectories[kp2].positions)):
                    
                    pos1 = keypoint_trajectories[kp1].positions[frame_idx]
                    pos2 = keypoint_trajectories[kp2].positions[frame_idx]
                    conf1 = keypoint_trajectories[kp1].confidences[frame_idx]
                    conf2 = keypoint_trajectories[kp2].confidences[frame_idx]
                    
                    if conf1 > 0.3 and conf2 > 0.3:
                        cv2.line(vis_frame, 
                                (int(pos1[0]), int(pos1[1])),
                                (int(pos2[0]), int(pos2[1])),
                                (0, 255, 0), 2)
            
            # Draw trajectories
            if show_trajectory:
                for kp_name, trajectory in keypoint_trajectories.items():
                    color = self.KEYPOINT_COLORS.get(kp_name, (255, 255, 255))
                    
                    # Get trajectory points
                    start_idx = max(0, frame_idx - trajectory_length)
                    end_idx = min(frame_idx + 1, len(trajectory.positions))
                    
                    if end_idx - start_idx > 1:
                        points = []
                        for i in range(start_idx, end_idx):
                            if i < len(trajectory.positions) and trajectory.confidences[i] > 0.3:
                                points.append(trajectory.positions[i])
                        
                        if len(points) > 1:
                            # Draw trajectory as fading line
                            for i in range(len(points) - 1):
                                alpha = (i + 1) / len(points)  # Fade effect
                                pt1 = (int(points[i][0]), int(points[i][1]))
                                pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                                
                                # Create faded color
                                faded_color = tuple(int(c * alpha) for c in color)
                                cv2.line(vis_frame, pt1, pt2, faded_color, 1)
            
            # Add frame info
            info_text = f"Frame: {frame_idx + 1}/{len(frames)}"
            cv2.putText(vis_frame, info_text, (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add metric info if available
            if frame_idx < len(self.tracking_metrics):
                metric = self.tracking_metrics[frame_idx]
                metric_text = f"Detected: {metric.num_detected}, Conf: {metric.avg_confidence:.2f}"
                cv2.putText(vis_frame, metric_text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if metric.gait_phase:
                    gait_text = f"Gait: {metric.gait_phase}"
                    cv2.putText(vis_frame, gait_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            annotated_frames.append(vis_frame)
        
        # Save as GIF
        imageio.mimsave(output_path, annotated_frames, fps=fps, loop=0)
        print(f"Tracking GIF saved to: {output_path}")
        
    def plot_tracking_metrics(self, output_path: str):
        """Plot tracking metrics over time.
        
        Args:
            output_path: Path to save the plot
        """
        if not self.tracking_metrics:
            print("No tracking metrics to plot")
            return
            
        # Extract data
        frames = [m.frame_idx for m in self.tracking_metrics]
        num_detected = [m.num_detected for m in self.tracking_metrics]
        avg_confidence = [m.avg_confidence for m in self.tracking_metrics]
        comp_time = [m.computation_time for m in self.tracking_metrics]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle('Pose Tracking Performance Metrics', fontsize=14)
        
        # Plot 1: Number of keypoints detected
        axes[0].plot(frames, num_detected, 'b-', linewidth=2)
        axes[0].set_ylabel('Keypoints Detected')
        axes[0].set_ylim([0, max(num_detected) + 2])
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(frames, num_detected, alpha=0.3)
        
        # Plot 2: Average confidence
        axes[1].plot(frames, avg_confidence, 'g-', linewidth=2)
        axes[1].set_ylabel('Average Confidence')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
        axes[1].fill_between(frames, avg_confidence, alpha=0.3)
        axes[1].legend()
        
        # Plot 3: Computation time
        axes[2].plot(frames, comp_time, 'r-', linewidth=2)
        axes[2].set_ylabel('Computation Time (ms)')
        axes[2].set_xlabel('Frame Number')
        axes[2].grid(True, alpha=0.3)
        axes[2].fill_between(frames, comp_time, alpha=0.3)
        
        # Add gait phase annotations if available
        gait_phases = [(m.frame_idx, m.gait_phase) for m in self.tracking_metrics if m.gait_phase]
        if gait_phases:
            unique_gaits = list(set(g[1] for g in gait_phases))
            gait_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_gaits)))
            gait_color_map = dict(zip(unique_gaits, gait_colors))
            
            for frame_idx, gait in gait_phases:
                color = gait_color_map[gait]
                for ax in axes:
                    ax.axvline(x=frame_idx, color=color, alpha=0.2, linewidth=1)
            
            # Add legend for gait phases
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=gait_color_map[g], label=g) for g in unique_gaits]
            axes[0].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Tracking metrics plot saved to: {output_path}")
        
    def create_confidence_heatmap(self, 
                                 keypoint_trajectories: Dict[str, Any],
                                 output_path: str):
        """Create a heatmap showing confidence levels for each keypoint over time.
        
        Args:
            keypoint_trajectories: Keypoint trajectories with confidence values
            output_path: Path to save the heatmap
        """
        # Prepare data
        keypoint_names = list(keypoint_trajectories.keys())
        if not keypoint_names:
            print("No keypoint data for heatmap")
            return
            
        max_frames = max(len(traj.confidences) for traj in keypoint_trajectories.values())
        
        # Create confidence matrix
        confidence_matrix = np.zeros((len(keypoint_names), max_frames))
        
        for i, kp_name in enumerate(keypoint_names):
            traj = keypoint_trajectories[kp_name]
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
        ax.set_title('Keypoint Detection Confidence Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Confidence Score')
        
        # Add grid
        ax.set_xticks(np.arange(0, max_frames, max(1, max_frames // 20)))
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confidence heatmap saved to: {output_path}")


class VideoProcessor:
    """Process video files for pose tracking."""
    
    @staticmethod
    def extract_frames_from_video(video_path: str, 
                                 max_frames: Optional[int] = None,
                                 target_fps: Optional[int] = None) -> List[np.ndarray]:
        """Extract frames from a video file.
        
        Args:
            video_path: Path to video file (mp4, avi, etc.)
            max_frames: Maximum number of frames to extract
            target_fps: Target FPS for frame extraction (None = use original)
            
        Returns:
            List of frames as numpy arrays
        """
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
        
    @staticmethod
    def save_tracking_video(frames: List[np.ndarray],
                           keypoint_trajectories: Dict[str, Any],
                           output_path: str,
                           fps: int = 30):
        """Save tracking results as a video file.
        
        Args:
            frames: List of video frames
            keypoint_trajectories: Keypoint trajectories from tracking
            output_path: Path to save the video (mp4)
            fps: Frames per second for output video
        """
        if not frames:
            print("No frames to save")
            return
            
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        visualizer = TrackingVisualizer()
        
        for frame_idx, frame in enumerate(frames):
            # Draw tracking results on frame
            vis_frame = frame.copy()
            
            # Similar visualization as in create_tracking_gif but for video
            for kp_name, trajectory in keypoint_trajectories.items():
                if frame_idx < len(trajectory.positions):
                    x, y = trajectory.positions[frame_idx]
                    confidence = trajectory.confidences[frame_idx]
                    
                    if confidence > 0.3:
                        color = visualizer.KEYPOINT_COLORS.get(kp_name, (255, 255, 255))
                        cv2.circle(vis_frame, (int(x), int(y)), 5, color, -1)
            
            # Convert RGB to BGR for OpenCV
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            out.write(vis_frame_bgr)
        
        out.release()
        print(f"Tracking video saved to: {output_path}")