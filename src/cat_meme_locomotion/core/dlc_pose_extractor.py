"""DeepLabCut-style animal pose estimation for motion extraction."""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
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
    """Represents a detected keypoint for animals."""
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


class DeepLabCutPoseExtractor:
    """Extract animal poses using DeepLabCut-style approach."""
    
    # Animal-specific keypoints for quadrupeds
    ANIMAL_KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'neck', 'left_shoulder', 'right_shoulder', 
        'left_front_paw', 'right_front_paw',
        'spine_center', 'left_hip', 'right_hip',
        'left_rear_paw', 'right_rear_paw',
        'tail_base', 'tail_tip'
    ]
    
    # Skeleton connections
    SKELETON_CONNECTIONS = [
        ('nose', 'neck'), ('neck', 'spine_center'), ('spine_center', 'tail_base'),
        ('neck', 'left_shoulder'), ('left_shoulder', 'left_front_paw'),
        ('neck', 'right_shoulder'), ('right_shoulder', 'right_front_paw'),
        ('spine_center', 'left_hip'), ('left_hip', 'left_rear_paw'),
        ('spine_center', 'right_hip'), ('right_hip', 'right_rear_paw'),
        ('tail_base', 'tail_tip'),
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear')
    ]
    
    def __init__(self, input_path: str, max_frames: Optional[int] = None, 
                 target_fps: Optional[int] = None):
        """Initialize DeepLabCut pose extractor.
        
        Args:
            input_path: Path to input GIF or video file
            max_frames: Maximum number of frames to process
            target_fps: Target FPS for frame extraction
        """
        self.input_path = input_path
        self.frames: List[np.ndarray] = []
        self.keypoint_trajectories: Dict[str, KeypointTrajectory] = {}
        self.motion_data: Dict = {}
        self.max_frames = max_frames
        self.target_fps = target_fps
        
        # Check if GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📍 Using device: {self.device}")
        
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
                
            print(f"📽️ Extracted {len(self.frames)} frames from GIF")
            
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
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {total_frames} frames @ {fps:.1f} FPS")
        
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
        
        print(f"✅ Extracted {len(frames)} frames from video")
        return frames
    
    def detect_keypoints(self, frame: np.ndarray) -> List[AnimalKeypoint]:
        """Detect keypoints in a single frame using computer vision."""
        # Preprocess frame
        processed = self._preprocess_frame(frame)
        
        # Use multiple detection methods
        keypoints = []
        
        # 1. Feature-based detection
        feature_keypoints = self._detect_features(processed)
        keypoints.extend(feature_keypoints)
        
        # 2. Color-based detection for specific parts
        color_keypoints = self._detect_by_color(processed, frame)
        keypoints.extend(color_keypoints)
        
        # 3. Contour-based detection
        contour_keypoints = self._detect_by_contours(processed)
        keypoints.extend(contour_keypoints)
        
        # Merge and filter keypoints
        keypoints = self._merge_keypoints(keypoints)
        
        return keypoints
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        return denoised
    
    def _detect_features(self, gray: np.ndarray) -> List[AnimalKeypoint]:
        """Detect keypoints using feature detection."""
        keypoints = []
        
        # Use SIFT for feature detection
        sift = cv2.SIFT_create(nfeatures=100)
        kp, desc = sift.detectAndCompute(gray, None)
        
        # Sort by response strength
        kp = sorted(kp, key=lambda x: x.response, reverse=True)
        
        # Map strongest features to body parts
        if len(kp) > 0:
            # Assume strongest feature is nose/head
            keypoints.append(AnimalKeypoint(
                name='nose',
                x=kp[0].pt[0],
                y=kp[0].pt[1],
                confidence=min(kp[0].response / 100, 1.0)
            ))
        
        # Group features by spatial proximity
        if len(kp) > 5:
            # Find eye candidates (paired features near nose)
            nose_pos = np.array([kp[0].pt[0], kp[0].pt[1]])
            eye_candidates = []
            
            for i in range(1, min(len(kp), 20)):
                dist = np.linalg.norm(np.array(kp[i].pt) - nose_pos)
                if 10 < dist < 50:  # Reasonable distance for eyes
                    eye_candidates.append(kp[i])
            
            if len(eye_candidates) >= 2:
                # Sort by x-coordinate
                eye_candidates.sort(key=lambda k: k.pt[0])
                
                keypoints.append(AnimalKeypoint(
                    name='left_eye',
                    x=eye_candidates[0].pt[0],
                    y=eye_candidates[0].pt[1],
                    confidence=min(eye_candidates[0].response / 100, 1.0) * 0.8
                ))
                
                keypoints.append(AnimalKeypoint(
                    name='right_eye',
                    x=eye_candidates[-1].pt[0],
                    y=eye_candidates[-1].pt[1],
                    confidence=min(eye_candidates[-1].response / 100, 1.0) * 0.8
                ))
        
        return keypoints
    
    def _detect_by_color(self, gray: np.ndarray, color_frame: np.ndarray) -> List[AnimalKeypoint]:
        """Detect keypoints using color information."""
        keypoints = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_RGB2HSV)
        
        # Detect dark regions (potential eyes, nose)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        
        # Find contours in dark regions
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        significant_contours = [c for c in contours if cv2.contourArea(c) > 20]
        
        # Sort by area
        significant_contours.sort(key=cv2.contourArea, reverse=True)
        
        # Use top contours as potential facial features
        for i, contour in enumerate(significant_contours[:3]):
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                area = cv2.contourArea(contour)
                confidence = min(area / 500, 1.0) * 0.6
                
                if i == 0:
                    name = 'nose'
                elif i == 1:
                    name = 'left_eye'
                else:
                    name = 'right_eye'
                
                keypoints.append(AnimalKeypoint(
                    name=name,
                    x=cx,
                    y=cy,
                    confidence=confidence
                ))
        
        return keypoints
    
    def _detect_by_contours(self, gray: np.ndarray) -> List[AnimalKeypoint]:
        """Detect keypoints using contour analysis."""
        keypoints = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return keypoints
        
        # Find largest contour (likely the animal body)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Estimate body parts based on bounding box
        # Neck (front-center of body)
        keypoints.append(AnimalKeypoint(
            name='neck',
            x=x + w * 0.3,
            y=y + h * 0.3,
            confidence=0.5
        ))
        
        # Spine center
        keypoints.append(AnimalKeypoint(
            name='spine_center',
            x=x + w * 0.5,
            y=y + h * 0.5,
            confidence=0.5
        ))
        
        # Tail base
        keypoints.append(AnimalKeypoint(
            name='tail_base',
            x=x + w * 0.8,
            y=y + h * 0.6,
            confidence=0.4
        ))
        
        # Estimate limbs from contour extremes
        # Find convex hull
        hull = cv2.convexHull(largest_contour)
        
        # Get extreme points
        leftmost = tuple(hull[hull[:,:,0].argmin()][0])
        rightmost = tuple(hull[hull[:,:,0].argmax()][0])
        topmost = tuple(hull[hull[:,:,1].argmin()][0])
        bottommost = tuple(hull[hull[:,:,1].argmax()][0])
        
        # Map extremes to potential limbs
        keypoints.extend([
            AnimalKeypoint('left_front_paw', leftmost[0], leftmost[1], 0.4),
            AnimalKeypoint('right_rear_paw', rightmost[0], rightmost[1], 0.4),
        ])
        
        return keypoints
    
    def _merge_keypoints(self, keypoints: List[AnimalKeypoint]) -> List[AnimalKeypoint]:
        """Merge and filter duplicate keypoints."""
        if not keypoints:
            return []
        
        # Group by name
        grouped = {}
        for kp in keypoints:
            if kp.name not in grouped:
                grouped[kp.name] = []
            grouped[kp.name].append(kp)
        
        # Average positions for each group
        merged = []
        for name, kps in grouped.items():
            if kps:
                avg_x = np.mean([kp.x for kp in kps])
                avg_y = np.mean([kp.y for kp in kps])
                max_conf = max(kp.confidence for kp in kps)
                
                merged.append(AnimalKeypoint(
                    name=name,
                    x=avg_x,
                    y=avg_y,
                    confidence=max_conf
                ))
        
        return merged
    
    def analyze_motion(self) -> Dict:
        """Analyze motion from all frames."""
        print("\n🔍 Analyzing motion with DeepLabCut-style detection...")
        
        # Extract frames if not already done
        if not self.frames:
            self.extract_frames()
        
        # Initialize trajectories
        for name in self.ANIMAL_KEYPOINT_NAMES:
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
            detected_counts.append(len(keypoints))
            
            # Update trajectories
            detected_names = set()
            for kp in keypoints:
                if kp.name in self.keypoint_trajectories:
                    self.keypoint_trajectories[kp.name].positions.append((kp.x, kp.y))
                    self.keypoint_trajectories[kp.name].confidences.append(kp.confidence)
                    detected_names.add(kp.name)
                    total_confidence += kp.confidence
            
            # Add None for missing keypoints
            for name in self.ANIMAL_KEYPOINT_NAMES:
                if name not in detected_names:
                    self.keypoint_trajectories[name].positions.append((0, 0))
                    self.keypoint_trajectories[name].confidences.append(0.0)
        
        # Analyze gait pattern
        gait_pattern = self._analyze_gait_pattern()
        
        # Calculate statistics
        avg_confidence = total_confidence / (len(self.frames) * len(self.ANIMAL_KEYPOINT_NAMES))
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
        """Analyze gait pattern from paw movements."""
        # Check if we have paw data
        paw_names = ['left_front_paw', 'right_front_paw', 'left_rear_paw', 'right_rear_paw']
        paw_data = {}
        
        for name in paw_names:
            if name in self.keypoint_trajectories:
                traj = self.keypoint_trajectories[name]
                if traj.positions:
                    # Calculate vertical movement
                    y_positions = [p[1] for p in traj.positions if p[1] > 0]
                    if y_positions:
                        paw_data[name] = np.array(y_positions)
        
        if len(paw_data) < 2:
            return "unknown"
        
        # Simple gait classification based on phase relationships
        # This is a simplified version - real gait analysis would be more complex
        if len(paw_data) >= 4:
            # Check for diagonal synchronization (trot)
            try:
                lf = paw_data.get('left_front_paw', np.array([]))
                rr = paw_data.get('right_rear_paw', np.array([]))
                
                if len(lf) > 10 and len(rr) > 10:
                    correlation = np.corrcoef(lf[:min(len(lf), len(rr))], 
                                            rr[:min(len(lf), len(rr))])[0, 1]
                    
                    if correlation > 0.6:
                        return "trot"
                    elif correlation < -0.6:
                        return "pace"
                    else:
                        return "walk"
            except:
                pass
        
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
        
        ax1.set_title('Detected Keypoints (First Frame)')
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
        
        summary_text = f"""DeepLabCut-style Motion Analysis Summary
        
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
    
    def create_tracking_gif(self, output_path: str, fps: int = 10):
        """Create animated GIF showing tracking results."""
        print(f"\n🎬 Creating tracking GIF...")
        
        annotated_frames = []
        
        for i, frame in enumerate(self.frames):
            # Copy frame
            vis_frame = frame.copy()
            
            # Detect keypoints for this frame
            keypoints = self.detect_keypoints(frame)
            
            # Draw keypoints
            for kp in keypoints:
                if kp.confidence > 0.3:
                    color = (255, 0, 0)  # Red for all keypoints
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
            info_text = f"Frame: {i + 1}/{len(self.frames)}"
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
        ax.set_title('Keypoint Detection Confidence Heatmap')
        
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
        
        plt.suptitle('Pose Tracking Performance Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# Test function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DeepLabCut pose extraction")
    parser.add_argument("--gif", type=str, default="assets/gifs/chipi-chipi-chapa-chapa.gif",
                       help="Path to GIF or video file")
    args = parser.parse_args()
    
    # Test extraction
    extractor = DeepLabCutPoseExtractor(args.gif)
    motion_data = extractor.analyze_motion()
    
    # Save visualization
    extractor.visualize_keypoints("outputs/dlc_pose_test.png")