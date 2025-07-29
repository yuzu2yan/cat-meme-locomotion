#!/usr/bin/env python3
"""Test detection quality for different inputs."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from cat_meme_locomotion.core.yolo_pose_extractor import YOLOPoseExtractor
import cv2
import numpy as np


def analyze_detection_quality(file_path: str, model_name: str = "yolov8x-pose.pt"):
    """Analyze detection quality for a file."""
    print(f"\n🔍 Analyzing: {file_path}")
    print(f"   Model: {model_name}")
    
    # Extract with different settings
    extractor = YOLOPoseExtractor(
        file_path,
        model_name=model_name,
        max_frames=30,  # Analyze first 30 frames
        target_fps=10
    )
    
    # Extract frames
    frames = extractor.extract_frames()
    print(f"   Frames: {len(frames)}")
    
    # Analyze each frame
    detection_results = []
    
    for i, frame in enumerate(frames[:10]):  # Check first 10 frames
        keypoints = extractor.detect_keypoints(frame)
        num_detected = len([kp for kp in keypoints if kp.confidence > 0.3])
        avg_conf = np.mean([kp.confidence for kp in keypoints]) if keypoints else 0
        
        detection_results.append({
            'frame': i,
            'num_detected': num_detected,
            'avg_confidence': avg_conf,
            'frame_shape': frame.shape
        })
        
        print(f"   Frame {i}: {num_detected} keypoints, avg conf: {avg_conf:.3f}")
    
    # Check if resizing helps
    print("\n🔧 Testing with resized frames...")
    for scale in [0.5, 2.0]:
        frame = frames[0]
        h, w = frame.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(frame, new_size)
        
        keypoints = extractor.detect_keypoints(resized)
        num_detected = len([kp for kp in keypoints if kp.confidence > 0.3])
        avg_conf = np.mean([kp.confidence for kp in keypoints]) if keypoints else 0
        
        print(f"   Scale {scale}x ({new_size}): {num_detected} keypoints, conf: {avg_conf:.3f}")
    
    return detection_results


def main():
    """Test detection quality on different files."""
    test_files = [
        ("assets/gifs/dancing-dog.gif", "GIF"),
        ("assets/mp4/grey-kitten-lying.mp4", "MP4"),
        ("assets/mp4/stray-cat-looking-for-food.mp4", "MP4"),
    ]
    
    print("🐱 Detection Quality Analysis")
    print("=" * 50)
    
    # Test with different models
    models = [
        "yolov8x-pose.pt",  # Largest, most accurate
        "yolov8l-pose.pt",  # Large
        "yolov8m-pose.pt",  # Medium
    ]
    
    for file_path, file_type in test_files:
        if Path(file_path).exists():
            print(f"\n📁 {file_type}: {Path(file_path).name}")
            
            # Test with default model
            results = analyze_detection_quality(file_path, models[0])
            
            # Calculate average
            avg_detected = np.mean([r['num_detected'] for r in results])
            avg_confidence = np.mean([r['avg_confidence'] for r in results])
            
            print(f"\n📊 Summary:")
            print(f"   Average keypoints: {avg_detected:.1f}")
            print(f"   Average confidence: {avg_confidence:.3f}")


if __name__ == "__main__":
    main()