"""Extract motion patterns from cat GIF animations."""

import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple


class CatMotionExtractor:
    """Extract and analyze motion patterns from cat GIF files."""
    
    def __init__(self, gif_path: str):
        self.gif_path = gif_path
        self.frames: List[np.ndarray] = []
        self.motion_data: Dict = {}
        
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
            
        return self.frames
    
    def detect_cat_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect cat position in a single frame using improved detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            # Filter contours by area (remove very small ones)
            min_area = frame.shape[0] * frame.shape[1] * 0.01  # At least 1% of frame
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if valid_contours:
                # Get the largest contour
                largest_contour = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, cy
        
        # Fallback: if no contour found, use center of mass of non-background pixels
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        M = cv2.moments(binary)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
            
        return None
    
    def extract_motion_pattern(self) -> Dict:
        """Extract motion pattern from all frames."""
        self.extract_frames()
        positions = []
        
        for frame in self.frames:
            pos = self.detect_cat_position(frame)
            if pos:
                positions.append(pos)
        
        if positions:
            positions = np.array(positions)
            y_positions = positions[:, 1]
            
            # Normalize vertical positions
            y_normalized = (y_positions - np.min(y_positions)) / (
                np.max(y_positions) - np.min(y_positions)
            )
            
            # Find bounce peaks
            peaks, _ = find_peaks(-y_normalized, distance=5)
            
            self.motion_data = {
                'positions': positions,
                'y_normalized': y_normalized,
                'peaks': peaks,
                'frequency': len(peaks) / len(self.frames) if self.frames else 0,
                'amplitude': np.std(y_normalized),
                'num_frames': len(self.frames)
            }
        
        return self.motion_data