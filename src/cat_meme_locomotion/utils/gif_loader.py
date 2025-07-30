"""Utility for loading GIF files."""

import numpy as np
from PIL import Image
from typing import List
import logging

logger = logging.getLogger(__name__)

class GifLoader:
    """Load frames from GIF files."""
    
    def load_gif(self, gif_path: str) -> List[np.ndarray]:
        """Load all frames from a GIF file.
        
        Args:
            gif_path: Path to the GIF file
            
        Returns:
            List of frames as numpy arrays (RGB format)
        """
        try:
            gif = Image.open(gif_path)
            frames = []
            
            while True:
                # Convert to RGB (in case GIF has transparency)
                frame = np.array(gif.convert('RGB'))
                frames.append(frame)
                
                try:
                    gif.seek(gif.tell() + 1)
                except EOFError:
                    break
            
            logger.info(f"Loaded {len(frames)} frames from {gif_path}")
            return frames
            
        except Exception as e:
            logger.error(f"Failed to load GIF: {e}")
            return []