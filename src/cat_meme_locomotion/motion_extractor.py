#!/usr/bin/env python3
"""CLI script for motion extraction only."""

import sys
from pathlib import Path

from cat_meme_locomotion.core.motion_extractor import CatMotionExtractor
from cat_meme_locomotion.utils.visualization import plot_motion_analysis, create_motion_report


def analyze_gif(gif_path: str = "assets/gifs/chipi-chipi-chapa-chapa.gif") -> None:
    """Analyze GIF and extract motion patterns."""
    gif_path = Path(gif_path)
    
    if not gif_path.exists():
        print(f"Error: GIF file not found at {gif_path}")
        sys.exit(1)
    
    print(f"Analyzing {gif_path.name}...")
    extractor = CatMotionExtractor(str(gif_path))
    motion_data = extractor.extract_motion_pattern()
    
    if motion_data:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        plot_motion_analysis(
            motion_data,
            extractor.frames,
            output_dir / "motion_analysis.png"
        )
        
        report = create_motion_report(
            motion_data,
            output_dir / "motion_report.txt"
        )
        
        print("\n" + report)
        print(f"\nResults saved to {output_dir}/")
    else:
        print("Failed to extract motion data")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_gif(sys.argv[1])
    else:
        analyze_gif()