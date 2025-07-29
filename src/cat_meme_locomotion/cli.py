#!/usr/bin/env python3
"""Command-line interface for cat-meme-locomotion."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unitree robot controller that mimics cat movements from GIF animations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available controllers:
  cv-pose     : OpenCV-based animal pose estimation (no external dependencies)
  simple      : Simple direct keypoint-to-joint mapping
  official    : Original enhanced motion extractor
  dlc         : DeepLabCut-based animal pose estimation (computer vision)
  yolo        : YOLO-based pose estimation (human pose model)

Examples:
  # Run with OpenCV pose estimation (recommended)
  %(prog)s cv-pose --gif assets/gifs/dancing-dog.gif --amplitude 2.0 --visualize
  
  # Run with simple mapping
  %(prog)s simple --gif assets/gifs/happy-cat.gif --speed 1.5
  
  # Run with original controller
  %(prog)s official --gif assets/gifs/chipi-chipi-chapa-chapa.gif
"""
    )
    
    subparsers = parser.add_subparsers(dest='controller', help='Controller type')
    
    # CV-based pose controller
    cv_parser = subparsers.add_parser('cv-pose', help='OpenCV-based animal pose estimation')
    cv_parser.add_argument('--gif', type=str, required=True, help='Path to GIF file')
    cv_parser.add_argument('--speed', type=float, default=1.0, help='Motion speed multiplier')
    cv_parser.add_argument('--amplitude', type=float, default=1.5, help='Motion amplitude multiplier')
    cv_parser.add_argument('--visualize', action='store_true', help='Save keypoint visualization')
    
    # Simple controller
    simple_parser = subparsers.add_parser('simple', help='Simple direct mapping controller')
    simple_parser.add_argument('--gif', type=str, required=True, help='Path to GIF file')
    simple_parser.add_argument('--speed', type=float, default=1.0, help='Motion speed multiplier')
    simple_parser.add_argument('--visualize', action='store_true', help='Save keypoint visualization')
    
    # Official controller
    official_parser = subparsers.add_parser('official', help='Original enhanced controller')
    official_parser.add_argument('--gif', type=str, required=True, help='Path to GIF file')
    
    # DeepLabCut controller
    dlc_parser = subparsers.add_parser('dlc', help='DeepLabCut-based animal pose estimation')
    dlc_parser.add_argument('--gif', type=str, required=True, help='Path to GIF or video file')
    dlc_parser.add_argument('--speed', type=float, default=1.0, help='Motion speed multiplier')
    dlc_parser.add_argument('--amplitude', type=float, default=1.2, help='Motion amplitude multiplier')
    dlc_parser.add_argument('--visualize', action='store_true', help='Save keypoint visualization')
    
    # YOLO controller
    yolo_parser = subparsers.add_parser('yolo', help='YOLO-based pose estimation')
    yolo_parser.add_argument('--gif', type=str, required=True, help='Path to GIF or video file')
    yolo_parser.add_argument('--model', type=str, default='yolov8x-pose.pt', help='YOLO model')
    yolo_parser.add_argument('--speed', type=float, default=1.0, help='Motion speed multiplier')
    yolo_parser.add_argument('--amplitude', type=float, default=1.2, help='Motion amplitude multiplier')
    
    args = parser.parse_args()
    
    if not args.controller:
        parser.print_help()
        sys.exit(1)
    
    # Validate GIF path
    gif_path = Path(args.gif)
    if not gif_path.exists():
        print(f"❌ Error: GIF file not found: {gif_path}")
        sys.exit(1)
    
    # Import and run the appropriate controller
    if args.controller == 'cv-pose':
        from .unitree_cv_pose_controller import run_cv_pose_simulation
        sys.argv = ['cv-pose', '--gif', args.gif, '--speed', str(args.speed), 
                    '--amplitude', str(args.amplitude)]
        if args.visualize:
            sys.argv.append('--visualize')
        run_cv_pose_simulation()
        
    elif args.controller == 'simple':
        from .unitree_ik_simple import run_simple_ik_simulation
        sys.argv = ['simple', '--gif', args.gif, '--speed', str(args.speed)]
        if args.visualize:
            sys.argv.append('--visualize')
        run_simple_ik_simulation()
        
    elif args.controller == 'official':
        from .unitree_genesis_official import main as run_official
        sys.argv = ['official', '--gif', args.gif]
        run_official()
        
    elif args.controller == 'dlc':
        from .unitree_dlc_controller import run_dlc_simulation
        sys.argv = ['dlc', '--gif', args.gif, '--speed', str(args.speed), 
                    '--amplitude', str(args.amplitude)]
        if args.visualize:
            sys.argv.append('--visualize')
        run_dlc_simulation()
        
    elif args.controller == 'yolo':
        from .unitree_yolo_controller import run_yolo_simulation
        sys.argv = ['yolo', '--gif', args.gif, '--model', args.model,
                    '--speed', str(args.speed), '--amplitude', str(args.amplitude)]
        run_yolo_simulation()


if __name__ == '__main__':
    main()