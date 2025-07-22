# Cat Meme Locomotion 🐱🤖

Replicate cat movements from GIF animations on Unitree Go2 robot in Genesis simulator.

## Overview

This project extracts motion patterns from animated GIFs and applies them to a Unitree Go2 quadruped robot using the Genesis physics simulator. Based on the official Genesis locomotion example, it creates realistic cat-like bouncing movements.

## Features

- **Motion Extraction**: Analyzes GIF frames to detect bounce patterns and peaks
- **3D Robot Simulation**: Uses Unitree Go2 URDF model with DAE mesh files
- **Cat-like Movement**: Implements trotting gait with synchronized diagonal legs
- **Configurable Speed**: Adjustable motion speed multiplier (default: 3x)
- **Custom GIF Support**: Use any GIF file as motion source

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd cat-meme-locomotion

# Install with uv
uv sync
```

## Usage

### Basic Usage

Run with default GIF (chipi-chipi-chapa-chapa):

```bash
uv run cat-unitree
```

### Custom GIF

Use your own GIF file:

```bash
uv run cat-unitree --gif path/to/your/cat.gif
```

### Adjust Speed

Change motion speed (default is 3x):

```bash
uv run cat-unitree --speed 2.0
```

### Combined Options

```bash
uv run cat-unitree --gif my_cat.gif --speed 4.0
```

### Extract Motion Only

Analyze GIF without running simulation:

```bash
uv run extract-motion
```

## Project Structure

```
cat-meme-locomotion/
├── assets/
│   └── gifs/
│       └── chipi-chipi-chapa-chapa.gif  # Default cat GIF
├── dae/                                   # Unitree Go2 mesh files
├── go2.urdf                              # Robot URDF definition
├── outputs/                              # Motion analysis results
├── src/
│   └── cat_meme_locomotion/
│       ├── core/
│       │   └── motion_extractor.py       # GIF analysis module
│       └── unitree_genesis_official.py   # Main robot controller
└── pyproject.toml                        # Project configuration
```

## Implementation Details

### Motion Extraction
- Uses OpenCV and PIL to process GIF frames
- Detects vertical motion using object detection
- Identifies bounce peaks with scipy signal processing
- Normalizes motion data for robot control

### Robot Control
- Based on Genesis official locomotion example
- PD control with kp=30.0, kd=1.0 for responsive movement
- 50Hz control frequency (25 FPS simulation)
- Joint ordering: FR → FL → RR → RL
- Standing pose:
  - Front legs: thigh=0.8, calf=-1.5
  - Rear legs: thigh=1.0, calf=-1.5

### Motion Mapping
- Trotting gait: FR+RL and FL+RR move together
- Dynamic amplitude: 0.4-0.5 based on bounce intensity
- Hip joints: minimal lateral movement
- Thigh joints: primary bounce drivers
- Calf joints: coordinated with thigh, phase-shifted

## Requirements

- Python >= 3.9
- CUDA-capable GPU
- Genesis-world >= 0.2.0
- OpenCV, PIL, NumPy, SciPy

## Command Line Options

```
usage: cat-unitree [-h] [--gif GIF] [--speed SPEED]

Unitree robot mimics cat motion from GIF

optional arguments:
  -h, --help     show this help message and exit
  --gif GIF      Path to the GIF file (default: assets/gifs/chipi-chipi-chapa-chapa.gif)
  --speed SPEED  Motion speed multiplier (default: 3.0)
```

## Tips

- For slower, more visible motion, use `--speed 1.0`
- For extremely fast motion, try `--speed 5.0` or higher
- GIFs with clear vertical motion work best
- Higher resolution GIFs provide better motion extraction

## Credits

- Genesis simulator by Genesis Embodied AI
- Unitree Go2 robot model
- Default animation: chipi-chipi-chapa-chapa cat meme

## License

See LICENSE file for details.