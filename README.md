# Cat Meme Locomotion 🐱🤖

Replicate cat meme movements on Unitree Go2 robot in Genesis simulator.

## Overview

This project extracts motion patterns from a cat GIF (chipi-chipi-chapa-chapa) and applies them to a Unitree Go2 quadruped robot in the Genesis physics simulator.

## Features

- **Motion Extraction**: Analyzes cat GIF to extract bounce patterns and frequency
- **3D Robot Simulation**: Uses Unitree Go2 URDF model with proper mesh files
- **Cat-like Movement**: Implements trotting gait with bounce motion matching the cat
- **Real-time Visualization**: 60 FPS simulation with Genesis renderer

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd cat-meme-locomotion

# Install with uv
uv sync
```

## Usage

Run the cat motion simulation:

```bash
uv run cat-unitree
```

Extract motion data from GIF:

```bash
uv run extract-motion
```

## Project Structure

```
cat-meme-locomotion/
├── src/cat_meme_locomotion/
│   ├── core/
│   │   └── motion_extractor.py    # GIF motion analysis
│   ├── unitree_3d_final.py        # Main robot controller
│   └── __init__.py
├── assets/gifs/
│   └── chipi-chipi-chapa-chapa.gif # Source cat animation
├── dae/                            # Unitree Go2 mesh files
├── go2.urdf                        # Robot description
└── pyproject.toml                  # Project configuration
```

## Implementation Details

### Motion Extraction
- Uses OpenCV and PIL to analyze GIF frames
- Detects vertical motion and bounce peaks (55 peaks detected)
- Normalizes motion data for robot control

### Robot Control (Genesis Official Style)
- Based on Genesis official locomotion example
- PD control with kp=20.0, kd=0.5 (50Hz control frequency)
- Proper joint ordering: FR → FL → RR → RL
- Official standing pose:
  - Front legs: thigh=0.8, calf=-1.5
  - Rear legs: thigh=1.0, calf=-1.5
- Trotting gait: FR+RL and FL+RR move together
- Cat bounce motion applied on top of standing pose

### Key Improvements from Official Example
1. Uses Genesis scene building with n_envs=1
2. Proper joint naming and motor indexing
3. PD gains applied through set_dofs_kp/kv
4. Control via control_dofs_position method
5. Stable 25 FPS simulation matching control frequency

## Requirements

- Python >= 3.9
- Genesis-world >= 0.2.0
- CUDA-capable GPU
- Unitree Go2 mesh files (DAE format)

## Credits

- Genesis simulator by Genesis Embodied AI
- Unitree Go2 robot model
- Original cat meme: chipi-chipi-chapa-chapa