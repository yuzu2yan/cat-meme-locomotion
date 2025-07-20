# 🐱 Cat Meme Locomotion

Replicate cat meme movements on a Unitree quadruped robot in Genesis simulator using motion extraction from GIF animations.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cat-meme-locomotion.git
cd cat-meme-locomotion

# Install with uv
uv pip install -e .
```

### Usage

```bash
# Run Unitree robot simulation with cat motion
uv run cat-unitree

# Extract motion data only
uv run extract-motion
```

### Features

- 🎯 **Motion Extraction**: Analyzes cat GIF to extract bounce patterns
- 🤖 **3D Simulation**: Unitree Go1 robot replicates cat movements in Genesis
- 🎮 **Real-time Control**: Dynamic joint control based on motion data
- 📊 **Motion Analysis**: Generates detailed motion statistics

### Make Commands

```bash
make install    # Install dependencies
make run        # Run Unitree simulation
make extract    # Extract motion only
make clean      # Clean generated files
```

## 📁 Project Structure

```
cat-meme-locomotion/
├── src/
│   └── cat_meme_locomotion/
│       ├── core/                  # Core modules
│       │   └── motion_extractor.py
│       ├── main.py               # Entry point
│       ├── unitree_3d.py         # Unitree robot controller
│       └── motion_extractor.py   # CLI for motion extraction
├── assets/
│   ├── gifs/                     # Input GIF files
│   └── models/                   # Robot URDF files
│       └── unitree/              # Unitree robot models
├── outputs/                      # Generated outputs
├── pyproject.toml               # Project configuration
└── Makefile                     # Convenience commands
```

## 🛠️ Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run linters
make lint

# Format code
make format

# Run tests
make test
```

## 📊 How It Works

1. **Motion Extraction**: Analyzes cat GIF frames to detect vertical movement patterns
2. **Pattern Analysis**: Identifies bounce peaks and movement frequency
3. **Motion Mapping**: Converts cat movements to Spot robot joint trajectories
4. **Real-time Simulation**: Spot robot mimics the cat's dance in Genesis simulator

## 📝 License

MIT License - see LICENSE file for details