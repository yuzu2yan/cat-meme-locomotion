# Cat Meme Locomotion 🐱🤖

猫のGIF動画（chipi-chipi-chapa-chapa）の動きをGenesis simulatorでUnitree Go2ロボットに再現するプロジェクト。

## Overview

Genesis公式のlocomotion実装をベースに、猫のバウンス動作を解析してロボットに適用します。

## Features

- **モーション抽出**: 猫のGIFから55個のバウンスピークを検出
- **3Dロボットシミュレーション**: Unitree Go2のURDFモデルとDAEメッシュファイルを使用
- **猫らしい動き**: トロット歩容（対角線上の脚が同期）でバウンス動作を再現
- **高速モーション**: 3倍速でGIFの動きに合わせた素早い動作

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd cat-meme-locomotion

# Install with uv
uv sync
```

## Usage

猫の動きをロボットで再現:

```bash
uv run cat-unitree
```

動作確認（モーション抽出のみ）:

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
- Genesis公式のlocomotion実装をベース
- PD制御: kp=30.0, kd=1.0（よりスナップの効いた動作）
- ジョイント順序: FR → FL → RR → RL
- 公式の立ち姿勢:
  - 前脚: thigh=0.8, calf=-1.5
  - 後脚: thigh=1.0, calf=-1.5
- トロット歩容: FR+RLとFL+RRが対角線上で同期
- 3倍速で猫のバウンス動作を再現

### Key Features
1. Genesis公式のシーン構築方法を採用
2. 正確なジョイント名とモーターインデックス
3. 高速化: 3倍速モーション再生
4. 増幅されたアンプリチュード（0.4-0.5）
5. 25 FPSで安定した制御

## Requirements

- Python >= 3.9
- Genesis-world >= 0.2.0
- CUDA-capable GPU
- Unitree Go2 mesh files (DAE format)

## Credits

- Genesis simulator by Genesis Embodied AI
- Unitree Go2 robot model
- Original cat meme: chipi-chipi-chapa-chapa