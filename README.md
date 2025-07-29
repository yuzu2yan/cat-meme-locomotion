# Cat Meme Locomotion 🐱🤖

Unitree Go2ロボットが猫のGIF/動画の動きを真似するコントローラー。複数の姿勢推定手法をサポートし、自動的にモーションキャプチャと学習結果を生成します。

<p align="center">
  <img src="assets/demo.gif" alt="Demo" width="600">
</p>

## 特徴 ✨

- 🎬 **GIFと動画（MP4）の両方に対応**
- 🤖 **複数の姿勢推定手法**:
  - **DeepLabCut (DLC)**: 動物専用のコンピュータビジョン手法
  - **YOLO**: 深層学習ベースの姿勢推定（人間用モデル）
  - **CV-Pose**: OpenCVベースのシンプルな動物姿勢推定
  - **Simple**: 直接的なキーポイントマッピング
- 📊 **自動生成される出力**:
  - モーションキャプチャGIF
  - キーポイント可視化
  - 学習メトリクス（信頼度、検出数の推移）
- 🎮 **リアルタイムシミュレーション**: Genesis物理エンジン使用

## インストール 🚀

### 必要要件

- Python 3.10+
- CUDA対応GPU（推奨）
- Ubuntu 20.04/22.04

### UV（推奨）を使用したインストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/cat-meme-locomotion.git
cd cat-meme-locomotion

# UVをインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係をインストール
uv pip install -e .
```

### 従来のpipを使用したインストール

```bash
# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# インストール
pip install -e .
```

## 使い方 🎮

### 基本的な使用方法

```bash
# DeepLabCut（動物に最適）でGIFを処理
uv run cat-locomotion dlc --gif assets/gifs/dancing-dog.gif

# YOLOでMP4動画を処理
uv run cat-locomotion yolo --gif assets/mp4/grey-kitten-lying.mp4

# パラメータを調整
uv run cat-locomotion dlc --gif assets/gifs/happy-cat.gif --speed 1.5 --amplitude 2.0
```

### 利用可能なコントローラー

| コントローラー | 説明 | 用途 |
|--------------|------|------|
| `dlc` | DeepLabCut風の動物姿勢推定 | 動物のGIF/動画に最適 |
| `yolo` | YOLOベースの姿勢推定 | 人間的な動きのGIFに適している |
| `cv-pose` | OpenCVベースの姿勢推定 | 外部依存なしで動作 |
| `simple` | シンプルな直接マッピング | 基本的な動き |
| `official` | オリジナルの拡張モーション抽出 | 従来の手法 |

### コマンドラインオプション

```bash
# ヘルプを表示
uv run cat-locomotion --help
uv run cat-locomotion dlc --help

# 共通オプション
--gif PATH          # 入力GIF/動画ファイルのパス
--speed FLOAT       # モーション速度倍率（デフォルト: 1.0）
--amplitude FLOAT   # モーション振幅倍率（デフォルト: 1.2）

# YOLOのみ
--model MODEL       # YOLOモデル（yolov8x-pose.pt, yolov8n-pose.pt等）
```

## 出力ファイル 📁

プログラムを実行すると、`outputs/`ディレクトリに以下のファイルが自動生成されます：

### DeepLabCut (DLC)
- `dlc_keypoints_*.png` - 検出されたキーポイントの可視化
- `dlc_tracking_*.gif` - モーションキャプチャのアニメーション
- `dlc_metrics_*/` - パフォーマンスメトリクス
  - `tracking_metrics.png` - 検出数と信頼度の推移
  - `confidence_heatmap.png` - キーポイント毎の信頼度ヒートマップ

### YOLO
- `yolo_keypoints_*.png` - YOLOで検出されたキーポイント
- `yolo_tracking_*.gif` - トラッキング結果のアニメーション
- `yolo_metrics_*/` - 検出メトリクス

## 姿勢推定手法の比較 🔍

| 特徴 | DeepLabCut (DLC) | YOLO |
|------|------------------|------|
| **検出方式** | コンピュータビジョン（SIFT、色検出、輪郭解析） | 深層学習（事前学習済みモデル） |
| **対象** | 動物専用に設計 | 人間用（COCO dataset） |
| **精度（GIF）** | 高い | 中程度 |
| **精度（MP4）** | 中程度 | 低い（動物には不適） |
| **処理速度** | 高速 | GPU使用時は高速 |
| **外部依存** | なし（OpenCV のみ） | ultralytics (YOLO) |

## トラブルシューティング 🔧

### ロボットが動かない場合
- すべてのコントローラーにフォールバックモーションが実装されているため、キーポイントが検出されなくても基本的な動きが生成されます
- `--amplitude`パラメータを大きくしてみてください（例: `--amplitude 2.0`）

### MP4の検出精度が低い場合
- DLCコントローラーを使用することを推奨します: `cat-locomotion dlc --gif video.mp4`
- より良い結果を得るには：
  - 動物が大きく映っている動画を使用
  - 背景がシンプルな動画を選択
  - 横向きの姿勢が多い動画が最適

### GPU関連のエラー
```bash
# CPUモードで実行（遅いが動作する）
CUDA_VISIBLE_DEVICES="" uv run cat-locomotion dlc --gif assets/gifs/happy-cat.gif
```

## プロジェクト構造 📂

```
cat-meme-locomotion/
├── src/cat_meme_locomotion/
│   ├── core/
│   │   ├── dlc_pose_extractor.py      # DeepLabCut風姿勢推定
│   │   ├── yolo_pose_extractor.py     # YOLO姿勢推定
│   │   └── motion_extractor.py        # 基本モーション抽出
│   ├── unitree_dlc_controller.py      # DLCロボットコントローラー
│   ├── unitree_yolo_controller.py     # YOLOロボットコントローラー
│   └── cli.py                         # コマンドラインインターフェース
├── assets/
│   ├── gifs/                          # サンプルGIFファイル
│   └── mp4/                           # サンプル動画ファイル
├── outputs/                           # 生成された出力ファイル
└── pyproject.toml                     # プロジェクト設定
```

## 開発 💻

### 開発環境のセットアップ

```bash
# 開発用依存関係をインストール
uv pip install -e ".[dev]"

# コードフォーマット
black src/
ruff check src/

# テスト実行
pytest tests/
```

### 新しいコントローラーの追加

1. `src/cat_meme_locomotion/`に新しいコントローラーファイルを作成
2. `cli.py`に新しいサブコマンドを追加
3. READMEを更新

## ライセンス 📄

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 謝辞 🙏

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) - 物理シミュレーション
- [Unitree Robotics](https://www.unitree.com/) - Go2ロボットモデル
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8実装

## 貢献 🤝

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容について議論してください。

1. フォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを開く

## 今後の改善予定 🚧

- [ ] 動物専用の姿勢推定モデルの実装
- [ ] リアルタイムウェブカメラ入力のサポート
- [ ] より多くのロボットモデルのサポート
- [ ] 3D姿勢推定の実装
- [ ] モーション学習とスタイル転送