# Cat Meme Locomotion - 詳細コード説明（日本語）

## プロジェクト概要

このプロジェクトは、GIFアニメーションや動画から動物の動きを抽出し、Unitree Go2ロボットに模倣させるシステムです。

### 主な特徴
- OpenCVベースの高速姿勢推定（外部依存なし）
- Ultralytics YOLOを使用した本物の姿勢推定（人間→動物キーポイントマッピング）
- Genesis物理シミュレーションフレームワーク
- リアルタイム動作制御（100Hz）

## アーキテクチャ概要

```
[GIF/動画入力] → [姿勢推定器] → [キーポイント] → [コントローラー] → [ロボット動作]
                     ↓                                    ↓
                [可視化出力]                        [物理シミュレーション]
```

## 主要コンポーネント

### 1. YOLO姿勢推定器 (simple_yolo_pose_extractor.py)

#### クラス構造

```python
class SimpleYOLOPoseExtractor:
    """YOLOを使用した姿勢推定（可視化なし）"""
    
    def __init__(self, input_path: str, model_name: str = 'yolov8x-pose.pt'):
        self.pose_model = YOLO(model_name)  # 姿勢推定モデル
        self.detect_model = YOLO('yolov8n.pt')  # 動物検出用
```

#### 人間→動物キーポイントマッピング

YOLOは人間の姿勢推定モデルしか提供していないため、人間のキーポイントを動物の体の部位にマッピングします：

```python
ANIMAL_KEYPOINT_MAPPING = {
    # 頭部・首
    'head': ['nose'],
    'neck': ['nose', 'left_shoulder', 'right_shoulder'],
    
    # 前脚（腕からマッピング）
    'front_left_paw': ['left_wrist'],
    'front_right_paw': ['right_wrist'],
    
    # 後脚（脚からマッピング）
    'back_left_paw': ['left_ankle'],
    'back_right_paw': ['right_ankle'],
    
    # 体の中心
    'spine_center': ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']
}
```

### 2. OpenCV姿勢推定器 (cv_animal_pose_extractor.py)

#### 特徴
- 機械学習モデル不要
- 高速処理（リアルタイム対応）
- 色とコントラストベースの検出

```python
class CVAnimalPoseExtractor:
    def extract_keypoints(self, image):
        # 1. HSV色空間でのマスク作成
        # 2. エッジ検出（Canny）
        # 3. 輪郭検出
        # 4. 重心と端点からキーポイント推定
```

### 3. ロボットコントローラー

#### 3.1 YOLOコントローラー (unitree_yolo_controller.py)

```python
class UnitreeYOLOController:
    def apply_yolo_motion(self, motion_data, keypoint_trajectories, 
                         animal_keypoint_trajectories):
        if self.use_animal_mapping:
            # 動物キーポイントを使用
            self._apply_animal_motion(target_dof_pos, normalized_trajectories, 
                                    motion_frame, phase)
        else:
            # 人間キーポイントを直接使用
            self._apply_human_motion(target_dof_pos, normalized_trajectories, 
                                   motion_frame, phase)
```

動物モーション適用の例：
```python
def _apply_animal_motion(self, target_dof_pos, normalized_trajectories, 
                       motion_frame, phase):
    # 前脚の動き
    if 'front_left_paw' in normalized_trajectories:
        y_norm = traj[1]  # Y座標（高さ）
        target_dof_pos[4] = 0.3 + self.motion_amplitude * 0.8 * (1 - y_norm)  # 太もも
        target_dof_pos[5] = -2.0 + self.motion_amplitude * 0.6 * y_norm  # ふくらはぎ
```

#### 3.2 CVコントローラー (unitree_cv_pose_controller.py)

高速で安定した動作を実現：
```python
class UnitreeCVPoseController:
    def _keypoints_to_joint_targets_opencv_style(self, keypoints, image_shape, 
                                                motion_frame, last_target):
        # 1. 体の傾きから前後バランスを計算
        # 2. 足の位置から歩行パターンを生成
        # 3. 頭の動きから方向を決定
```

### 4. 歩行パターン分析

```python
def _analyze_gait_pattern(self) -> str:
    # 足の動きの相関を分析
    if correlation > 0.7:
        return "trot"  # 対角の足が一緒に動く
    elif lateral_correlation > 0.7:
        return "pace"  # 同じ側の足が一緒に動く
    elif all_correlation > 0.6:
        return "bound"  # すべての足が一緒に動く
    else:
        return "walk"  # 通常の歩行
```

## 実行フロー

1. **入力処理**
   ```python
   # GIFまたは動画からフレーム抽出
   frames = extractor.extract_frames()
   ```

2. **姿勢推定**
   ```python
   # 各フレームでキーポイント検出
   for frame in frames:
       keypoints = extractor.detect_keypoints(frame)
       animal_keypoints = extractor.map_to_animal_keypoints(keypoints)
   ```

3. **モーション生成**
   ```python
   # キーポイントからロボットの関節角度を計算
   target_angles = controller._keypoints_to_joint_targets(keypoints)
   ```

4. **シミュレーション**
   ```python
   # Genesis物理エンジンでロボット制御
   robot.control_dofs_position(target_angles, motor_indices)
   scene.step()
   ```

## コマンドライン使用例

```bash
# YOLOを使用（本物の姿勢推定）
uv run cat-locomotion yolo --gif assets/gifs/dancing-dog.gif \
    --model yolov8n-pose.pt --speed 0.5 --amplitude 0.8

# OpenCVを使用（高速・軽量）
uv run cat-locomotion cv-pose --gif assets/gifs/happy-cat.gif \
    --amplitude 1.5 --visualize
```

## パフォーマンス最適化

1. **GPUアクセラレーション**
   - YOLO推論にCUDAを使用
   - Genesis物理演算にGPUを活用

2. **フレームスキップ**
   - 動画の場合、適切なFPSにダウンサンプリング
   - リアルタイム処理のための最適化

3. **キャッシング**
   - YOLOモデルの事前ロード
   - 正規化されたトラジェクトリの事前計算

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   - 解決策：より小さいYOLOモデル（yolov8n-pose.pt）を使用
   - フレーム数を制限（--max-frames オプション）

2. **セグメンテーションフォルト**
   - 原因：matplotlib可視化の問題
   - 解決策：simple_yolo_pose_extractorを使用（可視化無効）

3. **ロボットが浮く**
   - 原因：地面との接触不良
   - 解決策：初期高さとPDゲインの調整

## 今後の改善案

1. **動物専用の姿勢推定モデル**
   - 犬・猫専用のカスタムYOLOモデルのトレーニング
   - より正確なキーポイント検出

2. **リアルタイムカメラ入力**
   - ウェブカメラからの直接入力対応
   - ライブモーション模倣

3. **複数動物の同時追跡**
   - 複数の動物を同時に検出・追跡
   - 群れの動きの模倣