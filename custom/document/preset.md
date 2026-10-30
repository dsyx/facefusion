# 预设指南

本指南面向实际生产工作流，基于媒体模态（图像 / 视频）与生产阶段（草稿打样 / 正式成片）提供基准预设配置、正交微调方案与常见画质缺陷速查。参数物理意义与算法架构推导详见 [调参指南](tuning.md)。

## 设计原则与硬件基座

### 最小增量覆盖原则

FaceFusion 配置解析器采用空值回退机制：`.ini` 文件或命令行中未显式声明的参数自动回退至系统默认值。

预设文件与命令行示例遵循最小增量覆盖原则：
- 仅声明偏离系统默认值的参数与关键模型选型。
- 图像预设中移除视频编码、时序追踪与多线程参数。
- 草稿预设中显式声明 `--target-frame-amount 0`（系统默认值为 2），消除未启用追踪时的多余帧解码开销。

### 硬件基准与系统级定调

预设基于以下硬件平台定调资源与编码参数：
- CPU：Intel Core Ultra 7 270K Plus（24 核 24 线程）
- RAM：32GB DDR5
- GPU：NVIDIA GeForce RTX 5080 16GB GDDR7（Blackwell 架构，第 9 代双 NVENC 编码器）

系统级参数取值与依据：

| 参数 | 取值 | 依据 |
| :--- | :--- | :--- |
| `--video-memory-strategy` | `tolerant` | 16GB 显存可常驻全管线模型权重与推理池，消除任务间卸载开销 |
| `--execution-thread-count` | `12` | 兼顾 24 核 CPU 调度与 16GB 显存峰值，仅在视频管线生效 |
| `--output-video-encoder` | `hevc_nvenc` | 基于双 NVENC 硬件单元，利用 64x64 CTU 与 SAO 环路滤波消除平坦肤色区色带，自动注入 hvc1 标签保障兼容性 |

## 预设总览与调用方式

预设按媒体模态与交付阶段划分为三个基础预设：

| 预设文件 | 定位与诉求 | 适用阶段 | 核心参数 (`detector` / `landmarker` / `swapper` / `occluder` / `processors` / `quality`) |
| :--- | :--- | :--- | :--- |
| [`image.ini`](../preset/image.ini) | 静态写真精修（画质优先） | 静态封面、海报、单张写真精修 | `retinaface` / `[2dfan4]` / `hyperswap_1c_256` (`1024x1024`, weight `0.6`) / `xseg_2` / `swapper + enhancer(restoreformer++)` / 图像质量 `95` |
| [`video_draft.ini`](../preset/video_draft.ini) | 视频打样排雷（吞吐优先） | 试跑调参、选图对比、遮罩边界检查 | `[yolo_face]` / `[2dfan4]` / `hyperswap_1a_256` (`512x512`) / `xseg_2` / 单 `swapper` / `hevc_nvenc` (`veryfast`, 质量 `35`, `-cq 33`) |
| [`video.ini`](../preset/video.ini) | 视频正式成片（鲁棒保真） | 正式成片全片压制、复杂动态镜头 | `retinaface` (0.4) / `hrffa` / `hyperswap_1a_256` (`512x512`) / 追踪 (`0.25`) / `xseg_2` / 单 `swapper` / `hevc_nvenc` (`slow`, 质量 `55`, `-cq 23`) |

> 注：方括号 `[...]` 表示沿用系统默认值。

预设调用示例：

```powershell
# Run Web UI with video preset
uv run python facefusion.py run `
  --config-path custom/preset/video.ini

# Run headless with draft preset
uv run python facefusion.py headless-run `
  --config-path custom/preset/video_draft.ini `
  -s custom/source/source_1.png `
  -t custom/target/video.mp4 `
  -o custom/output/video.mp4

# Run batch with video preset
uv run python facefusion.py batch-run `
  --config-path custom/preset/video.ini `
  -s "custom/source/*.png" `
  -t "custom/target/*.mp4" `
  -o "custom/output/{target_name}.{target_extension}"
```

## 预设详解与选型依据

### 静态写真精修

预设文件：[`custom/preset/image.ini`](../preset/image.ini)

面向高分辨率单张人像写真。静态图无帧间时序抖动顾虑，注重微观毛孔纹理与五官还原：

```powershell
# Run headless image swap
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png custom/source/source_2.png `
  -t custom/target/portrait.png `
  -o custom/output/portrait.png `
  --face-detector-model retinaface `
  --face-selector-mode one `
  --face-mask-types box occlusion `
  --face-occluder-model xseg_2 `
  --face-mask-blur 0.35 `
  --face-mask-padding 3 3 3 3 `
  --output-image-quality 95 `
  --processors face_swapper face_enhancer `
  --face-enhancer-model restoreformer_plus_plus `
  --face-enhancer-blend 65 `
  --face-swapper-model hyperswap_1c_256 `
  --face-swapper-pixel-boost 1024x1024 `
  --face-swapper-weight 0.6 `
  --video-memory-strategy tolerant
```

选型依据：
- 身份保真与纹理：选用同系列身份注入权重最高的 `hyperswap_1c_256`（权重 0.6 强化骨相），开启 `1024x1024` 像素提升（16 个子视图交织推理）提供原生高分辨率。
- 超分融合：采用基于多头交叉注意力的 `restoreformer_plus_plus`，混合度设为 65 透出 35% 像素提升生成的原生颗粒感，避免塑料涂抹假面。
- 边缘裁切：内缩 3% 消除裁切边界伪影，搭配 0.35 羽化过渡带消除色温接缝。
- 移除死代码：剥离时序追踪、多线程、视频编码器参数；不启用针对动态视频的表情还原。

### 视频打样排雷

预设文件：[`custom/preset/video_draft.ini`](../preset/video_draft.ini)

面向正式渲染前的快速试跑、选图对比与遮罩检查。大幅压缩计算流水线，吞吐率达正式成片的 3 至 4 倍：

```powershell
# Run headless draft video
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png `
  -t custom/target/video.mp4 `
  -o custom/output/video.mp4 `
  --face-selector-mode one `
  --target-frame-amount 0 `
  --face-mask-types box occlusion `
  --face-occluder-model xseg_2 `
  --face-swapper-model hyperswap_1a_256 `
  --face-swapper-pixel-boost 512x512 `
  --output-audio-encoder aac `
  --output-video-encoder hevc_nvenc `
  --output-video-preset veryfast `
  --output-video-quality 35 `
  --execution-thread-count 12 `
  --video-memory-strategy tolerant
```

选型依据：
- 单换脸管线：仅运行单模块 `face_swapper`，跳过面部超分与后处理模型，最大化推理吞吐。
- 轻量检测与零帧分发：采用默认检测器 `yolo_face` 与关键点 `2dfan4`；显式设 `--target-frame-amount 0` 消除相邻帧解码开销。
- 尺寸甜点位：选用 `512x512`（4 个子视图），推理量仅为 768 的 44%，足以清晰检视五官贴合度与遮罩边界。
- 高速轻量编码：采用 `hevc_nvenc` 搭配 `veryfast`（映射为硬件 `-preset fast` 高速流水线）与质量 35（对应 `-cq 33`，VMAF 70~81），文件体积极小且导出迅速。

### 视频正式成片

预设文件：[`custom/preset/video.ini`](../preset/video.ini)

面向工业级视频正式交付，兼顾大角度姿态抗崩与时序原生胶片质感：

```powershell
# Run headless production video
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png custom/source/source_2.png `
  -t custom/target/video.mp4 `
  -o custom/output/video.mp4 `
  --face-detector-model retinaface `
  --face-detector-score 0.4 `
  --face-landmarker-model hrffa `
  --face-selector-mode one `
  --face-tracker-score 0.25 `
  --face-mask-types box occlusion `
  --face-occluder-model xseg_2 `
  --face-swapper-model hyperswap_1a_256 `
  --face-swapper-pixel-boost 512x512 `
  --output-audio-encoder aac `
  --output-video-encoder hevc_nvenc `
  --output-video-preset slow `
  --output-video-quality 55 `
  --execution-thread-count 12 `
  --video-memory-strategy tolerant
```

选型依据：
- 鲁棒感知超集：采用密集金字塔架构的 `retinaface`（阈值下调至 0.4）与抗极端欧拉角的 `hrffa`，配合时序追踪（阈值 0.25，窗口 2），杜绝甩头转场瞬间因单帧漏检闪回原脸。
- 姿态稳定性：选用 `hyperswap_1a_256`，在同系列中姿态与表情误差最低，抗大角度畸变能力最强。
- 杜绝时序抖动：不启用 `face_enhancer`。GAN 超分逐帧独立推理会导致高频细节（睫毛、胡渣、毛孔）帧间闪烁蠕动；依托 `512x512` 直出保留真实胶片质感。
- 成片编码甜点位：采用 `hevc_nvenc` 搭配 `slow`（高质量硬件流水线）与质量 55（对应 `-cq 23`，VMAF 89~93），体积较视觉无损锐减约 35%，彻底消除平坦肤色区色带。

## 跨场景正交微调维度

以下策略与预设文件正交，针对特定镜头按需在预设命令后追加 CLI 参数：

### 多人同框与目标锁定

- 单人主角锁定（`reference` 模式）：同框有多人时切换为 `reference`，系统基于人脸特征余弦相似度匹配目标。若首帧非主角正脸，传入 `--reference-frame-number <帧号>` 与 `--reference-face-position <序号>`。
- 距离阈值微调：若配角与主角接近导致串脸，调小 `--reference-face-distance` 至 0.20 ~ 0.25；若主角大角度转头断连，放宽至 0.35 ~ 0.40。
- 全员替换：设 `--face-selector-mode many`。
- 性别过滤风险：避免使用 `--face-selector-gender auto`。性别分类器在暗光、侧脸或变装时易单帧误判，误判会导致该帧人脸被完全过滤从而闪回原脸；跨性别换脸时更会导致全局死锁。

### 微表情补偿与超分辨率

- 夸张表情补偿：若原视频含大笑、夸张微表情且默认换脸显得平缓，追加表情还原处理器：
  ```powershell
  --processors face_swapper expression_restorer --expression-restorer-factor 85
  ```
- 远景小脸超分：处理远景全身镜头且人脸物理分辨率极低（低于 128x128）时，按需追加离散码本增强模型：
  ```powershell
  --processors face_swapper face_enhancer `
    --face-enhancer-model codeformer `
    --face-enhancer-blend 65 `
    --face-enhancer-weight 0.6
  ```

### 遮挡与特殊光影适配

- 保留原生眼镜：启用语义分割遮罩并剔除 `glasses`：
  ```powershell
  --face-mask-types box occlusion region `
    --face-mask-regions skin left-eyebrow right-eyebrow left-eye right-eye nose mouth upper-lip lower-lip
  ```
- 碎发与手势穿模：遮挡模型切换为三模交集 `--face-occluder-model many`，羽化带宽收窄至 `--face-mask-blur 0.25`。
- 源图刘海导致额头重影：换脸模型切换为属性解耦的 `--face-swapper-model blendswap_256`。
- 复杂舞台光与视线偏转：换脸模型切换为带视线损失的 `--face-swapper-model ghost_3_256`。
- 骨相融合度调节：若不像源人物调大 `--face-swapper-weight` 至 0.60 ~ 0.70；若贴合生硬调小至 0.30 ~ 0.40。

## 常见画质缺陷速查

| 缺陷症状 | 根因诊断 | 针对性微调方案 |
| :--- | :--- | :--- |
| 转头甩发瞬间闪回原脸 | 大侧脸单帧检测跌破阈值且未开时序插值 | 确保使用 `video.ini`（内置 `retinaface 0.4` + `hrffa` + `tracker 0.25`）；极度大侧脸可下调 `--face-detector-score` 至 0.35 |
| 播放时脸部细微噪点蠕动闪烁 | 启用了逐帧独立的 GAN 超分模型破坏时序一致性 | 移除 `face_enhancer`（`video.ini` 默认无超分），依靠 `512x512` 原生直出 |
| 大笑或微表情显得僵硬粘连 | 换脸特征向量平抑了原视频大幅度表情形变 | 追加 `--processors face_swapper expression_restorer`，设置 `--expression-restorer-factor 85` |
| 手指、刘海、话筒被换脸覆盖穿模 | 未开启遮挡遮罩或单模型对指缝发丝分割不充分 | 设置 `--face-mask-types box occlusion`，切换 `--face-occluder-model xseg_2` 或 `many` |
| 眼镜框被模糊抹平或半截消失 | 换脸区域覆盖了原生眼镜 | 设置 `--face-mask-types box occlusion region`，在 `--face-mask-regions` 中剔除 `glasses` |
| 脸颊四周出现明显接缝或色温断层 | 羽化宽度不足或裁切边缘存在伪影 | 调大 `--face-mask-blur` 至 0.35 ~ 0.45，设置 `--face-mask-padding 3 3 3 3` |
| 平坦肤色区出现阶梯状色带 | 编码器未启用现代量化控制或码率不足 | 设置 `--output-video-encoder hevc_nvenc`，`--output-video-quality 55`，`--output-video-preset slow` |
| 跨性别换脸全片不换脸 | 启用了 `--face-selector-gender auto` 导致目标人脸被过滤 | 移除 `--face-selector-gender`，保持默认无过滤 |
