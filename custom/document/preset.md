# 预设指南

本指南面向实际生产工作流，基于目标媒体的物理模态（图像 / 视频）与生产阶段（草稿打样 / 正式成片）提供基准预设配置、正交微调方案及常见画质缺陷速查。各项参数的底层物理意义、算法架构对比与数学推导详见 [调参指南](tuning.md)。

## 设计原则与硬件基座

### 最小增量覆盖原则

FaceFusion 的配置解析器采用空值回退机制：`.ini` 文件中未声明或留空、以及命令行（CLI）中未显式传入的参数，会自动采用系统内置默认值。

预设文件与本文档中的 CLI 命令均遵循最小增量覆盖原则：
- 仅显式声明偏离官方默认值的关键参数与硬件调优参数。
- 凡与官方默认值一致且无强制绑定关系的通用参数（如 `--workflow-strategy memory`、`--execution-providers cuda`、`--face-detector-size 640x640`、换脸模型默认的 `hyperswap_1a_256`、处理器默认单 `face_swapper` 等）均予以省略。这样既能凸出预设的核心调参意图，又能在上游版本升级底层模型或算子时自动受益。
- 消除死代码与冗余开销：在图像预设中彻底移除仅视频生效的编码与多线程参数；在草稿预设中显式声明 `--target-frame-amount 0`（系统默认值为 2），消除未开追踪时的冗余帧解码开销。

### 硬件基准与系统级定调

[`custom/preset/`](../preset/) 下的预设针对以下硬件平台定调系统级资源与编码参数：
- 处理器（CPU）：Intel Core Ultra 7 270K Plus（24 核 24 线程，8P + 16E）
- 系统内存（RAM）：32GB DDR5
- 图形处理器（GPU）：NVIDIA GeForce RTX 5080 16GB GDDR7（Blackwell 架构，第 5 代 Tensor Cores，第 9 代双 NVENC 硬件编码器）

硬件升级确立了系统的吞吐上限与编码基线：

| 系统级参数 | 预设取值 | 设定依据 |
| :--- | :--- | :--- |
| `--video-memory-strategy` | `tolerant` | 16GB GDDR7 显存足以将串联管线（约 3.5GB ~ 5.5GB）的模型权重与推理池常驻显存，消除任务间卸载与重载开销 |
| `--execution-thread-count` | `12` | 兼顾 24 核 CPU 调度与 16GB 显存峰值水位，流式帧读取与 GPU 并发推理达到最佳平衡（注意：该参数仅在视频管线生效，图像管线无效） |
| `--output-video-encoder` | `hevc_nvenc` | 依托第 9 代独立双 NVENC 硬件单元启用 HEVC（H.265），具备 $64 \times 64$ 编码树单元（CTU）与 SAO 环路滤波，彻底杜绝 H.264 在平坦肤色区常见的阶梯状色带；底层自动注入 `hvc1` 标签，实现全平台硬解播放 |
| `--output-video-preset` | `slow` | NVENC 底层高质量硬件流水线映射，编码过程由专用硬件 ASIC 处理，不抢占 CUDA 渲染算力 |

## 预设总览与调用方式

预设彻底摈弃了传统按镜头景别划分场景的模糊界限，重构为由**物理媒体模态**与**生产交付阶段**构成的两维极简预设体系：

| 预设文件 | 定位与核心诉求 | 适用工作流阶段 | 关键参数选型 (`detector` / `landmarker` / `swapper` / `occluder` / `processors` / `quality`) |
| :--- | :--- | :--- | :--- |
| [`image.ini`](../preset/image.ini) | **静态写真与大特写精修**<br/>（极致纹理、不计单帧耗时） | 静态封面、海报精修、高分辨率单张写真人像（`image-to-image`） | `retinaface` / `[2dfan4]` / `hyperswap_1c_256` (`1024x1024`, weight `0.6`) / `xseg_2` / `swapper + enhancer(restoreformer++)` / 图像质量 `95` |
| [`video_draft.ini`](../preset/video_draft.ini) | **视频打样与极速排雷**<br/>（秒级吞吐、极小体积） | 调参试跑、多源图排雷验证、遮罩边界检查、对口型预览 | `[yolo_face]` / `[2dfan4]` / `[hyperswap_1a_256]` (`512x512`) / `xseg_2` / 单 `swapper`（无后处理） / `hevc_nvenc` (质量 `30`, `-cq 36`) |
| [`video.ini`](../preset/video.ini) | **视频正式成片交付**<br/>（鲁棒超集、原生胶片质感） | 正式成片全片压制、包含转头甩发与遮挡的复杂动态视频 | `retinaface` (0.4) / `hrffa` / `[hyperswap_1a_256]` (`512x512`) / 追踪 (`0.25`) / `xseg_2` / 单 `swapper`（杜绝闪烁） / `hevc_nvenc` (质量 `50`, `-cq 26`) |

> 注：表中方括号 `[...]` 表示未显式覆写、直接沿用系统官方默认值。

通过 `--config-path` 参数可加载预设文件，命令行中显式传入的参数优先级高于预设文件配置：

```powershell
# 启动 Web UI 并预加载正式成片预设
uv run python facefusion.py run `
  --config-path custom/preset/video.ini

# 无头模式（Headless）加载草稿预设快速试跑打样
uv run python facefusion.py headless-run `
  --config-path custom/preset/video_draft.ini `
  -s custom/source/person_01.png custom/source/person_02.png `
  -t custom/target/video.mp4 `
  -o custom/output/video_preview.mp4

# 批量模式（Batch）加载正式视频预设压制全片
uv run python facefusion.py batch-run `
  --config-path custom/preset/video.ini `
  -s "custom/source/*.png" `
  -t "custom/target/*.mp4" `
  -o "custom/output/{target_name}.mp4"
```

## 预设详解与选型依据

### 静态写真与大特写精修

预设文件：[`custom/preset/image.ini`](../preset/image.ini)

适用于面向单反/微单拍摄的高分辨率静态人像写真。静态单张图像处理没有帧间时序抖动（Jitter）顾虑，视线聚焦于微观毛孔与眼神光，核心诉求是输出画质天花板。

```powershell
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

关键选型依据：
- **极致身份保真与原生纹理生成**：选用同系列身份注入权重最高的 `hyperswap_1c_256`，将 `--face-swapper-weight` 调至 `0.6` 强化骨相，并开启 `1024x1024` 像素提升（$4 \times 4 = 16$ 子视图交织推理）以输出最高原生分辨率。
- **重构导向超分抑塑胶感**：选用基于多头交叉注意力的 `restoreformer_plus_plus`，将 `--face-enhancer-blend` 设为 `65`，透出 35% 由 `1024x1024` 子视图生成的原生皮肤颗粒感，杜绝传统 GFP-GAN 带来的油腻塑料面具感。
- **消除裁切伪影与色温边缘**：通过 `--face-mask-padding 3 3 3 3` 内缩 3% 裁除边框伪影，配合 `--face-mask-blur 0.35` 平滑过渡带消除大特写下的接缝。
- **剔除冗余项**：彻底剥离静态图管线中无效的时序追踪、多线程、视频编码器参数；不启用针对动态视频的 `expression_restorer`。

### 视频打样与极速排雷

预设文件：[`custom/preset/video_draft.ini`](../preset/video_draft.ini)

专为长视频正式压制前的**快速试跑、选图对比、遮罩防穿模检查**而生。通过大幅精简计算流水线，吞吐率可达正式成片的 3 ~ 4 倍，几秒内即可确认成效。

```powershell
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png `
  -t custom/target/video.mp4 `
  -o custom/output/video_draft.mp4 `
  --face-selector-mode one `
  --target-frame-amount 0 `
  --face-mask-types box occlusion `
  --face-occluder-model xseg_2 `
  --face-swapper-pixel-boost 512x512 `
  --output-audio-encoder aac `
  --output-video-encoder hevc_nvenc `
  --output-video-preset slow `
  --output-video-quality 30 `
  --execution-thread-count 12 `
  --video-memory-strategy tolerant
```

关键选型依据：
- **单模块极速流水线**：`processors` 保持官方默认单 `face_swapper`，完全不挂载耗时的面部超分与后处理模型，推理吞吐最大化。
- **轻量检测与零帧分发开销**：检测（`yolo_face`）、对齐（`2dfan4`）、换脸（`hyperswap_1a_256`）全走系统内置轻量默认值；显式指定 `--target-frame-amount 0` 消除未开追踪时的冗余帧读取开销。
- **黄金甜点位尺寸**：采用 `512x512`（$2 \times 2 = 4$ 子视图），前向推理次数比 768 少 2.25 倍，且足以清晰呈现五官对齐与遮罩贴合情况。
- **极速小体积编码**：采用 `hevc_nvenc` 搭配 `--output-video-quality 30`（精准对应硬件 `-cq 36`），文件体积极小、导出极快，便于快速拖动时间轴检查。

### 视频正式成片交付

预设文件：[`custom/preset/video.ini`](../preset/video.ini)

面向工业级视频正式成品交付。核心设计理念是**“鲁棒防御超集 + 时序零抖动原生胶片质感 + 视觉无损编码”**，保证在甩头、暗光、遮挡等复杂镜头下不翻车、不闪烁。

```powershell
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png custom/source/source_2.png `
  -t custom/target/video.mp4 `
  -o custom/output/video_final.mp4 `
  --face-detector-model retinaface `
  --face-detector-score 0.4 `
  --face-landmarker-model hrffa `
  --face-selector-mode one `
  --face-tracker-score 0.25 `
  --face-mask-types box occlusion `
  --face-occluder-model xseg_2 `
  --face-swapper-pixel-boost 512x512 `
  --output-audio-encoder aac `
  --output-video-encoder hevc_nvenc `
  --output-video-preset slow `
  --output-video-quality 50 `
  --execution-thread-count 12 `
  --video-memory-strategy tolerant
```

关键选型依据：
- **鲁棒感知超集防闪回**：选用密集金字塔架构的 `retinaface`（阈值下调至 `0.4`）与极端偏航俯仰稳健的 `hrffa`，配合 `--face-tracker-score 0.25`（默认窗口 2），彻底杜绝转头甩发时因单帧漏检而闪回原脸。
- **默认 1a 姿态稳定**：换脸模型沿用内置的 `hyperswap_1a_256`，同系列中姿态与表情误差最低，抗极端大角度变形最稳。
- **坚决剔除超分，杜绝时序闪烁**：不挂载 `face_enhancer`。GAN 超分是逐帧独立推理，在视频中会不可避免地导致睫毛、胡渣、毛孔在帧间产生微小的高频蠕动与闪烁（Jitter），且容易抹平原生美感；依托 `512x512` 原生直出能够完美保留真实皮肤肌理与胶片颗粒。
- **心理视觉绝对无损**：采用 `hevc_nvenc` 搭配 `--output-video-quality 50`（精准对应硬件 `-cq 26`）。在 HEVC 编码下，`-cq 26` 是公认的视觉无损上限，既彻底消除了平坦肤色区色带，又杜绝了虚高 CQ 带来的码率暴增。

## 跨场景正交微调维度

以下微调策略与预设文件相互正交。当遇到特定特殊镜头时，直接在预设命令后追加 CLI 参数即可：

### 多人同框与目标锁定

- **单人主角锁定（`reference` 模式）**：
  在存在多个同框人物时，将 `--face-selector-mode` 切换为 `reference`，系统基于面部特征余弦相似度锁定目标。若视频首帧主角未正对镜头，传入 `--reference-frame-number <帧号>` 与 `--reference-face-position <序号>` 指定基准帧。
- **防串脸与防断联调节**：
  若配角与主角长相接近导致误换串脸，下调 `--reference-face-distance` 至 `0.20` ~ `0.25`；若主角转头幅度极大导致换脸断开，放宽至 `0.35` ~ `0.40`。
- **全员替换**：将 `--face-selector-mode` 设为 `many`。
- **注意**：切勿盲目添加 `--face-selector-gender auto`。性别分类器在暗光、侧脸、变装等边缘情况下极易单帧误判，误判瞬间会直接导致该帧被强行过滤清空，从而引发致命的闪回原脸；在跨性别换脸时更会导致全局死锁。单人视频默认 `one`，多人视频使用 `reference` 才是最稳健的方案。

### 极端表情与搞怪还原

- **按需启用表情还原**：
  若目标视频包含大笑、挑眉、密集语速口播或 Wink 微表情，且觉得默认换脸模型的面部动作略显平缓，显式追加表情还原处理器：
  ```powershell
  --processors face_swapper expression_restorer --expression-restorer-factor 85
  ```
  该模块通过 LivePortrait 形变场将目标人脸的局部生动表情强行补偿贴回。

### 远景模糊重建与超分辨率

- **远景小脸线条重建**：
  当处理练习室远景直拍或舞台群舞视频，人脸物理分辨率极低（低于 $128 \times 128$），五官严重模糊失真时，可按需追加针对低清重建的码本模型：
  ```powershell
  --processors face_swapper face_enhancer `
  --face-enhancer-model codeformer `
  --face-enhancer-blend 65 `
  --face-enhancer-weight 0.6
  ```

### 重度遮挡与眼镜保留

- **保留目标原生眼镜**：
  目标人物戴眼镜、直接换脸导致镜框被模糊抹平时，启用语义分割区域并排除 `glasses`：
  ```powershell
  --face-mask-types box occlusion region `
  --face-mask-regions skin left-eyebrow right-eyebrow left-eye right-eye nose mouth upper-lip lower-lip
  ```
- **浓密刘海与近景手势穿模**：
  将 `--face-occluder-model` 升级为 `many`（并行取 `xseg_1/2/3` 交集），并可将 `--face-mask-blur` 略微下调至 `0.25`，增强遮挡边缘锐度。

### 特殊光影与源图脸型适配

- **源图带刘海引发额头重影**：将换脸模型切换为混合解耦的 `blendswap_256`。
- **复杂舞台追光与眼神方向偏转**：将换脸模型切换为引入视线方向损失的 `ghost_3_256`。
- **骨相融合度调节**：若觉得不像源人物，调高 `--face-swapper-weight` 至 `0.60` ~ `0.70`；若五官贴合生硬，调低至 `0.30` ~ `0.40` 融入部分目标人脸特征。

## 常见画质缺陷与微调速查

| 画质缺陷或异常症状 | 底层根因诊断 | 针对性参数微调方案 |
| :--- | :--- | :--- |
| 转头、甩发瞬间偶发闪回原脸 1 ~ 2 帧 | 大侧脸或模糊导致单帧检测跌破阈值，未开启时序插值 | 确保使用 `video.ini`（已内置 `retinaface 0.4` + `hrffa` + `tracker 0.25`）；若仍有漏检，可将 `--face-detector-score` 进一步下调至 `0.35` |
| 视频播放时脸部细微噪点蠕动、睫毛微闪 | 开启了逐帧独立的 GAN 超分模型（`face_enhancer`）破坏了时序一致性 | **移除 `face_enhancer`**（`video.ini` 默认已移除），依靠 `512x512` 原生直出保持胶片质感 |
| 换脸后人物大笑或微表情显得面瘫、口型粘连 | 换脸潜在向量在注入身份时平抑了目标大幅度表情形变 | 追加 `--processors face_swapper expression_restorer`，并将 `--expression-restorer-factor` 调至 `85` ~ `90` |
| 手指、刘海、话筒或水杯被换脸纹理覆盖穿模 | 未开启遮挡遮罩，或单模型对指缝/细发丝分割不充分 | 确保设置 `--face-mask-types box occlusion`；将 `--face-occluder-model` 切换为 `xseg_2` 或三模交集 `many` |
| 眼镜框被模糊抹平或半截消失 | 换脸区域覆盖了原生眼镜，且遮挡模型未能将其识别为障碍物 | 设置 `--face-mask-types box occlusion region`，并在 `--face-mask-regions` 中剔除 `glasses` |
| 脸颊四周出现明显方框接缝或肤色断层 | 遮罩羽化宽度不足，或裁切边界存在边缘推理伪影 | 调大 `--face-mask-blur` 至 `0.35` ~ `0.45`；设置 `--face-mask-padding 3 3 3 3` 内缩裁除边界 |
| 平坦肤色区出现阶梯状色带或马赛克宏块 | 编码器未启用现代量化控制，或码率不足 | 确保设置 `--output-video-encoder hevc_nvenc`，将 `--output-video-quality` 设为 `50`（`-cq 26` 视觉无损），保持 `--output-video-preset slow` |
| 跨性别换脸时全片完全不换脸 | 误启用了 `--face-selector-gender auto`，目标与源性别不一致导致目标人脸被全局过滤 | **彻底移除 `--face-selector-gender`**，保持系统默认无性别过滤状态 |
