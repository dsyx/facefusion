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

| 预设文件 | 定位与诉求 | 适用阶段 | 核心参数 (`detector` / `landmarker` / `swapper` / `mask` / `processors` / `quality`) |
| :--- | :--- | :--- | :--- |
| [`image.ini`](../preset/image.ini) | 静态写真精修（画质优先） | 静态封面、海报、单张写真精修 | `retinaface` (`[0.5]`) / `[2dfan4]` / `hyperswap_1c_256` (`1024x1024`) / `[box]` / `[单 swapper]` / `[无损 PNG]` |
| [`video_draft.ini`](../preset/video_draft.ini) | 视频打样排雷（吞吐优先） | 试跑调参、选图对比、遮罩边界检查 | `[yolo_face]` (`[0.5]`) / `[2dfan4]` / `hyperswap_1a_256` (`512x512`) / `box occlusion` (`[xseg_1]`) / `[单 swapper]` / `hevc_nvenc` (`[veryfast]`, 质量 `35`, `-cq 33`) |
| [`video.ini`](../preset/video.ini) | 视频正式成片（鲁棒保真） | 正式成片全片压制、复杂动态镜头 | `retinaface` (`[0.5]`) / `[2dfan4]` / `hyperswap_1a_256` (`512x512`) / 追踪 (`0.25`) / `box occlusion` (`[xseg_1]`) / `[单 swapper]` / `hevc_nvenc` (`slow`, 质量 `55`, `-cq 23`) |

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

# Run batch with video preset (Cartesian product)
uv run python facefusion.py batch-run `
  --config-path custom/preset/video.ini `
  -s "custom/source/*.png" `
  -t "custom/target/*.mp4" `
  -o "custom/output/{target_name}_{source_name}.{target_extension}"

# Run multi-source fusion batch with video preset
Get-ChildItem "custom/target/*.mp4" | ForEach-Object {
  uv run python facefusion.py headless-run `
    --config-path custom/preset/video.ini `
    -s (Get-ChildItem "custom/source/source_*.png").FullName `
    -t $_.FullName `
    -o "custom/output/$($_.Name)"
}
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
  --face-swapper-model hyperswap_1c_256 `
  --face-swapper-pixel-boost 1024x1024 `
  --video-memory-strategy tolerant
```

选型依据：
- 身份保真与原生高清：选用同系列身份注入权重最高的 `hyperswap_1c_256`（内部权重 `0.6` 通过轻度仿射外推适度强化源人物骨相辨识度），开启 `1024x1024` 像素提升（16 个子视图交织推理）直接提供原生超高分辨率，完全摆脱对后置面部超分模型（`face_enhancer`）的依赖，杜绝塑料假面感与生成伪影。
- 单 `box` 遮罩杜绝下巴白边：沿用系统默认的 `box` 矩形羽化遮罩而不默认开启 `occlusion`（未启用 `occlusion` 时 `face_occluder_model` 不参与推理）。原因在于 `hyperswap_1c_256` 会向源人物骨相收敛，当源人脸下巴短于目标人物的尖下巴时，模型会在新下巴下方合成颈部暗影；若开启 `occlusion`（XSeg 依据目标原图较长的下巴边缘做 $\sigma=5$ 的窄带截断），目标原图底端未被遮盖的亮色下巴尖会在新下巴暗影外侧露出一圈模糊白边轮廓。仅使用 `box` 遮罩（在 `1024x1024` 下拥有 $\sigma=38$ 的宽幅高斯羽化）可将下巴及颈部阴影一并平滑覆盖，同时 `hyperswap` 自身具备良好的额前刘海保留能力。
- 移除死代码与冗余配置：剥离无效的 `face_occluder_model`、默认 `processors`、时序追踪、多线程与视频编码器参数；输出 PNG 格式采用无损压缩，无需声明有损图像压缩参数。

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
  --face-swapper-model hyperswap_1a_256 `
  --face-swapper-pixel-boost 512x512 `
  --output-audio-encoder aac `
  --output-video-encoder hevc_nvenc `
  --output-video-quality 35 `
  --execution-thread-count 12 `
  --video-memory-strategy tolerant
```

选型依据：
- 单换脸管线：仅运行默认单模块 `face_swapper`，跳过面部超分与后处理模型，最大化推理吞吐。
- 轻量检测与零帧分发：采用默认检测器 `yolo_face` 与关键点 `2dfan4`；显式设 `--target-frame-amount 0` 消除相邻帧解码开销。
- 稳健遮挡边界：开启 `box occlusion` 并沿用系统默认的 `xseg_1` 遮挡模型，既能防止动态侧脸时换脸区域溢出至鬓角头发或背景，又能避免 `xseg_2` 在鬓角后方产生孤立遮罩白斑及下巴区域的半透明灰斑。
- 尺寸甜点位：选用 `512x512`（4 个子视图），推理量仅为 768 的 44%，足以清晰检视五官贴合度与遮罩边界。
- 高速轻量编码：采用 `hevc_nvenc` 搭配系统默认的 `veryfast` 预设（映射为硬件 `-preset fast` 高速流水线）与质量 35（对应 `-cq 33`，VMAF 70~81），文件体积极小且导出迅速。

### 视频正式成片

预设文件：[`custom/preset/video.ini`](../preset/video.ini)

面向工业级视频正式交付，兼顾大幅度转身防穿帮与时序原生胶片质感：

```powershell
# Run headless production video
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png custom/source/source_2.png `
  -t custom/target/video.mp4 `
  -o custom/output/video.mp4 `
  --face-detector-model retinaface `
  --face-selector-mode one `
  --face-tracker-score 0.25 `
  --face-mask-types box occlusion `
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
- 转头防穿帮感知组合（`retinaface 0.5` + `2dfan4` + `xseg_1` + `tracker 0.25`）：
  - 检测阈值保持默认 `0.5`：在舞蹈 $360^\circ$ 转身过程中，可见侧脸帧的 `retinaface` 置信度通常 $\ge 0.52$，而转至 $>95^\circ$ 背身/后脑勺时耳部与颈部会产生 `0.40` ~ `0.49` 的低分误检框；保持 `0.5` 阈值可精准拦截背身误检，同时交由时序追踪（阈值 `0.25`，默认窗口 `2`）平滑插值偶发单帧漏检。
  - 关键点保持默认 `2dfan4`（而非 `hrffa`）：`2dfan4` 的热力图紧贴可见 2D 面部轮廓，且在背身 $>90^\circ$ 时热力峰值置信度自动跌破 `0.5` 从而安全回退至检测器 5 点；反之，`hrffa` 在大侧脸转头时其 $1.7\times$ 扩框全头 3D 预测会将不可见的远端眼角/嘴角投影至可见侧脸鬓角上（导致收敛后的 5 点对齐模板向头发偏移 20~45 像素，把眼睛画在头发上），且在背身帧会将耳朵误回归为五官，加之其硬编码置信度恒为 `1.0` 无法触发阈值回退，极易在转头瞬间穿帮。
  - 遮挡模型保持默认 `xseg_1`：在侧脸转头鬓角垂落遮挡脸颊时，`xseg_1` 仅保留前侧鼻唇轮廓区域，避免 `xseg_2` 在鬓角发丝后方分割出孤立白斑导致侧脸重影穿模。
- 姿态与下颌贴合度：选用 `hyperswap_1a_256`，在同系列中姿态与表情误差最低，且比 `1c` 更忠实保留目标人物原有的下颌长度，大幅减轻开启 `occlusion` 时的下巴轮廓白边。
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

- 夸张表情补偿：若原视频含大笑、夸张微表情且默认换脸显得平缓，追加表情还原处理器（保持系统默认 factor 80，对应 0.96 阻尼安全区，避免 factor > 83.3 导致超调撕裂）：
  ```powershell
  --processors face_swapper expression_restorer
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
- 正面细碎遮挡穿模：当正面镜头存在手指托腮、近景话筒或复杂前额碎发穿模时，将遮挡模型切换为高频细节模型 `--face-occluder-model xseg_2` 或三模交集 `--face-occluder-model many`（注：`--face-mask-blur` 仅控制 `box` 矩形遮罩边缘羽化，`occlusion` 遮罩固定使用 $\sigma=5$ 高斯平滑）。
- 下巴模糊白边轮廓抑制：当启用 `occlusion` 后因源人物下巴短于目标人物而在下巴底端露出目标原脸亮色白边时，调小 `--face-swapper-weight` 至 `0.35` ~ `0.40`（或在静态图中关闭 `occlusion` 仅用 `--face-mask-types box`），使生成下颌贴合目标原有轮廓长度。
- 卧姿或倒立大翻滚角镜头：仅当人物面部持续朝向镜头但发生大幅度平面内旋转（Roll）时，按需追加 `--face-landmarker-model hrffa` 或开启多角度检测 `--face-detector-angles 0 90 180 270`（切勿在含 $360^\circ$ 背身转头的舞蹈视频中启用 `hrffa`）。
- 源图刘海导致额头重影：换脸模型切换为属性解耦的 `--face-swapper-model blendswap_256`。
- 复杂舞台光与视线偏转：换脸模型切换为带视线损失的 `--face-swapper-model ghost_3_256`。
- 骨相融合度调节：若不像源人物调大 `--face-swapper-weight` 至 `0.55` ~ `0.65`（适度仿射外推强化）；若贴合生硬或下巴轮廓不匹配调小至 `0.30` ~ `0.40`（融入目标脸型）。

## 常见画质缺陷速查

| 缺陷症状 | 根因诊断 | 针对性微调方案 |
| :--- | :--- | :--- |
| 人物转脸或背身旋转时五官贴在头发/耳朵上穿帮 | 1. `hrffa` 全头 3D 回归将远端不可见点投影至鬓角且硬编码置信度 `1.0` 无法回退；<br/>2. 检测阈值过低（如 `0.4`）在背身时误检耳部；<br/>3. `xseg_2` 在鬓角发丝后方分割出孤立白斑 | 使用更新后的 `video.ini`（默认 `2dfan4` + `retinaface 0.5` + `xseg_1`）；若极端侧脸仍有残留，可将 `--face-detector-score` 微调至 `0.55` |
| 下巴底端出现模糊白边轮廓或双下巴重影 | 源人脸下巴短于目标尖下巴，且开启 `occlusion` / `region` 后遮罩沿目标原下巴边缘窄带（$\sigma=5$）截断，使目标原下巴尖亮色边缘露在新下巴阴影外侧 | 静态图使用 `--face-mask-types box`（依靠宽高斯羽化覆盖下巴过渡区）；必须开启 `occlusion` 的视频或遮挡图将 `--face-swapper-weight` 下调至 `0.35` ~ `0.40`（或由 `1c` 换用 `hyperswap_1a_256`） |
| 转头甩发瞬间闪回原脸 | 侧脸单帧检测跌破阈值且未开启时序插值 | 确保使用 `video.ini`（内置 `retinaface` + `tracker 0.25`），并将 `--target-frame-amount` 调大至 `3` ~ `4` 拓宽插值窗口（避免盲目下调检测阈值引发背身误检） |
| 播放时脸部细微噪点蠕动闪烁 | 启用了逐帧独立的 GAN 超分模型破坏时序一致性 | 移除 `face_enhancer`（`video.ini` 默认无超分），依靠 `512x512` 原生直出 |
| 大笑或微表情显得僵硬粘连 | 换脸特征向量平抑了原视频大幅度表情形变 | 追加 `--processors face_swapper expression_restorer`（默认 factor 80 提供 0.96 倍高保真形变复原，避免超调撕裂） |
| 正面手指、话筒或细碎刘海被换脸覆盖穿模 | 未开启遮挡遮罩或基础模型 `xseg_1` 对指缝细节分割不够锐利 | 设置 `--face-mask-types box occlusion`，切换 `--face-occluder-model xseg_2` 或 `many` |
| 眼镜框被模糊抹平或半截消失 | 换脸区域覆盖了原生眼镜 | 设置 `--face-mask-types box occlusion region`，在 `--face-mask-regions` 中剔除 `glasses` |
| 脸颊四周出现明显矩形接缝或色温断层 | `box` 矩形遮罩羽化宽度不足 | 调大 `--face-mask-blur` 至 `0.35` ~ `0.45` |
| 平坦肤色区出现阶梯状色带 | 编码器未启用现代量化控制或码率不足 | 设置 `--output-video-encoder hevc_nvenc`，`--output-video-quality 55`，`--output-video-preset slow` |
| 跨性别换脸全片不换脸 | 启用了 `--face-selector-gender auto` 导致目标人脸被过滤 | 移除 `--face-selector-gender`，保持默认无过滤 |
