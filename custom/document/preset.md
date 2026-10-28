# 预设指南

本指南面向实际生产工作流，基于目标媒体的镜头景别与运动特征提供基准预设配置、跨场景正交微调方案及常见画质缺陷速查。各项参数的底层物理意义、算法架构对比与数学推导详见 [调参指南](tuning.md)。

## 设计原则与硬件基座

### 最小增量覆盖原则

FaceFusion 的配置解析器采用空值回退机制：`.ini` 文件中留空或命令行（CLI）中未显式声明的参数，会自动采用系统内置默认值。

预设文件与本文档中的 CLI 命令均遵循最小增量覆盖原则：
- 仅显式声明偏离官方默认值的场景特定增量参数与硬件调优参数。
- 凡与官方默认值一致且无强制绑定关系的通用参数（如 `--workflow-strategy memory`、`--execution-providers cuda`、`--face-detector-size 640x640`、`--expression-restorer-model live_portrait`、未开启特殊超分时的 `--face-enhancer-model gfpgan_1.4` 等）均予以省略。这样既能凸出各场景的核心调参意图，又能在上游版本升级默认模型或算子时自动受益。
- 注意一项底层例外：当未启用时序追踪（`face_tracker_score` 保持默认 `0.0`）时，系统默认的 `target_frame_amount` 仍为 `2`（每帧会额外读取前后各 2 帧）。因此在无需时序追踪的场景中，预设显式设置 `--target-frame-amount 0` 以消除冗余帧解码开销。

### 硬件基准与系统级定调

[`custom/preset/`](../preset/) 下的预设针对以下硬件平台定调系统级资源与编码参数：
- 处理器（CPU）：Intel Core Ultra 7 270K Plus（24 核 24 线程，8P + 16E）
- 系统内存（RAM）：32GB DDR5
- 图形处理器（GPU）：NVIDIA GeForce RTX 5080 16GB GDDR7（Blackwell 架构，第 5 代 Tensor Cores，第 9 代双 NVENC 编码器）

硬件升级仅改变系统吞吐上限、显存驻留策略与可承受的子视图分辨率，不盲目改变场景本身的算法选型（例如常规正脸镜头保持误检率最低的默认 `yolo_face`，无需因算力富余强行替换为 `retinaface`）：

| 系统级参数 | 预设取值 | 设定依据 |
| :--- | :--- | :--- |
| `--video-memory-strategy` | `tolerant` | 16GB GDDR7 显存足以将整条串联管线（合计约 3.5GB ~ 5.5GB）的模型权重与推理池完全常驻显存，消除任务间卸载与重载开销 |
| `--execution-thread-count` | `12` ~ `16` | 结合 24 核 CPU 与底层 [`video_manager.py`](../../facefusion/video_manager.py) 中 `skip_margin = 16` 的顺序帧读取窗口机制：常规 `512x512` 像素提升场景设为 `16`（贴合顺序流窗口上限，避免乱序跨度超过 16 触发 FFmpeg 进程重启）；`768x768` / `1024x1024` 高倍率子视图场景设为 `12` ~ `14`，兼顾 16GB 显存峰值水位与流多处理器并发率 |
| `--output-video-encoder` | `h264_nvenc` / `hevc_nvenc` | 依托独立的第 9 代双 NVENC 硬件编码单元，直接启用 `--output-video-preset slow`（映射为硬件最高质量 `-preset slow` 档位），编码过程不抢占 CUDA 推理算力 |

## 预设总览与调用方式

预设严格按照目标媒体的镜头景别与运动特征划分为 4 个正交场景，避免将“多人同框”、“遮挡物”或“调试阶段”等正交属性与媒体类型混排：

| 预设文件 | 场景类型 | 镜头景别与动态特征 | 核心模型组合 (`detector` / `landmarker` / `swapper` (`boost`) / `occluder` / `enhancer`) |
| :--- | :--- | :--- | :--- |
| [`portrait.ini`](../preset/portrait.ini) | 静态写真与特写精修 | 静态人像图片（`image-to-image`）或面部占比较大的 4K/1080p 影视慢镜头特写 | `retinaface` / `[2dfan4]` / `hyperswap_1c_256` (`1024x1024`) / `xseg_2` / `restoreformer_plus_plus` |
| [`standard.ini`](../preset/standard.ini) | 常规叙事与口播访谈 | 日常口播、Vlog、访谈对话等中近景稳定机位，姿态平缓（$\pm 30^\circ$ 内） | `[yolo_face]` / `[2dfan4]` / `hyperswap_1b_256` (`512x512`) / `xseg_2` / `[gfpgan_1.4]` |
| [`rhythmic.ini`](../preset/rhythmic.ini) | 中近景律动与变装短视频 | 半身跳舞、手势舞、卡点变装短视频，含硬切转场、高频手部/碎发掠面、Wink 微表情与原生美颜滤镜 | `scrfd` / `hrffa` / `hyperswap_1b_256` (`768x768`) / `xseg_2` / `[gfpgan_1.4]` |
| [`dynamic.ini`](../preset/dynamic.ini) | 远景全身与大幅度运动视频 | 练习室全身直拍、舞台群舞、剧烈运动镜头，含远景小脸、极端俯仰甩头、高速运动模糊与走位交错 | `retinaface` / `hrffa` / `[hyperswap_1a_256]` (`512x512`) / `xseg_3` / `codeformer` |

> 注：表中方括号 `[...]` 表示不显式覆写、直接沿用系统当前默认值。

通过 `--config-path` 参数可加载预设文件，命令行中显式传入的参数优先级高于预设文件配置：

```powershell
# 启动 Web UI 并预加载指定场景预设
uv run python facefusion.py run `
  --config-path custom/preset/rhythmic.ini

# 无头模式（Headless）加载预设并渲染单个目标素材
uv run python facefusion.py headless-run `
  --config-path custom/preset/rhythmic.ini `
  -s custom/source/person_01.png custom/source/person_02.png `
  -t custom/target/video.mp4 `
  -o custom/output/video_out.mp4

# 批量模式（Batch）加载预设并按通配符批量渲染
uv run python facefusion.py batch-run `
  --config-path custom/preset/rhythmic.ini `
  -s "custom/source/*.png" `
  -t "custom/target/*.mp4" `
  -o "custom/output/{target_name}.mp4"
```

## 场景预设详解

### 静态写真与特写精修

预设文件：[`custom/preset/portrait.ini`](../preset/portrait.ini)

适用面向单反/微单拍摄的高分辨率静态人像写真，或面部占据画面主体的影视级大特写镜头。此类场景下观众视线高度聚焦于面部微纹理，任何微小的边缘接缝、肤色断层或 GAN 超分带来的塑胶涂抹感都会被放大。

```powershell
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png custom/source/source_2.png custom/source/source_3.png `
  -t custom/target/portrait.png `
  -o custom/output/portrait.png `
  --processors face_swapper expression_restorer face_enhancer `
  --face-detector-model retinaface `
  --face-selector-mode one `
  --face-selector-gender auto `
  --target-frame-amount 0 `
  --face-mask-types box occlusion `
  --face-occluder-model xseg_2 `
  --face-mask-blur 0.35 `
  --face-mask-padding 3 3 3 3 `
  --face-swapper-model hyperswap_1c_256 `
  --face-swapper-pixel-boost 1024x1024 `
  --face-swapper-weight 0.6 `
  --face-enhancer-model restoreformer_plus_plus `
  --face-enhancer-blend 65 `
  --output-image-quality 95 `
  --output-video-encoder hevc_nvenc `
  --output-video-preset slow `
  --execution-thread-count 12 `
  --video-memory-strategy tolerant
```

关键增量参数选型依据：
- 高身份保真与微纹理生成：选用同系列身份注入权重最高的 `hyperswap_1c_256`，将 `--face-swapper-weight` 调至 `0.6` 强化源人物骨相特征，并开启 `1024x1024` 像素提升（$4 \times 4 = 16$ 子视图交织推理）以输出最高原生分辨率。
- 抑制塑胶涂抹感：选用重构导向多头交叉注意力架构的 `restoreformer_plus_plus`，并将 `--face-enhancer-blend` 降至 `65`，透出 35% 由 `1024x1024` 子视图生成的原生皮肤颗粒感。
- 无痕边缘过渡：通过 `--face-mask-padding 3 3 3 3` 内缩 3% 裁除裁切框边界伪影，配合 `--face-mask-blur 0.35` 宽过渡带消除大特写下的脸颊色温接缝。
- 母带级编码输出：视频特写采用 `hevc_nvenc` + `slow` 预设（配合默认的视频质量 `80` 即 `-cq 10` 及默认的 `flac` 无损音轨），防止平坦肤色区产生宏块色带。

场景内微调建议：
- 原片画质极佳、追求极致毛孔锐度：可将 `--face-swapper-model` 切换为强化高频皮肤微纹理判别损失的 `alphaface_256`，并将 `--face-enhancer-blend` 下调至 `45` ~ `55`（或换用原生高分辨率模型 `gpen_bfr_1024` / `gpen_bfr_2048`）。
- 单张高质量正脸源图、追求复杂光影贴合：当仅使用 1 张源图且目标画面存在复杂环境光影时，可将 `--face-swapper-model` 切换为基于空间特征位移场的 `uniface_256`。

### 常规叙事与口播访谈

预设文件：[`custom/preset/standard.ini`](../preset/standard.ini)

适用于日常口播、Vlog、教程录制、室内访谈等中近景稳定镜头。画面光照平稳，头部偏航角与俯仰角大多在 $\pm 30^\circ$ 以内，偶有说话手势、喝水或麦克风遮挡，核心诉求是在保证口型同步与五官自然的前提下实现高吞吐效率。

```powershell
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png custom/source/source_2.png custom/source/source_3.png `
  -t custom/target/talking_head.mp4 `
  -o custom/output/talking_head.mp4 `
  --processors face_swapper expression_restorer face_enhancer `
  --face-selector-mode one `
  --face-selector-gender auto `
  --target-frame-amount 0 `
  --face-mask-types box occlusion `
  --face-occluder-model xseg_2 `
  --face-swapper-model hyperswap_1b_256 `
  --face-swapper-pixel-boost 512x512 `
  --face-enhancer-blend 75 `
  --output-audio-encoder aac `
  --output-video-encoder h264_nvenc `
  --output-video-preset slow `
  --output-video-quality 70 `
  --execution-thread-count 16 `
  --video-memory-strategy tolerant
```

关键增量参数选型依据：
- 保持默认轻量高准检测：沿用系统默认的 `yolo_face` 检测器与 `2dfan4` 关键点模型，在中近景正脸场景下推理最快且背景误检率最低。
- 消除冗余帧读取：稳定机位无漏检闪烁风险，保持默认不开启追踪并将 `--target-frame-amount` 显式置 `0`，配合 `16` 线程最大化流式帧吞吐。
- 均衡换脸与手势防穿模：选用身份与姿态平衡的 `hyperswap_1b_256`（`512x512`），搭配 `xseg_2` 防范口播手势与近景话筒边缘穿模。

场景内微调建议：
- 口播语速快、唇齿开合密集：若说话时嘴型略显粘连，可将 `--expression-restorer-factor` 从默认 `80` 提高至 `85` ~ `90`；若眉眼出现轻微伴随抖动，可添加 `--expression-restorer-areas lower-face` 仅还原下半脸唇齿动作。
- 近景半身画质升级：若人物面部占比较大，可将 `--face-swapper-pixel-boost` 提升至 `768x768`（同时将 `--execution-thread-count` 调至 `12`）。

### 中近景律动与变装短视频

预设文件：[`custom/preset/rhythmic.ini`](../preset/rhythmic.ini)

适用于 B 站、抖音等平台的半身宅舞、手势舞、卡点变装及颜值律动短视频。该类素材具有四项典型特征：在音乐重拍处存在硬切（Cut）换装，前后镜头妆容、假发与灯光色温突变；包含捂脸变装、手指比心、撩发等高频近距离手部与发丝遮挡；依赖单眼眨眼（Wink）、挑眉、咬唇等微表情传递神态；原视频普遍带有磨皮美颜滤镜。

```powershell
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png custom/source/source_2.png custom/source/source_3.png `
  -t custom/target/rhythmic_video.mp4 `
  -o custom/output/rhythmic_video.mp4 `
  --processors face_swapper expression_restorer face_enhancer `
  --face-detector-model scrfd `
  --face-detector-score 0.4 `
  --face-landmarker-model hrffa `
  --face-selector-mode one `
  --face-selector-gender auto `
  --face-tracker-score 0.25 `
  --face-mask-types box occlusion `
  --face-occluder-model xseg_2 `
  --face-mask-blur 0.25 `
  --face-swapper-model hyperswap_1b_256 `
  --face-swapper-pixel-boost 768x768 `
  --face-swapper-weight 0.55 `
  --expression-restorer-factor 85 `
  --face-enhancer-blend 65 `
  --output-audio-encoder aac `
  --output-video-encoder h264_nvenc `
  --output-video-preset slow `
  --output-video-quality 75 `
  --execution-thread-count 12 `
  --video-memory-strategy tolerant
```

关键增量参数选型依据：
- 节奏型甩头检测与对齐：采用 `scrfd`（下调置信度至 `0.4`）配合大角度对齐模型 `hrffa`，确保歪头杀、低头甩发及抬头变装瞬间关键点不漂移。
- 免疫硬切变装拒识：单人短视频采用 `--face-selector-mode one` 配合 `--face-selector-gender auto` 锁定画面最大同性别面孔，规避默认 `reference` 模式下因变装前后浓妆、假发或冷暖光剧变导致特征余弦距离超限而漏换的问题。
- 窄窗口时序追踪：启用 `--face-tracker-score 0.25` 并沿用默认的 `target_frame_amount = 2` 窄窗口，既能弥合甩头瞬间的单帧模糊漏检，又可防止硬切转场帧前后发生跨镜头长窗口误插值。
- 高清子视图与防二次磨皮：利用 RTX 5080 原生 FP16 算力启用 `768x768`（$3 \times 3$ 子视图）提升原生五官清晰度，同时将 `--face-enhancer-blend` 压低至 `65`，避免在原片美颜滤镜基础上二次过度磨皮产生硅胶假面感。
- 手势防虚影：采用高频细节遮挡模型 `xseg_2`，并将 `--face-mask-blur` 略收窄至 `0.25`，防止手掌贴近脸颊时边缘产生半透明虚影。

场景内微调建议：
- 捂脸变装指缝仍有轻微穿模：将 `--face-occluder-model` 升级为 `many`（并行取 `xseg_1/2/3` 交集），并将 `--face-mask-blur` 下调至 `0.20`。
- 含躺拍、侧翻或倒立运镜：显式添加 `--face-detector-angles 0 90 270`（或按实际画面方向选取）。

### 远景全身与大幅度运动视频

预设文件：[`custom/preset/dynamic.ini`](../preset/dynamic.ini)

适用于练习室全身直拍（Dance Practice）、舞台远景群舞、街舞或剧烈运动镜头。核心难点在于全身构图下人脸像素占比极小（常低于 $128 \times 128$ 像素），且伴随大幅度甩头俯仰（极端欧拉角）、高速旋转造成的运动模糊（Motion Blur）以及多人走位交错。

```powershell
uv run python facefusion.py headless-run `
  -s custom/source/source_1.png custom/source/source_2.png custom/source/source_3.png `
  -t custom/target/stage_dance.mp4 `
  -o custom/output/stage_dance.mp4 `
  --processors face_swapper expression_restorer face_enhancer `
  --face-detector-model retinaface `
  --face-detector-score 0.35 `
  --face-landmarker-model hrffa `
  --face-selector-gender auto `
  --reference-face-distance 0.4 `
  --face-tracker-score 0.25 `
  --target-frame-amount 3 `
  --face-mask-types box occlusion `
  --face-occluder-model xseg_3 `
  --face-swapper-pixel-boost 512x512 `
  --expression-restorer-factor 75 `
  --face-enhancer-model codeformer `
  --face-enhancer-blend 75 `
  --face-enhancer-weight 0.6 `
  --output-audio-encoder aac `
  --output-video-encoder h264_nvenc `
  --output-video-preset slow `
  --output-video-quality 75 `
  --execution-thread-count 14 `
  --video-memory-strategy tolerant
```

关键增量参数选型依据：
- 远景小脸与运动模糊召回：选用密集锚框特征金字塔架构的 `retinaface`，并将 `--face-detector-score` 下调至 `0.35` 严防高速旋转丢帧；配合 `hrffa` 保障 $\pm 85^\circ$ 俯仰角与 $\pm 90^\circ$ 偏航角下的五官对齐。
- 抗姿态畸变与侧脸保护：沿用系统默认的 `hyperswap_1a_256` 换脸模型（同系列中姿态误差最低、抗大角度畸变最强），搭配针对大角度侧脸外轮廓优化的 `xseg_3` 遮挡模型，防止侧头时背光侧脸颊被误裁切。
- 宽窗口追踪防闪烁：沿用默认的 `reference` 模式锁定目标舞者，将 `--reference-face-distance` 放宽至 `0.4` 容忍大角度特征偏移，并启用 `--face-tracker-score 0.25` 与 `--target-frame-amount 3` 跨帧插值填补偶发漏检。
- 动态模糊五官重建：选用基于离散矢量量化码本查表的 `codeformer`（配合 `--face-enhancer-weight 0.6`），对远景低清与运动模糊人脸的五官线条锐化能力显著优于默认的 `gfpgan_1.4`。

场景内微调建议：
- 纯单人全身舞蹈（无伴舞、无背景落地镜反射）：可将 `--face-selector-mode` 设为 `one`，省去选定基准帧步骤并杜绝极端侧脸因超出匹配距离而漏换。
- 远景运动模糊极强、换脸后五官过锐与背景脱节：将 `--face-enhancer-weight` 调高至 `0.75` ~ `0.85`（偏向保留原貌结构）并将 `--face-enhancer-blend` 下调至 `55` ~ `60`。
- 舞者贴近画面顶部或边缘跳跃时漏检：添加 `--face-detector-margin 15 15 15 15` 为画面四周扩展补零边距。

## 跨场景正交微调维度

以下微调策略与具体视频类型正交，可根据目标素材中的实际画面元素，直接叠加至上述任一场景预设之上：

### 多人同框与目标锁定

- 单人替特定主角（`reference` 模式）：保持默认的 `--face-selector-mode reference` 与 `--face-selector-gender auto`。若视频第 `0` 帧为黑屏或主角未出场，需先通过 Web UI 预览滑块定位主角正脸清晰帧，传入 `--reference-frame-number <帧号>` 与 `--reference-face-position <序号>`。
- 防串脸与防断连权衡：若同框配角与主角同性别且长相接近导致串脸，将 `--reference-face-distance` 下调至 `0.20` ~ `0.25`，并可叠加 `--face-selector-age-start` 与 `--face-selector-age-end` 过滤年龄段；若同框人员外貌差异明显但主角频繁大幅转头导致断连，则将 `--reference-face-distance` 放宽至 `0.35` ~ `0.40`。
- 全员无差别替换：将 `--face-selector-mode` 设为 `many`。

### 重度遮挡与眼镜保留

- 保留目标人物原生眼镜：当目标人物佩戴黑框或细框眼镜、直接换脸导致镜框被抹平或半截消失时，启用语义分割遮罩并从区域列表中剔除 `glasses`：
  ```powershell
  --face-mask-types box occlusion region `
  --face-mask-regions skin left-eyebrow right-eyebrow left-eye right-eye nose mouth upper-lip lower-lip
  ```
- 厚重刘海与复杂前景障碍物：当存在浓密碎发遮眼、多人手部交错或近景话筒重度遮挡时，将 `--face-occluder-model` 设为 `many`（并行取 `xseg_1/2/3` 最小值交集），并可叠加顶部遮罩内缩 `--face-mask-padding 5 0 0 0` 消除发际线交界处的异色块。
- 暗光与逆光局部遮挡：当画面处于低照度暗光且伴随半脸阴影时，将 `--face-landmarker-model` 设为 `many`（利用 `2dfan4` 与 `peppa_wutz` 内置的暗光 CLAHE 直方图均衡增强与双模置信度互补）。

### 特殊光影与脸型骨相适配

- 源图带刘海或外脸型差异引发额头脏斑：当源图无法完全避开额前碎发，或源图与目标外轮廓差异导致换脸后额头出现暗斑、脸颊外沿重影时，将 `--face-swapper-model` 切换为基于属性解耦编码器的 `blendswap_256`（注意该模型仅读取第一张源图）。
- 跨色温复杂光影与视线飘忽：当目标视频处于霓虹彩光、舞台追光或强侧光环境，换脸后出现面部偏色或眼神方向游移时，将 `--face-swapper-model` 切换为引入自适应注意力去归一化（AAD）与视线方向损失的 `ghost_3_256`（或 `ghost_2_256`）。
- 源与目标骨相融合度调节：若换脸后觉得不像源人物，可将 `--face-swapper-weight` 调大至 `0.60` ~ `0.75`（仿射外推强化源特征）或换用引入 3DMM 三维形状先验的 `hififace_unofficial_256`；若觉得五官生硬浮在目标脸上，则将 `--face-swapper-weight` 下调至 `0.25` ~ `0.40`（融入部分目标人脸特征以平滑过渡）。

### 片段快速试跑与大批量提速

- 片段快速试跑验证：对长视频正式渲染前，无需另建预设，直接在对应场景命令后追加 `--trim-frame-start <起始帧> --trim-frame-end <结束帧>` 截取 3 ~ 5 秒核心镜头；若仅需秒级确认选图与遮罩边界，可临时指定 `--processors face_swapper` 跳过后处理。
- 遮罩与关键点异常可视化诊断：当遇到难以解释的边缘黑块、镂空或局部抖动时，在 `--processors` 末尾追加 `face_debugger`（如 `--processors face_swapper face_debugger --face-debugger-items face-landmark-5/68 face-mask`）渲染数帧，直观检视关键点与遮罩边界。
- 同参数大批量渲染提速：使用 `batch-run` 处理大批量同规格视频时，可添加 `--execution-providers tensorrt`，首次运行自动编译生成 `.caches` 硬件引擎后，后续帧推理吞吐率将获得进一步提升。

## 常见画质缺陷与微调速查

| 画质缺陷或异常症状 | 底层根因诊断 | 针对性参数微调方案 |
| :--- | :--- | :--- |
| 转头、甩发或眨眼瞬间偶发闪回原脸 1 ~ 2 帧 | 运动模糊或大侧脸导致单帧检测置信度跌破阈值，或 `reference` 模式下特征距离超标，且未开启时序插值 | 开启 `--face-tracker-score 0.25`（配合 `--target-frame-amount 2` 或 `3`）；下调 `--face-detector-score` 至 `0.35` ~ `0.40` 并使用 `retinaface` 或 `scrfd`；单人视频改用 `--face-selector-mode one` 或将 `--reference-face-distance` 放宽至 `0.35` ~ `0.45` |
| 卡点变装或切镜头后整段不换脸 | 变装前后妆容、假发或打光剧变，导致转场后人脸与初始基准帧的特征余弦距离超出阈值 | 单人视频直接将 `--face-selector-mode` 设为 `one`（配合 `--face-selector-gender auto`）；多人视频将 `--reference-face-distance` 放宽至 `0.40` ~ `0.45` |
| 硬切转场帧瞬间人脸飞到背景上产生残影 | `--target-frame-amount` 窗口过宽，在镜头硬切处将前后两个不同镜头的检测框强行做了跨帧线性插值 | 将 `--target-frame-amount` 减小至 `1` 或 `2`，或将 `--face-tracker-score` 提高至 `0.35` ~ `0.40`（仅允许高重合度相邻帧插值） |
| 手指、刘海、话筒或水杯被换脸纹理覆盖穿模 | 未启用 `occlusion` 遮罩，或默认 `xseg_1` 对细碎发丝与指缝分割偏保守，或羽化带宽溢出 | 确保设置 `--face-mask-types box occlusion`；将 `--face-occluder-model` 切换为 `xseg_2` 或三模交集 `many`；将 `--face-mask-blur` 适度下调至 `0.20` ~ `0.25` |
| 眼镜框被模糊抹平或半截消失 | 换脸区域覆盖了原生眼镜，且 `xseg` 未能将细镜框完整识别为遮挡物 | 设置 `--face-mask-types box occlusion region`，并在 `--face-mask-regions` 中剔除 `glasses` |
| 脸部四周出现明显方框接缝或肤色断层 | 遮罩羽化模糊度过小、裁切框边界存在推理伪影，或源图带有强侧光阴影污染了肤色基准 | 调大 `--face-mask-blur` 至 `0.35` ~ `0.45`；设置 `--face-mask-padding 3 3 3 3` 内缩裁除边界；剔除带强阴影的源图，或换用 `uniface_256` / `blendswap_256` |
| 额头出现莫名脏斑或外脸型边缘重影 | 源图自身带有刘海遮挡额头，或源图与目标外脸型差异过大导致特征泄漏 | 更换无刘海遮挡的标准正脸作为第一张源图；将 `--face-swapper-model` 切换为 `blendswap_256`；或设置顶部内缩 `--face-mask-padding 8 0 0 0` |
| 大角度侧头、仰头或低头时五官扭曲飞脱 | 默认 `2dfan4` 在极端俯仰角/偏航角下关键点热力图漂移，导致仿射对齐矩阵畸变 | 将 `--face-landmarker-model` 切换为 `hrffa`（保持默认 `--face-landmarker-score 0.5`，切勿设为 `1.0`）；换脸模型采用默认的 `hyperswap_1a_256`，遮挡模型选用 `xseg_3` |
| 眼神呆滞、Wink 眨眼失灵或大笑神态僵硬 | 换脸网络在注入源身份特征时压平了目标帧的局部微表情形变与视线方向 | 在 `--processors` 中紧接 `face_swapper` 启用 `expression_restorer`，将 `--expression-restorer-factor` 调至 `85` ~ `90`；若视线仍飘忽则换用 `ghost_3_256` |
| 皮肤呈硅胶塑料感或涂抹感过重 | 原视频已自带美颜滤镜，叠加高混合度 `gfpgan` 造成二次过度磨皮，或 `pixel_boost` 过低 | 将 `--face-enhancer-blend` 下调至 `55` ~ `65`；将 `--face-swapper-pixel-boost` 提升至 `768x768` 或 `1024x1024`；或将 `--face-enhancer-model` 切换为 `restoreformer_plus_plus` |
| 高动态画面背景或脸颊出现马赛克压缩色块 | 视频编码器量化参数过高，高运动画面码率不足 | 将 `--output-video-quality` 提高至 `75` ~ `85`，保持 `--output-video-preset slow`，或将编码器切换为 `hevc_nvenc` |
