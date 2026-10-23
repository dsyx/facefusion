# 调参指南

本指南旨在深入阐述 FaceFusion 的底层工作原理、核心参数物理意义、数学推导公式及系统级调参策略。

## 工作原理与处理管线

FaceFusion 采用解耦的流式帧处理管线。在逐帧处理流程中，系统调度目标帧，并将其流转至各已启用的处理器（Processor）执行串行推理：

```mermaid
flowchart LR
    A["素材输入"] --> B["帧提取与调度"]
    B --> C["人脸检测与跟踪"]
    C --> D["串行处理器链"]
    D --> E["多媒体编码封装"]
```

在串行处理器链中，每个局部人脸处理器均保持自包含的执行闭环：

```mermaid
flowchart LR
    F1["当前目标帧"] --> F2["仿射对齐裁剪"]
    F2 --> F3["神经网络推理"]
    F3 --> F4["遮罩生成与约束"]
    F4 --> F5["逆仿射贴回融合"]
    F5 --> F6["输出递推帧"]
```

串行管线中的下一个处理器将直接以前一个处理器的输出帧作为输入，依次完成几何替换、微表情修正与超分辨率高清重建。

## 源素材规范与融合机制

源素材（Source）的质量直接决定特征向量提取的上限；若源图存在几何畸变或光影缺陷，下游处理器的补偿能力将受到显著制约。

### 源图片选取规范

人脸图像应满足以下条件：
- 分辨率与清晰度：推荐人脸局部区域在 $512 \times 512$ 像素以上，五官对焦清晰，无运动模糊（Motion Blur）或过度压缩噪点。
- 角度与姿态：正脸或 15 度以内的微侧脸最佳。避免俯仰角过大（仰头或低头）的照片，否则特征点对齐易发生几何畸变。
- 光照与阴影：光照均匀、两侧明暗对称的照片最佳。强侧光、明暗对比过强（如半边阴影）或逆光照片会将强光阴影特征作为身份特征固化到特征向量中，导致换脸后目标人脸发黑或出现脏斑。
- 表情自然度：优先选用表情自然平稳、嘴唇自然闭合、无明显露齿大笑或夸张表情的照片。换脸模型会将源人脸表情中的五官形变部分代入身份特征向量中，表情扭曲的源图会导致生成画面神态僵硬。
- 遮挡剔除：避免刘海遮眼、佩戴黑框眼镜、手托下巴等遮挡五官的照片。

### 多源图融合机制

FaceFusion 支持传入多张源人脸图片。系统提取每张源人脸的原始高维特征向量（Embedding）$\vec{e}_i$ 及 $L_2$ 范数归一化单位特征向量 $\hat{e}_i = \vec{e}_i / \|\vec{e}_i\|_2$，并分别独立计算两者的算术均值：

$$\vec{e}_{\text{avg}} = \frac{1}{N} \sum_{i=1}^N \vec{e}_i, \quad \hat{e}_{\text{avg}} = \frac{1}{N} \sum_{i=1}^N \frac{\vec{e}_i}{\|\vec{e}_i\|_2}$$

下游换脸处理器根据所选模型的特征空间定义，直接采用 $\hat{e}_{\text{avg}}$（如 `hyperswap`），或基于 $\vec{e}_{\text{avg}}$ 经线性投影、跨空间特征转换后生成最终的身份特征向量。

> 参考：
> - [`facefusion/face_recognizer.py:calculate_face_embedding()`](../../facefusion/face_recognizer.py)
> - [`facefusion/face_creator.py:average_face_identity()`](../../facefusion/face_creator.py)
> - [`facefusion/processors/modules/face_swapper/core.py:prepare_source_embedding()`](../../facefusion/processors/modules/face_swapper/core.py)

首图几何基准与参考角度控制：
- 首图几何与属性继承：多图融合仅对身份特征向量求取均值，而合并后源人脸的边界框（Bounding Box）、五官关键点（Landmarks）、头部姿态角以及性别、年龄、人种属性均直接继承自第一张源图；同时，依赖源图外观纹理裁剪输入的换脸模型（如 `blendswap`、`uniface`）也仅读取第一张源图的画面。因此，第一张源图必须为正视镜头（俯仰角 Pitch、偏航角 Yaw、翻滚角 Roll 偏角均 $< 10^\circ$）的高清无遮挡标准正脸。
- 辅助源图角度容差：后续辅助源图仅提取特征向量参与算术均值计算，用于丰富特征表征并抵消单张照片的局部阴影与光斑噪点。辅助图的水平偏角（Yaw）建议控制在 $\pm 15^\circ$ 以内，俯仰角（Pitch）控制在 $\pm 10^\circ$ 以内。
- 避免大角度侧脸：严禁混入偏航角 $> 30^\circ$ 的大角度侧脸照片。侧脸会因背光侧五官被遮挡而丢失对称特征向量，均值混合后会严重拉偏正脸特征基准，导致生成的人脸五官扁平、眼距畸变或边缘模糊。

源图数量建议与上限：
- 推荐数量（3 ~ 5 张）：选取 3 至 5 张光照均匀、微表情自然的正面照片即可在表征丰富度与计算效率间取得最佳平衡。特征向量均值能够有效中和局部反光与阴影噪点，且特征提取仅在启动阶段执行一次，不增加后续推理耗时。
- 上限控制（不超过 8 张）：融合过多照片会触发平均脸效应，高维特征向量向人群均值过度收敛，反而稀释人物独特的眼眶、鼻梁与唇峰轮廓特征；同时源图越多，混入微小遮挡、失焦或偏色样本的概率成倍上升，易造成特征污染。

## 工作流与帧调度

### 工作流模式 (Workflow Mode)

```text
UI: N/A | CLI: --workflow-mode | INI: [workflow] workflow_mode
Choices: auto, image-to-image, image-to-video
Default: auto
```

定义媒体处理流水线的输入与输出模式。默认为 `auto`，系统根据目标素材类型自动路由至图像或视频处理管线；显式指定时，会在执行前强制校验目标素材是否与所选模式匹配。

### 帧调度策略 (Workflow Strategy)

```text
UI: WORKFLOW STRATEGY | CLI: --workflow-strategy | INI: [workflow] workflow_strategy
Choices: memory, disk
Default: memory
```

| 模式 | 运行机制 | 优势 | 劣势 | 推荐场景 |
| :--- | :--- | :--- | :--- | :--- |
| `memory` | 内存流式管道处理。边读取解码、边推理、边编码写入视频编码器 | 处理速度快，全程无临时文件落盘，无磁盘 I/O 磨损 | 占用部分内存缓冲（仅缓冲处理中的并发帧） | 绝大多数场景首选 |
| `disk` | 传统图片切片处理。先通过 FFmpeg 将视频全部抽帧为临时图片落盘，逐张处理后再拼装编码 | 解耦单帧提取与编码流程 | 产生大量临时图像切片，显著增加磁盘空间占用与 I/O 读写开销，处理速度较慢 | 仅在排查流式管道特殊环境异常需检视中间帧时使用；长视频强烈不建议使用 |

### 临时帧格式 (Temp Frame Format)

```text
UI: TEMP FRAME FORMAT | CLI: --temp-frame-format | INI: [frame_extraction] temp_frame_format
Choices: bmp, jpeg, png, tiff
Default: png
Note: Only effective when WORKFLOW STRATEGY = disk
```

默认的 `png` 格式采用无损压缩，编码与解码开销大，可能成为 I/O 吞吐瓶颈；磁盘空间充裕且追求磁盘模式极限吞吐时，可设为未压缩的 `bmp`。

### 临时像素格式 (Temp Pixel Format)

```text
UI: N/A | CLI: --temp-pixel-format | INI: [frame_extraction] temp_pixel_format
Choices: bgr24, bgra
Default: bgr24
Note: Only effective when WORKFLOW STRATEGY = memory; set to bgra to preserve the alpha channel when using background_remover
```

定义内存流式管道向视频编码器输送帧数据时的像素格式。

### 截取起始帧 (Trim Frame Start)

```text
UI: TRIM FRAME | CLI: --trim-frame-start | INI: [frame_extraction] trim_frame_start
Type: integer
Range: >= 0
Default: None
Note: Controlled by the start handle of the TRIM FRAME range slider in Web UI
```

指定待处理视频的起始帧索引（留空默认从第 `0` 帧开始）。常用于片段调试（如跳过片头黑屏），无需在渲染前手动预剪辑视频。

### 截取结束帧 (Trim Frame End)

```text
UI: TRIM FRAME | CLI: --trim-frame-end | INI: [frame_extraction] trim_frame_end
Type: integer
Range: >= 0
Default: None
Note: Controlled by the end handle of the TRIM FRAME range slider in Web UI
```

指定待处理视频的截止帧索引。留空时默认处理至视频最后一帧。

### 目标帧分发数量 (Target Frame Amount)

```text
UI: N/A | CLI: --target-frame-amount | INI: [frame_distribution] target_frame_amount
Range: 0 - 10
Step: 1
Default: 2
Note: Only effective when FACE TRACKER SCORE > 0; otherwise causes redundant frame decoding I/O without anti-flicker benefits
```

传递给后续处理器的单侧时序帧偏移半宽。系统以当前中心帧为基准，提取前后各 $K$ 帧（向处理器分发共 $2K + 1$ 帧时序邻近帧），主要配合人脸追踪器进行跨帧一致性计算与平滑滤波。

- 调大（如 `3` ~ `5`）：时序观察窗口加宽，追踪器对人脸短时间侧头、偶发漏检或光影突变的插值补偿能力更强，视频抗闪烁效果更佳；但每帧需额外读取和比对相邻帧，计算开销增加且耗时延长；若画面运动剧烈，过宽的窗口可能引入历史过期轨迹。
- 调小（如 `0` ~ `1`）：时序窗口极窄或关闭，处理速度最快；但在动态镜头下容易因单帧检测波动而出现微小闪烁。
- 推荐值：开启人脸追踪（如将人脸追踪重合度设为 `0.25`）时日常建议设为 `2` 或 `3`；若未开启人脸追踪，建议设为 `0` 以免空耗解码 I/O。

## 人脸检测与对齐

人脸检测负责在目标画面中定位人脸位置并输出边界框；对齐负责基于人脸特征点拟合几何仿射变换。

### 人脸检测模型 (Face Detector Model)

```text
UI: FACE DETECTOR MODEL | CLI: --face-detector-model | INI: [face_detector] face_detector_model
Choices: many, retinaface, scrfd, yolo_face, yunet
Default: yolo_face
```

| 模型 | 说明 |
| :--- | :--- |
| `many` | 并行运行多个检测器取并集，用于极端疑难帧 |
| `retinaface` | 经典检测网络，小尺寸人脸与复杂光照稳定 |
| `scrfd` | 侧脸及大角度动作抓取能力强，抗运动模糊好 |
| `yolo_face` | 速度快，正脸及微侧脸识别稳健，综合表现最佳 |
| `yunet` | 超轻量级设计，适合低功耗或 CPU 推理 |

### 人脸检测尺寸 (Face Detector Size)

```text
UI: FACE DETECTOR SIZE | CLI: --face-detector-size | INI: [face_detector] face_detector_size
Choices (model-dependent): 160x160, 320x320, 480x480, 512x512, 640x640
Default: 640x640
```

送入检测网络前图像的约束尺寸。尺寸越大，远距离微小人脸召回率越高；尺寸越小，推理越快。

### 人脸检测置信度 (Face Detector Score)

```text
UI: FACE DETECTOR SCORE | CLI: --face-detector-score | INI: [face_detector] face_detector_score
Range: 0.0 - 1.0
Step: 0.05
Default: 0.5
```

置信度阈值（Confidence Threshold）。若视频中侧脸或运动模糊时发生漏检，可适当下调至 `0.35` ~ `0.40`；若背景画面频繁出现将衣服、杂物误识为人脸，可提高至 `0.65` 以上。

### 人脸检测旋转角 (Face Detector Angles)

```text
UI: FACE DETECTOR ANGLES | CLI: --face-detector-angles | INI: [face_detector] face_detector_angles
Choices (multi): 0, 90, 180, 270
Default: 0
```

针对躺卧、倒立或侧翻等非正向画面。系统按指定旋转角 $\theta$ 构建中心仿射变换矩阵 $M \in \mathbb{R}^{2 \times 3}$ 并扩展画布尺寸 $(W_\theta, H_\theta)$：

$$\begin{bmatrix} W_\theta \\ H_\theta \end{bmatrix} = \begin{bmatrix} |\cos\theta| & |\sin\theta| \\ |\sin\theta| & |\cos\theta| \end{bmatrix} \begin{bmatrix} W \\ H \end{bmatrix}$$

$$M = \begin{bmatrix} \cos\theta & \sin\theta & (1 - \cos\theta)c_x - \sin\theta c_y + \frac{W_\theta - W}{2} \\ -\sin\theta & \cos\theta & \sin\theta c_x + (1 - \cos\theta)c_y + \frac{H_\theta - H}{2} \end{bmatrix}$$

其中 $(c_x, c_y) = (\frac{W}{2}, \frac{H}{2})$。在旋转图像上完成人脸检测后，通过逆仿射变换矩阵 $M_{\text{inv}} \in \mathbb{R}^{2 \times 3}$ 将检测框坐标与关键点坐标反向映射回原图坐标系：

$$\begin{bmatrix} x \\ y \end{bmatrix} = M_{\text{inv}} \begin{bmatrix} x_\theta \\ y_\theta \\ 1 \end{bmatrix}$$

> 参考：
> - [`facefusion/face_helper.py:create_rotation_matrix_and_size()`](../../facefusion/face_helper.py)
> - [`facefusion/face_detector.py:detect_faces_by_angle()`](../../facefusion/face_detector.py)

每增加一个检测角度，检测阶段耗时成倍增加。

### 人脸检测外边距 (Face Detector Margin)

```text
UI: FACE DETECTOR MARGIN | CLI: --face-detector-margin | INI: [face_detector] face_detector_margin
Format: <top> <right> <bottom> <left>
Range: 0 - 100
Step: 1
Default: 0 0 0 0
Note: Web UI uses a single slider to adjust all 4 margins equally
```

在检测前对画面边界施加外扩补零填充（Padding），避免贴近画面边缘的人脸因缺少周围上下文而被卷积网络漏检。输入边距 $(m_{\text{top}}, m_{\text{right}}, m_{\text{bottom}}, m_{\text{left}}) \in [0, 100]^4$ 线性映射为各方向边长的补零扩展像素量：

$$\Delta_{\text{top}} = \lfloor H \times \frac{m_{\text{top}}}{200} \rfloor, \quad \Delta_{\text{bottom}} = \lfloor H \times \frac{m_{\text{bottom}}}{200} \rfloor$$

$$\Delta_{\text{left}} = \lfloor W \times \frac{m_{\text{left}}}{200} \rfloor, \quad \Delta_{\text{right}} = \lfloor W \times \frac{m_{\text{right}}}{200} \rfloor$$

检测完成后，通过减去左侧与顶部边距偏移恢复原图绝对坐标：

$$\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x_{\text{padded}} - \Delta_{\text{left}} \\ y_{\text{padded}} - \Delta_{\text{top}} \end{bmatrix}$$

> 参考：
> - [`facefusion/face_detector.py:prepare_margin()`](../../facefusion/face_detector.py)

### 人脸关键点模型 (Face Landmarker Model)

```text
UI: FACE LANDMARKER MODEL | CLI: --face-landmarker-model | INI: [face_landmarker] face_landmarker_model
Choices: 2dfan4, hrffa, many, peppa_wutz
Default: 2dfan4
```

| 模型 | 说明 |
| :--- | :--- |
| `2dfan4` | 68 点密集关键点检测，抗遮挡与复杂姿态稳健 |
| `hrffa` | 68 点高分辨率特征点定位 |
| `many` | 并行运行 `2dfan4` 与 `peppa_wutz` 检测器并择优选取，提升复杂姿态稳定性 |
| `peppa_wutz` | 经典轻量模型 |

系统基于随机抽样一致算法（RANSAC，Random Sample Consensus）对检测到的 5 个核心基准关键点与各处理器模型内置的标准人脸对齐模板拟合相似变换（Similarity Transformation）矩阵 $M \in \text{Sim}(2)$，作为局部人脸几何对齐、规范化裁切与后续逆贴回的基准：

$$\min_{M} \sum_{i=1}^5 \left\| M \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix} - \vec{t}_i \right\|^2$$

> 参考：
> - [`facefusion/face_helper.py:estimate_matrix_by_face_landmark_5()`](../../facefusion/face_helper.py)

### 人脸关键点置信度 (Face Landmarker Score)

```text
UI: FACE LANDMARKER SCORE | CLI: --face-landmarker-score | INI: [face_landmarker] face_landmarker_score
Range: 0.0 - 1.0
Step: 0.05
Default: 0.5
Note: Avoid setting this to 1.0 when using hrffa, as hrffa uses a fixed confidence score of 1.0 and requires a threshold strictly below 1.0
```

人脸关键点检测置信度阈值。当 68 点关键点模型的检测得分高于该阈值时，系统将其收敛转换为高精度 5 点对齐基准；若低于阈值，则回退采用人脸检测器输出的初始 5 点关键点（不会丢弃人脸）；设为 `0` 时直接跳过 68 点关键点模型推理。

## 人脸筛选、定位与时序跟踪

### 人脸筛选模式 (Face Selector Mode)

```text
UI: FACE SELECTOR MODE | CLI: --face-selector-mode | INI: [face_selector] face_selector_mode
Choices: many, one, reference
Default: reference
```

| 模式 | 匹配逻辑 | 适用场景 |
| :--- | :--- | :--- |
| `many` | 替换画面中识别到的全部人脸 | 全员替换 |
| `one` | 仅替换单张人脸（配合排序规则） | 画面仅单人，或固定只替换最大人脸 |
| `reference` | 基于选定基准帧人脸的特征向量距离进行比对 | 多人同框画面中只替换指定主角 |

### 人脸排序规则 (Face Selector Order)

```text
UI: FACE SELECTOR ORDER | CLI: --face-selector-order | INI: [face_selector] face_selector_order
Choices:
  - Size: large-small, small-large
  - Score: best-worst, worst-best
  - Horizontal: left-right, right-left
  - Vertical: top-bottom, bottom-top
Default: large-small
```

定义画面中多张人脸的排序依据。`large-small` 优先处理前景主角，`left-right` 按水平物理方位排序。

### 人脸性别筛选 (Face Selector Gender)

```text
UI: FACE SELECTOR GENDER | CLI: --face-selector-gender | INI: [face_selector] face_selector_gender
Choices: auto, female, male
Default: None
Note: Web UI provides an additional 'none' option; omit in CLI or leave empty in INI for None
```

按性别属性过滤画面人脸：
- `None`：不执行性别过滤，处理画面中所有符合检测条件的人脸。
- `auto`：系统自动识别源素材中最大人脸的性别作为基准，过滤目标人脸（仅保留与该源人脸同性别的人脸，排除异性人脸）。
- `female` / `male`：显式强制仅保留判定为指定性别的人脸，配合男女群演镜头进行精准排除。

### 人脸人种筛选 (Face Selector Race)

```text
UI: FACE SELECTOR RACE | CLI: --face-selector-race | INI: [face_selector] face_selector_race
Choices: auto, arabic, asian, black, indian, latino, white
Default: None
Note: Web UI provides an additional 'none' option; omit in CLI or leave empty in INI for None
```

按人种特征分类器过滤非目标人脸：
- `None`：不执行人种过滤，处理画面中所有符合检测条件的人脸。
- `auto`：系统自动识别源素材中最大人脸的人种分类作为基准，过滤目标人脸（仅保留与该源人脸同人种分类的人脸）。
- 指定人种（如 `asian`、`white` 等）：显式强制仅保留判定为该人种分类的人脸。

### 人脸起始年龄 (Face Selector Age Start)

```text
UI: FACE SELECTOR AGE | CLI: --face-selector-age-start | INI: [face_selector] face_selector_age_start
Range: 0 - 100
Step: 1
Default: None
Note: Controlled by the start handle of the FACE SELECTOR AGE range slider in Web UI. In CLI/INI, must be specified together with FACE SELECTOR AGE END
```

人脸识别的最低估算年龄下限。

### 人脸截止年龄 (Face Selector Age End)

```text
UI: FACE SELECTOR AGE | CLI: --face-selector-age-end | INI: [face_selector] face_selector_age_end
Range: 0 - 100
Step: 1
Default: None
Note: Controlled by the end handle of the FACE SELECTOR AGE range slider in Web UI. In CLI/INI, must be specified together with FACE SELECTOR AGE START
```

人脸识别的最高估算年龄上限。

### 基准帧编号 (Reference Frame Number)

```text
UI: REFERENCE FRAME NUMBER | CLI: --reference-frame-number | INI: [face_selector] reference_frame_number
Type: integer
Range: >= 0
Default: 0
Note: In Web UI, this is set by scrubbing the PREVIEW FRAME slider when selecting a reference face
```

指定从视频的第几帧截取基准人脸（0-indexed）。若视频开头有黑屏或片头动画，须跳过并指向主角清晰出现的帧编号。

### 基准人脸索引 (Reference Face Position)

```text
UI: REFERENCE FACE POSITION | CLI: --reference-face-position | INI: [face_selector] reference_face_position
Type: integer
Range: >= 0
Default: 0
Note: Selected visually by clicking a face thumbnail in the REFERENCE FACE gallery in Web UI
```

指定在基准帧的多张检测人脸中（按当前排序规则排列），选择第几张人脸作为基准对象（索引从 0 开始）。

### 基准人脸匹配阈值 (Reference Face Distance)

```text
UI: REFERENCE FACE DISTANCE | CLI: --reference-face-distance | INI: [face_selector] reference_face_distance
Range: 0.0 - 1.0
Step: 0.05
Default: 0.3
```

系统计算目标人脸与基准人脸的 $L_2$ 归一化单位特征向量之间的归一化余弦距离（Normalized Cosine Distance），并线性映射至 $[0, 1]$：

$$d_{\text{norm}} = \frac{1 - \hat{e}_{\text{target}} \cdot \hat{e}_{\text{ref}}}{2}$$

> 参考：
> - [`facefusion/face_selector.py:compare_faces()`](../../facefusion/face_selector.py)

当 $d_{\text{norm}} < \theta_{\text{ref}}$（即设定的匹配阈值）时判定为同一人。
- 调大（如 `0.35` ~ `0.45`）：放宽判定条件，解决主角转头、光影突变时断连漏换；但过大可能误将他人识别为目标。
- 调小（如 `0.20` ~ `0.25`）：严苛匹配，杜绝认错人；但在侧脸或大表情时可能漏检。

### 人脸追踪重合度 (Face Tracker Score)

```text
UI: FACE TRACKER SCORE | CLI: --face-tracker-score | INI: [face_tracker] face_tracker_score
Range: 0.0 - 0.5
Step: 0.05
Default: 0.0
```

人脸多帧追踪重合度阈值，基于时序边界框交并比（IoU，Intersection over Union）。设为大于 `0`（如 `0.20` ~ `0.30`）时开启时序追踪。系统跟踪人脸运动轨迹，并对偶发漏检帧进行空间插值补偿，降低视频闪烁。

## 核心人脸处理器 (Processors)

### 处理器管线 (Processors)

```text
UI: PROCESSORS | CLI: --processors | INI: [processors] processors
Choices (multi, ordered): face_swapper, deep_swapper, expression_restorer, face_editor, lip_syncer, age_modifier, face_enhancer, frame_colorizer, frame_enhancer, background_remover, face_debugger
Default: face_swapper
```

指定当前工作流启用的处理器模块列表及其串行执行顺序。系统严格按照命令行或界面中指定的参数顺序自左向右依次传递图像帧：

| 处理阶段 | 处理器 | 模块职责 | 建议接入顺序与典型作用 |
| :--- | :--- | :--- | :--- |
| 核心换脸<br/>(二选一) | `face_swapper` | 核心人脸替换 | 基础换脸首位接入 |
| | `deep_swapper` | 深度换脸模型通道 | 特殊换脸模型接入 |
| 表情与动态微调 | `expression_restorer` | 表情动态补偿还原 | 紧接换脸处理器，修复面部僵硬与大幅度表情丢失 |
| | `face_editor` | 细微面部表情与姿态微调 | 换脸后微调视线方向、眨眼幅度与头部微姿态 |
| | `lip_syncer` | 基于音频驱动口型动作同步 | 换脸与表情调整后，匹配外部输入音频口型 |
| | `age_modifier` | 面部年龄增减修饰 | 面部幼年化或老化微调 |
| 人脸细节超分 | `face_enhancer` | 人脸超分辨率细节修复 | 关键顺序：必须置于所有面部形变与唇形编辑之后，修复面部五官高清纹理（若提前执行，后续编辑网络会破坏重建的高清微纹理） |
| 整帧全局后处理 | `frame_colorizer` | 黑白老旧画面色彩还原 | 贴回后对整帧画面进行色彩还原（先于整帧超分以节省计算开销） |
| | `frame_enhancer` | 整帧超分辨率放大 | 贴回与着色后，对整帧画面进行无损超分辨率放大 |
| | `background_remover` | 视频人像抠图背景移除 | 移除人像背景或输出 Alpha 透明通道 |
| 调参排错诊断 | `face_debugger` | 调参排错诊断可视化 | 调试专用，置于管线末端在画面叠加渲染框、关键点与遮罩网格 |

本指南重点详述常用三大核心人脸处理器：核心换脸 (`face_swapper`)、表情动态补偿 (`expression_restorer`) 与人脸超分细节修复 (`face_enhancer`)。

### 核心换脸 (Face Swapper)

#### 换脸模型 (Face Swapper Model)

```text
UI: FACE SWAPPER MODEL | CLI: --face-swapper-model | INI: [processors] face_swapper_model
Choices: alphaface_256, blendswap_256, ghost_1_256, ghost_2_256, ghost_3_256, hififace_unofficial_256, hyperswap_1a_256, hyperswap_1b_256, hyperswap_1c_256, inswapper_128, inswapper_128_fp16, simswap_256, simswap_unofficial_512, uniface_256
Default: hyperswap_1a_256
```

| 模型 | 说明 |
| :--- | :--- |
| `alphaface_256` | 基于 AlphaFace 架构，平衡相似度与光照适应 |
| `blendswap_256` | 侧重面部边缘与轮廓自然融合，减少人工拼接感 |
| `ghost_1_256` ~ `ghost_3_256` | Ghost 系列模型，支持多尺度与特征细化 |
| `hififace_unofficial_256` | 基于 HiFiFace 结构，主打高保真五官与几何纹理保留 |
| `hyperswap_1a_256` | 姿态贴合与稳定性高，综合表现稳健 |
| `hyperswap_1b_256` | 介于 1a 与 1c 之间的特征平衡 |
| `hyperswap_1c_256` | 特征保真度好，细节表现丰富 |
| `inswapper_128` | 经典轻量模型，五官相似度高 |
| `inswapper_128_fp16` | 半精度版本，显存与计算开销较小 |
| `simswap_256` | 五官特征表达鲜明 |
| `simswap_unofficial_512` | 512 尺寸扩展版本 |
| `uniface_256` | 统一换脸架构，适配多样化人脸姿态与光照 |

#### 像素提升 (Face Swapper Pixel Boost)

```text
UI: FACE SWAPPER PIXEL BOOST | CLI: --face-swapper-pixel-boost | INI: [processors] face_swapper_pixel_boost
Choices:
  - 128 Native (inswapper_128): 128x128, 256x256, 384x384, 512x512, 768x768, 1024x1024
  - 256 Native (hyperswap, uniface): 256x256, 512x512, 768x768, 1024x1024 (blendswap adds 384x384)
  - 512 Native (simswap_512): 512x512, 768x768, 1024x1024
Default: Model native resolution (such as 256x256, 128x128, or 512x512)
Note: Must be an integer multiple of the model's native resolution (avoid 384x384 for blendswap_256 and 768x768 for simswap_unofficial_512)
```

通过像素解交织（Pixel Unshuffle）与周期性多相位子采样（Periodic Multi-phase Sub-sampling）推理突破原生模型固定尺寸限制。根据指定的 Boost 分辨率与原生模型尺寸计算整数网格比例系数 $k$（$k = W_{\text{boost}} / W_{\text{model}} \in \mathbb{N}^+$）：

$$k = \frac{W_{\text{boost}}}{W_{\text{model}}}$$

将高分辨率人脸裁切张量分解为 $k^2$ 个模型原生尺寸的完整子视图分别推理，再重新交错编织还原：

$$\mathbb{R}^{k H_{\text{model}} \times k W_{\text{model}} \times 3} \xrightarrow{\text{implode}} \mathbb{R}^{k^2 \times H_{\text{model}} \times W_{\text{model}} \times 3} \xrightarrow{\text{explode}} \mathbb{R}^{k H_{\text{model}} \times k W_{\text{model}} \times 3}$$

> 参考：
> - [`facefusion/processors/pixel_boost.py:implode_pixel_boost()`](../../facefusion/processors/pixel_boost.py)
> - [`facefusion/processors/pixel_boost.py:explode_pixel_boost()`](../../facefusion/processors/pixel_boost.py)

不同于局部切块，每张子图均保留完整五官拓扑结构，生成式对抗网络（GAN，Generative Adversarial Network）换脸模型能在保持全局面部结构一致性的前提下，输出极高清晰度的皮肤微纹理。日常推荐 `512x512`（画质速度平衡）；特写镜头推荐 `1024x1024`；低显存设备建议设为与模型同宽的原生尺寸（如 `256x256`）。

#### 换脸权重 (Face Swapper Weight)

```text
UI: FACE SWAPPER WEIGHT | CLI: --face-swapper-weight | INI: [processors] face_swapper_weight
Range: 0.0 - 1.0
Step: 0.05
Default: 0.5
Note: Only applies to embedding-based swap models; has no effect on blendswap or uniface
```

输入权重 $w \in [0, 1]$ 线性映射为加权系数 $\alpha = 0.35 - 0.7w$（默认 $w=0.5$ 时 $\alpha=0$，完全保留源人脸特征），对源人脸与目标人脸的特征向量进行加权融合与仿射外推（Affine Extrapolation）：

$$\vec{e}_{\text{mixed}} = (1 - \alpha)\vec{e}_{\text{source}} + \alpha\vec{e}_{\text{target}}$$

> 参考：
> - [`facefusion/processors/modules/face_swapper/core.py:balance_source_embedding()`](../../facefusion/processors/modules/face_swapper/core.py)

- 调小（`0.0` ~ `0.3`）：$\alpha > 0$。当 $w = 0.0$ 时达到边界 $\alpha = 0.35$（保留 $65\%$ 源人脸特征，融入 $35\%$ 目标人脸特征），使五官结构更自然贴合目标人物的面部轮廓。
- 居中（`0.5`，默认）：$\alpha = 0$。$\vec{e}_{\text{mixed}} = \vec{e}_{\text{source}}$，纯粹采用源人脸特征向量，不混入任何目标特征。
- 调大（`0.7` ~ `1.0`）：$\alpha < 0$。系统执行仿射外推；当 $w = 1.0$ 时达到极值 $\alpha = -0.35$（叠加 $135\%$ 源特征并反向扣除 $35\%$ 目标特征），以反向对比形式极度强化源人脸辨识度（但可能放大局部光影噪点）。

### 表情还原 (Expression Restorer)

#### 表情还原模型 (Expression Restorer Model)

```text
UI: EXPRESSION RESTORER MODEL | CLI: --expression-restorer-model | INI: [processors] expression_restorer_model
Choices: live_portrait
Default: live_portrait
```

基于 LivePortrait 动作驱动框架，通过动态动作特征体积（Feature Volume）与动作点位（Motion Points）变换，将目标视频原人脸的微表情实时迁移补偿至换脸后的面部张量上。

#### 表情还原系数 (Expression Restorer Factor)

```text
UI: EXPRESSION RESTORER FACTOR | CLI: --expression-restorer-factor | INI: [processors] expression_restorer_factor
Range: 0 - 100
Step: 1
Default: 80
```

表情还原强度。输入值 $f$ 线性映射为 $[0, 1.2]$ 倍表情系数进行补偿插值：

$$E_{\text{out}} = \frac{1.2f}{100} E_{\text{target}} + \left(1 - \frac{1.2f}{100}\right) E_{\text{temp}}$$

> 参考：
> - [`facefusion/processors/modules/expression_restorer/core.py:apply_restore()`](../../facefusion/processors/modules/expression_restorer/core.py)
> - [`facefusion/processors/modules/expression_restorer/core.py:restore_expression()`](../../facefusion/processors/modules/expression_restorer/core.py)

设为 `80`（对应系数 $0.96$）能自然复原目标人物眨眼、扬眉、大笑时的五官形变，消除面部僵硬感；调大至 `100` 产生超额超调补偿（$1.2$ 倍），适合夸张喜剧表情。

#### 表情还原部位 (Expression Restorer Areas)

```text
UI: EXPRESSION RESTORER AREAS | CLI: --expression-restorer-areas | INI: [processors] expression_restorer_areas
Choices (multi): upper-face, lower-face
Default: upper-face lower-face
```

控制仅还原眉眼动作（`upper-face`）或仅还原唇齿动作（`lower-face`）。

### 人脸高清增强 (Face Enhancer)

#### 人脸增强模型 (Face Enhancer Model)

```text
UI: FACE ENHANCER MODEL | CLI: --face-enhancer-model | INI: [processors] face_enhancer_model
Choices: codeformer, gfpgan_1.2, gfpgan_1.3, gfpgan_1.4, gpen_bfr_256, gpen_bfr_512, gpen_bfr_1024, gpen_bfr_2048, restoreformer_plus_plus
Default: gfpgan_1.4
```

| 模型 | 特征与演进方向 |
| :--- | :--- |
| `codeformer` | 基于离散可学习码本先验（VQ Codebook，Vector Quantized Codebook）的网络。五官线条（眼眶、睫毛、牙齿）雕刻锐利清晰，支持通过权重调节在“AI 高清重建”与“保全原特征”之间连续平衡，抗重度模糊与低分辨率能力最强。 |
| `gfpgan_1.2`<br/>`gfpgan_1.3`<br/>`gfpgan_1.4` | 基于生成面部先验（GFP，Generative Facial Prior）的修复网络。1.2 -> 1.3 -> 1.4 逐步提升了面部先验保真度并减少伪影；1.4 版本在平滑肤色、消除噪点与面部自然真实感上综合平衡最佳，首选默认模型。 |
| `gpen_bfr_256`<br/>`gpen_bfr_512`<br/>`gpen_bfr_1024`<br/>`gpen_bfr_2048` | 基于 GAN 逆映射架构的面部增强网络。从 256/512 基础尺寸演进至 1024/2048 超高清分辨率。分辨率越高，重建的皮肤毛孔微纹理与高频细节越细腻，但显存开销与计算耗时呈指数增加。 |
| `restoreformer_plus_plus` | 完全基于多头交叉注意力机制（Cross-Attention）的高清修复网络。相比传统 GAN 模型更擅长结合局部先验修复大角度侧脸和严重损伤人脸，五官质感相对柔和自然。 |

#### 人脸增强混合度 (Face Enhancer Blend)

```text
UI: FACE ENHANCER BLEND | CLI: --face-enhancer-blend | INI: [processors] face_enhancer_blend
Range: 0 - 100
Step: 1
Default: 80
```

增强结果的线性混合透明度。输出画面为原图与增强图的加权合成：

$$I_{\text{out}} = \left(1 - \frac{\text{blend}}{100}\right) I_{\text{raw}} + \frac{\text{blend}}{100} I_{\text{enhanced}}$$

> 参考：
> - [`facefusion/processors/modules/face_enhancer/core.py:blend_paste_frame()`](../../facefusion/processors/modules/face_enhancer/core.py)

推荐设置为 `60` ~ `80`，透出原图部分自然噪点，避免过度平滑导致的涂抹感与皮肤纹理丢失。

#### 人脸增强权重 (Face Enhancer Weight)

```text
UI: FACE ENHANCER WEIGHT | CLI: --face-enhancer-weight | INI: [processors] face_enhancer_weight
Range: 0.0 - 1.0
Step: 0.05
Default: 0.5
Note: Only applies to codeformer; has no effect on other face enhancer models
```

针对 `codeformer` 的保真度权重（Fidelity Weight）。调低更偏向 AI 锐化与细节重建，调高更偏向保全原有人脸结构特征。

## 动态遮罩工程

遮罩（Mask）控制换脸后图像向原图写回的 Alpha 混合权重，是消除接缝、方框与穿模的核心。

### 遮罩类型 (Face Mask Types)

```text
UI: FACE MASK TYPES | CLI: --face-mask-types | INI: [face_masker] face_mask_types
Choices (multi): box, occlusion, region, area
Default: box
Note: Only face_swapper, deep_swapper, and face_debugger support all 4 mask types; face_enhancer, expression_restorer, and age_modifier always enforce a 0-padding box mask with optional occlusion; face_editor uses only a box mask; lip_syncer supports optional occlusion plus a box mask (edtalk) or fixed lower-face area mask (wav2lip)
```

各遮罩类型按生效机制说明如下：

| 遮罩类型 | 机制与作用 |
| :--- | :--- |
| `box` | 外周矩形根据参数计算高斯羽化边缘，保证贴回区域与原图平滑融合 |
| `occlusion` | 神经网络动态检测头发、手指、麦克风等遮挡物并在遮罩中剔除 |
| `region` | 语义分割人脸器官，仅替换或剔除特定人脸区域 |
| `area` | 基于关键点凸包多边形（Convex Hull）控制上下半脸生效范围 |

所有启用的遮罩通过取交集（逐点最小值）形成最终单通道遮罩，经逆仿射变换对齐后执行 Alpha 混合无缝贴回原图画面：

$$M_{\text{final}} = \min(M_{\text{box}}, M_{\text{occlusion}}, M_{\text{area}}, M_{\text{region}})$$

$$I_{\text{out}} = I_{\text{target}} \cdot (1 - M_{\text{final}}) + I_{\text{warped}} \cdot M_{\text{final}}$$

> 参考：
> - [`facefusion/face_helper.py:paste_back()`](../../facefusion/face_helper.py)

### 遮罩羽化模糊 (Face Mask Blur)

```text
UI: FACE MASK BLUR | CLI: --face-mask-blur | INI: [face_masker] face_mask_blur
Range: 0.0 - 1.0
Step: 0.05
Default: 0.3
```

控制矩形边缘的高斯模糊过渡带宽。高斯核标准差 $\sigma$ 与裁切宽度 $W_{\text{crop}}$ 及模糊系数成正比：

$$\text{blur\_amount} = \lfloor W_{\text{crop}} \times 0.5 \times \text{blur} \rfloor$$

$$\sigma = 0.25 \times \text{blur\_amount}$$

$$M_{\text{box}} = M_{\text{initial}} * G_\sigma$$

> 参考：
> - [`facefusion/face_masker.py:create_box_mask()`](../../facefusion/face_masker.py)

调小边缘锐利，过小会有接缝；调大过渡平缓，过大会残留原人脸边缘。

### 遮罩内缩边距 (Face Mask Padding)

```text
UI: FACE MASK PADDING | CLI: --face-mask-padding | INI: [face_masker] face_mask_padding
Format: <top> <right> <bottom> <left>
Range: 0 - 100
Step: 1
Default: 0 0 0 0
Note: Web UI splits this into 4 separate sliders (TOP, RIGHT, BOTTOM, LEFT). Only applies to face_swapper, deep_swapper, lip_syncer (edtalk_256 only), and face_debugger; has no effect on face_enhancer, expression_restorer, face_editor, or age_modifier
```

对矩形遮罩实施边界内缩，按百分比裁剪有效区域。四个边界的内缩阈值结合了最小边界保护（`1` 像素）、高斯过渡半宽与内缩百分比：

$$d_{\text{top}} = \max\left(1, \lfloor \frac{\text{blur\_amount}}{2} \rfloor, \lfloor H_{\text{crop}} \times \frac{p_{\text{top}}}{100} \rfloor\right), \quad d_{\text{bottom}} = \max\left(1, \lfloor \frac{\text{blur\_amount}}{2} \rfloor, \lfloor H_{\text{crop}} \times \frac{p_{\text{bottom}}}{100} \rfloor\right)$$

$$d_{\text{left}} = \max\left(1, \lfloor \frac{\text{blur\_amount}}{2} \rfloor, \lfloor W_{\text{crop}} \times \frac{p_{\text{left}}}{100} \rfloor\right), \quad d_{\text{right}} = \max\left(1, \lfloor \frac{\text{blur\_amount}}{2} \rfloor, \lfloor W_{\text{crop}} \times \frac{p_{\text{right}}}{100} \rfloor\right)$$

在边界内缩区域处置零后执行高斯滤波，常用于消除额头碎发或下巴边缘的生硬轮廓。

> 参考：
> - [`facefusion/face_masker.py:create_box_mask()`](../../facefusion/face_masker.py)

### 遮挡模型 (Face Occluder Model)

```text
UI: FACE OCCLUDER MODEL | CLI: --face-occluder-model | INI: [face_masker] face_occluder_model
Choices: many, xseg_1, xseg_2, xseg_3
Default: xseg_1
```

| 模型 | 说明 |
| :--- | :--- |
| `many` | 并行运行多个遮挡模型并取交集 |
| `xseg_1` | 经典遮挡分割模型，速度与精度均衡 |
| `xseg_2` | 针对发丝与手指边缘的细粒度分割 |
| `xseg_3` | 进阶优化版本 |

### 遮罩部位范围 (Face Mask Areas)

```text
UI: FACE MASK AREAS | CLI: --face-mask-areas | INI: [face_masker] face_mask_areas
Choices (multi): upper-face, lower-face, mouth
Default: upper-face lower-face mouth
```

在启用 `area` 遮罩类型时生效。控制仅替换上半脸、下半脸或嘴唇部位。

### 人脸解析模型 (Face Parser Model)

```text
UI: FACE PARSER MODEL | CLI: --face-parser-model | INI: [face_masker] face_parser_model
Choices: bisenet_resnet_18, bisenet_resnet_34
Default: bisenet_resnet_34
```

| 模型 | 说明 |
| :--- | :--- |
| `bisenet_resnet_18` | 轻量快速版本 |
| `bisenet_resnet_34` | 人脸器官语义分割高精度模型 |

### 遮罩器官细分 (Face Mask Regions)

```text
UI: FACE MASK REGIONS | CLI: --face-mask-regions | INI: [face_masker] face_mask_regions
Choices (multi): skin, left-eyebrow, right-eyebrow, left-eye, right-eye, glasses, nose, mouth, upper-lip, lower-lip
Default: skin left-eyebrow right-eyebrow left-eye right-eye glasses nose mouth upper-lip lower-lip
```

在启用 `region` 遮罩类型时生效。可按器官精细控制保留或替换项。例如若需保留目标演员原生的黑框眼镜，在列表中剔除 `glasses` 即可。

## 媒体编码与输出

FaceFusion 依托 FFmpeg 执行多媒体的帧流读取、像素格式转换、音轨混流与最终封装，各项输出参数均通过内部换算映射为 FFmpeg 编码指令。

### 输出图像质量 (Output Image Quality)

```text
UI: OUTPUT IMAGE QUALITY | CLI: --output-image-quality | INI: [output_creation] output_image_quality
Range: 0 - 100
Step: 1
Default: 80
```

输出图像的编码格式由输出文件路径的扩展名自动推断决定，不同格式在底层采用完全不同的量化策略：
- JPEG 格式（`.jpg`、`.jpeg`）：通过线性插值反向换算为 FFmpeg `mjpeg` 编码器的量化标度（取值范围 `1` ~ `31`），公式为 $q_{\text{scale}} = \text{round}(31 \times (1 - q/100))$。推荐 `80`（对应 `-q:v 6`）。
- PNG 格式（`.png`）：固有 DEFLATE 无损压缩格式，输出像素 100% 保持一致，质量数值被底层完全忽略。
- WebP 格式（`.webp`）：直接将输入质量数值透传给 FFmpeg `-q:v` 参数（取值范围 `0` ~ `100`）。推荐 `80` ~ `90`。

> 参考：
> - [`facefusion/ffmpeg_builder.py:set_image_quality()`](../../facefusion/ffmpeg_builder.py)
> - [FFmpeg Documentation - MJPEG Image Encoder](https://ffmpeg.org/ffmpeg-codecs.html#mjpeg)
> - [FFmpeg Documentation - PNG Image Encoder](https://ffmpeg.org/ffmpeg-codecs.html#png)
> - [Portable Network Graphics (PNG) Specification](https://www.w3.org/TR/png/)
> - [FFmpeg Documentation - libwebp Image Encoder](https://ffmpeg.org/ffmpeg-codecs.html#libwebp)
> - [WebP Compression Techniques](https://developers.google.com/speed/webp/docs/compression)

### 输出图像缩放 (Output Image Scale)

```text
UI: OUTPUT IMAGE SCALE | CLI: --output-image-scale | INI: [output_creation] output_image_scale
Range: 0.25 - 8.0
Step: 0.25
Default: 1.0
```

输出静态图像分辨率相对目标素材的缩放倍率。

### 输出音频编码器 (Output Audio Encoder)

```text
UI: OUTPUT AUDIO ENCODER | CLI: --output-audio-encoder | INI: [output_creation] output_audio_encoder
Choices: aac, flac, libmp3lame, libopus, libvorbis, pcm_s16le, pcm_s32le
Default: flac
```

指定最终输出媒体的伴音编码器。若目标视频包含音频轨道，系统在完成逐帧画面渲染后，通过 FFmpeg 将原音轨抽取、转码并复用混流至输出容器中：
- `aac`：通用高级音频编码，适合 MP4 容器且需在网页与移动端播放。
- `flac`：无损音频编码，适合高质量母带导出。

> 参考：
> - [`facefusion/ffmpeg_builder.py:set_audio_encoder()`](../../facefusion/ffmpeg_builder.py)

### 输出音频质量 (Output Audio Quality)

```text
UI: OUTPUT AUDIO QUALITY | CLI: --output-audio-quality | INI: [output_creation] output_audio_quality
Range: 0 - 100
Step: 1
Default: 80
```

系统将输入的音频质量数值反向线性换算为底层编码器的速率或质量控制参数：
- `aac`：线性映射为 FFmpeg 原生动态可变码率（VBR）因子 `-q:a`（取值范围 `0.1` ~ `2.0`），推荐 `80`。
- `flac`：固有无损压缩格式，输出音质无损保留，质量数值被底层直接忽略。

> 参考：
> - [`facefusion/ffmpeg_builder.py:set_audio_quality()`](../../facefusion/ffmpeg_builder.py)

### 输出音频音量 (Output Audio Volume)

```text
UI: OUTPUT AUDIO VOLUME | CLI: --output-audio-volume | INI: [output_creation] output_audio_volume
Range: 0 - 100
Step: 1
Default: 100
```

控制输出伴音的增益倍率。系统通过 FFmpeg 音频滤镜 `-filter:a volume=v/100` 对音轨施加线性缩放：
- `100`：保持原始音量（增益系数 $1.0$）。
- 调小（如 `50`）：音量衰减至 $50\%$。
- `0`：静音输出。

> 参考：
> - [`facefusion/ffmpeg_builder.py:set_audio_volume()`](../../facefusion/ffmpeg_builder.py)

### 输出视频编码器 (Output Video Encoder)

```text
UI: OUTPUT VIDEO ENCODER | CLI: --output-video-encoder | INI: [output_creation] output_video_encoder
Choices:
  - CPU: libx264, libx264rgb, libx265, libvpx-vp9, rawvideo
  - NVIDIA: h264_nvenc, hevc_nvenc
  - AMD: h264_amf, hevc_amf
  - Intel: h264_qsv, hevc_qsv
  - Apple: h264_videotoolbox, hevc_videotoolbox
Default: libx264
```

### 输出视频质量 (Output Video Quality)

```text
UI: OUTPUT VIDEO QUALITY | CLI: --output-video-quality | INI: [output_creation] output_video_quality
Range: 0 - 100
Step: 1
Default: 80
Note: For NVENC, avoid setting quality to 100 (-cq 0 disables constant quality and falls back to low-bitrate VBR); use 99 for maximum quality
```

系统将输入的质量数值 $q \in [0, 100]$ 线性反向换算为底层编码器的量化与速率控制参数：
- CPU 软件编码器（`libx264`、`libx265`）：使用恒定速率因子（CRF，Constant Rate Factor），基于心理视觉模型动态调节宏块量化步长。
- NVIDIA NVENC 硬件编码器（`h264_nvenc`、`hevc_nvenc`）：使用恒定质量级别（CQ，Constant Quality），约束硬件可变码率的目标量化基准。

$$\text{CRF / CQ} = \text{round}\left(51 \times \left(1 - \frac{q}{100}\right)\right)$$

> 参考：
> - [`facefusion/ffmpeg_builder.py:set_video_quality()`](../../facefusion/ffmpeg_builder.py)
> - [FFmpeg H.264 Video Encoding Guide](https://trac.ffmpeg.org/wiki/Encode/H.264)
> - [FFmpeg H.265 Video Encoding Guide](https://trac.ffmpeg.org/wiki/Encode/H.265)
> - [FFmpeg HWAccelIntro - NVENC](https://trac.ffmpeg.org/wiki/HWAccelIntro#NVENC)
> - [Using FFmpeg with NVIDIA GPU Hardware Acceleration](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/)

主流编码器的推荐档位与底层映射：

CPU 软件编码器：`libx264`（CRF 模式）
- 绝对无损：`100` (`-crf 0`，体积庞大，母版归档)
- 影视母带：`80` (`-crf 10`，特写镜头渲染、商业交付)
- 视觉无损：`65` (`-crf 18`，肉眼难辨差异，高质量导出首选)
- 标准均衡：`55` (`-crf 23`，经典压缩平衡档，常规首选)
- 极速草稿：`45` (`-crf 28`，体积极小，快速打样)

CPU 软件编码器：`libx265`（CRF 模式）
- 绝对无损：`100` (`-crf 0`)
- 影视母带：`80` (`-crf 10`)
- 视觉无损：`55` (`-crf 23`，因 x265 压缩效率更高，CRF 23 即达视觉无损)
- 标准均衡：`45` (`-crf 28`，网络流媒体分发首选)
- 极速草稿：`35` (`-crf 33`)

NVIDIA NVENC 硬件编码器：`h264_nvenc`（CQ 模式）
- 影视母带：`80` (`-cq 10`，高码率归档)
- 视觉无损：`60` (`-cq 20`，常规高质量导出首选基准点；当 CQ 超过 23 时，平坦肤色区易出现宏块色带)
- 标准均衡：`50` (`-cq 26`)
- 极速草稿：`40` (`-cq 31`)

NVIDIA NVENC 硬件编码器：`hevc_nvenc`（CQ 模式）
- 影视母带：`80` (`-cq 10`，母带归档，码率较高)
- 视觉无损：`50` (`-cq 26`，基于 $64 \times 64$ CTU 与 SAO 环路滤波，肤质过渡平滑，码率显著低于 H.264 同级画质)
- 标准均衡：`40` (`-cq 31`，流媒体与高效率分发首选)
- 极速草稿：`30` (`-cq 36`)

### 输出视频预设 (Output Video Preset)

```text
UI: OUTPUT VIDEO PRESET | CLI: --output-video-preset | INI: [output_creation] output_video_preset
Choices: veryslow, slower, slow, medium, fast, faster, veryfast, superfast, ultrafast
Default: veryfast
```

编码速度与压缩率平衡预设。不同编码器在底层执行不同的调度策略：

> 参考：
> - [`facefusion/ffmpeg_builder.py:set_video_preset()`](../../facefusion/ffmpeg_builder.py)
> - [`facefusion/ffmpeg_builder.py:map_nvenc_preset()`](../../facefusion/ffmpeg_builder.py)
> - [FFmpeg H.264 Video Encoding Guide - Preset](https://trac.ffmpeg.org/wiki/Encode/H.264#Preset)
> - [NVIDIA Video Codec SDK - Presets](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/#presets)

CPU 软件编码器预设调度 (`libx264` / `libx265`)
- 直接透传所选预设给 FFmpeg `-preset` 参数，完整支持 9 档细粒度算法调度：
  - 极限穷举：`veryslow`（全尺寸宏块率失真优化，速度极慢，边际收益微弱）
  - 深度压缩：`slower` ~ `slow`（深度多参考帧与自适应 B 帧决策，商业级交付）
  - 标准均衡：`medium`（开启全面宏块与运动搜索，压缩率与画质达到经典平衡）
  - 常规快速：`fast` ~ `faster`（启用基础运动估计与亚像素细化，速度与体积适中）
  - 极速草稿：`veryfast` ~ `ultrafast`（关闭大部分耗时搜索，耗时极短）

NVIDIA NVENC 硬件预设映射 (`h264_nvenc` / `hevc_nvenc`)
- 底层芯片仅提供 3 档硬件流水线预设，FaceFusion 自动执行归一化映射：
  - 高质量档：传入 `veryslow`、`slower`、`slow` 均自动归一化映射为 `-preset slow`
  - 标准平衡档：传入 `medium` 映射为 `-preset medium`
  - 极速草稿档：传入 `fast`、`faster`、`veryfast`、`superfast`、`ultrafast` 均映射为 `-preset fast`

### 输出视频缩放 (Output Video Scale)

```text
UI: OUTPUT VIDEO SCALE | CLI: --output-video-scale | INI: [output_creation] output_video_scale
Range: 0.25 - 8.0
Step: 0.25
Default: 1.0
```

输出视频分辨率相对目标素材的缩放倍率。小于 `1.0` 时缩减分辨率（加速渲染与节省显存）；大于 `1.0` 时拉伸放大。

### 输出视频帧率 (Output Video FPS)

```text
UI: OUTPUT VIDEO FPS | CLI: --output-video-fps | INI: [output_creation] output_video_fps
Type: float
Range: 1.0 - 60.0
Step: 0.01 (Web UI)
Default: None
Note: Preserves the target video's native FPS when omitted
```

强制重采样输出视频的帧率。

## 执行引擎与显存管理

### 执行加速提供器 (Execution Providers)

```text
UI: EXECUTION PROVIDERS | CLI: --execution-providers | INI: [execution] execution_providers
Choices (multi, fallback queue): coreml, cpu, cuda, directml, migraphx, openvino, qnn, rocm, tensorrt
Default: Automatically select the optimal provider for the current environment
```

定义底层 ONNX Runtime 的模型计算硬件后端。

主流执行器说明：
- `cpu`：基于 CPU 的 ONNX Runtime 原生运算执行器。不依赖任何独立显卡或专用驱动，全平台通用兼容，但推理速度慢；适用于无专用 GPU 环境下的功能验证、流程排错或轻量测试。
- `cuda`：基于 NVIDIA CUDA 的标准 GPU 加速执行器。驱动与运行环境成熟，算子兼容性完整，显存管理稳健；为 NVIDIA 显卡日常处理与生产环境的首选默认配置。
- `tensorrt`：基于 NVIDIA TensorRT 的深度硬件优化执行器。首次运行会自动构建并缓存硬件序列化的 TensorRT Engine（优化等级为 4），后续帧推理吞吐率显著提升；适用于追求极致性能的批量稳定生产。

### 并发执行线程数 (Execution Thread Count)

```text
UI: EXECUTION THREAD COUNT | CLI: --execution-thread-count | INI: [execution] execution_thread_count
Range: 1 - 32
Step: 1
Default: 8
```

分配给处理器的并行推理工作线程数。显存容量与并发线程必须平衡：16GB 及以上显存建议设置为 `8` ~ `16`；6GB ~ 8GB 显存建议设置为 `2` ~ `4`，防止多线程同时分配张量导致 CUDA 显存溢出（OOM，Out of Memory）。

### 显存回收与驻留策略 (Video Memory Strategy)

```text
UI: VIDEO MEMORY STRATEGY | CLI: --video-memory-strategy | INI: [memory] video_memory_strategy
Choices: strict, moderate, tolerant
Default: strict
```

| 策略 | 内存释放行为 | 适用硬件环境 |
| :--- | :--- | :--- |
| `strict` | 处理完毕后清空当前处理器及底层公共模块推理池 | 显存较小设备（如 <= 6GB） |
| `moderate` | 保留公共人脸检测、定位等核心模型常驻显存，释放处理器推理池 | 显存适中设备（如 8GB ~ 12GB） |
| `tolerant` | 全部模型与张量完全常驻显存，不执行清理。推理吞吐速度最快 | 显存充裕设备（如 >= 16GB） |

### 日志输出级别 (Log Level)

```text
UI: LOG LEVEL | CLI: --log-level | INI: [misc] log_level
Choices: error, warn, info, debug
Default: info
```

定义控制台与日志系统的过滤输出级别：
- `error`：仅输出阻断流程的严重异常与显存溢出报错。
- `warn`：输出潜在风险告警（如模型未加速、偶发坏帧）。
- `info`：输出初始化模型信息、任务进度百分比与处理耗时。
- `debug`：输出底层完整调试流（包含每一帧检测坐标、ONNX Runtime 会话构建、FFmpeg 完整转码参数行等），用于深度性能调优与故障排查。
