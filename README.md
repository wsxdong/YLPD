# YLPDYOLOv10-PSA-DAT-Fall-Detection
Fall Detection

人体跌倒检测是构建智能化校园安防体系、保障学生运动安全的关键技术之一。在大学生体育教学与日常活动中，运动损伤具有突发性与高发性，田径、球类、体操等多样化体育项目带来的剧烈姿态变化，以及室外强光、室内逆光、人群遮挡等复杂环境因素，对实时监测提出了严峻挑战。传统基于穿戴设备的检测方法存在传感器易脱落、误报率高、部署成本大等问题，难以满足公共体育场景对普适性、低成本与非侵入性的综合需求。

本文提出一种融合极化自注意力（PSA）与可变形注意力Transformer（DAT）的轻量化跌倒检测模型YOLOv10-PSA-DAT。该模型以YOLOv10为基线，通过在骨干网络中嵌入PSA模块增强多尺度特征捕获能力，并在颈部结构集成DAT模块以提升遮挡与姿态变形场景下的特征判别能力，有效解决了现有方法在计算效率与复杂场景鲁棒性之间难以平衡的核心问题。实验表明，该模型在自建数据集上取得99.40%的准确率和2.4%的误报率，对于提升大学体育教学安全保障水平具有重要参考意义。

YOLOv10-PSA-DAT 核心创新点

（1）极化自注意力（PSA）模块：针对复杂体育场景下的多尺度特征捕获需求，在骨干网络中嵌入PSA模块。该模块将特征分解为通道极化与空间极化分支，通道分支关注跌倒目标的关键语义特征，空间分支增强目标位置与尺度信息。在保持特征提取能力的同时，显著增强了模型对不同尺度跌倒目标的识别能力，且未引入过大计算开销。
（2）可变形注意力Transformer（DAT）模块：为增强模型在遮挡与姿态变形场景下的特征判别力，在网络颈部结构引入DAT模块。该模块通过学习参考点偏移和双线性插值采样，动态聚焦于被遮挡或姿态变形的跌倒肢体细节，跨尺度聚合多层级特征，显著提升了模型对复杂场景的适应能力。
（3）双模块协同机制：通过PSA与DAT的协同设计，在保持高推理速度的前提下显著增强了对复杂动态环境的适应能力。PSA模块强化局部多尺度特征提取，DAT模块提升全局上下文与跨尺度建模能力，二者互补实现了性能的全面提升。

效率与精度平衡：相较于经典YOLOv10基线模型，提出的方法在保持较高推理速度的同时，检测精度显著提升。同时，通过精简C2f模块通道数（减半）与移除一对多辅助检测头（仅保留一对一头）的轻量化设计，模型参数量大幅降低，满足嵌入式终端的部署需求。

实验数据集：大学生跌倒检测数据集 3.1 数据集概况 本研究基于自建大学体育场景跌倒检测数据集，包含三种大学生体育课跌倒状态，数据集存储于百度网盘，需自行下载后使用：

数据集名称	包含类别	图像总数	图像分辨率	数据分布（训练:验证:测试）
Our fall detection	Standing、Fall	4213	统一resize至适配模型输入	7:2:1
3.2 数据集获取与结构
为进一步增强模型可比性与泛化能力，本研究融合了两个公开摔倒检测数据集作为自建数据的补充：

LE2I with Upright and Fall：基于原始 LE2I 基准构建，包含日常活动与突发摔倒等多样室内场景（https://universe.roboflow.com/new-workspace-qfcus/le2i-with-upright-and-fall/dataset/2）。

Falldown Detection F8XTB：以体育场、运动场等室外场景的标注网络图像为主（https://universe.roboflow.com/kid-g8rt3/falldown-detection-f8xtb/dataset/1）。
文件夹组织（下载后解压至项目根目录，结构如下）：

our_fall_detection/
├── images/
│ ├── train/ # 训练集图像（包含跑道、篮球场、室内场景图像）
│ ├── val/ # 验证集图像（同上）
│ └── test/ # 测试集图像（同上）
└── labels/
├── train/ # 训练集标注文件（YOLO格式txt，与图像一一对应）
├── val/ # 验证集标注文件
└── test/ # 测试集标注文件

实验环境配置

4.1 依赖安装 推荐使用Anaconda创建虚拟环境，确保依赖版本匹配：

1. 创建并激活虚拟环境
conda create -n yolov10-psa-dat python=3.12.4
conda activate yolov10-psa-dat

2. 安装PyTorch、TorchVision（需适配CUDA版本，示例为CUDA 11.8；CPU用户可替换为cpu版本）
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

3. 安装其他核心依赖库（数据处理、可视化、模型工具等）
pip install numpy pandas scipy matplotlib seaborn opencv-python pillow tqdm thop

4. 安装YOLOv10所需的其他依赖
请根据你实际使用的YOLOv10代码库中的requirements.txt进行安装。
需要注意的是，如果您需要使用FlashAttention加速，其版本需要与PyTorch和CUDA严格匹配。可参考原教程，下载与 torch 2.2.2+cu118 兼容的 flash_attn wheel 文件进行本地安装。

4.2 硬件要求 GPU：推荐显存≥8GB的NVIDIA GPU（如RTX 4060，CUDA≥11.7）。训练约200轮约3-4小时，显存占用≤4GB。
CPU：单帧推理约0.3-0.5秒，不推荐完整训练。

实验结果 5.1 核心指标对比 本文提出的YOLOv10-PSA-DAT与基线YOLOv10及其变体在自建大学体育场景跌倒检测数据集上的性能对比如下。结果表明，双模块集成模型在检测精度上显著提升，同时保持了轻量化与实时性优势。

模型	mAP@50 (%)	mAP@[0.5:0.95] (%)	参数量 (M)	推理速度 (ms)
YOLOv10n (基线)	97.2	89.7	2.5	17.7
+ PSA	98.5	92.8	2.5	17.9
+ DAT	98.8	91.3	2.5	18.1
YOLOv10-PSA-DAT (本文)	99.40	94.4	2.5	17.5
注：

mAP@50 指在 IoU 阈值为 0.5 时的平均精度均值，用于衡量模型的基础检测性能。数值越高，模型对目标位置和类别的识别能力越强。

mAP@[0.5:0.95] 指在 IoU 阈值从 0.5 到 0.95（步长 0.05）下的平均 mAP，用于衡量模型的定位鲁棒性。数值越高，模型在不同重叠度要求下的表现越稳定。

参数量 (M) 指模型可训练参数的总量（以百万为单位），用于衡量模型的轻量化程度。数值越小，模型体积越小、占用内存越少，越适合边缘设备部署。

推理速度 (ms) 指模型在 NVIDIA GeForce RTX 4060 Laptop GPU 上处理单帧图像的平均耗时，用于衡量模型的响应效率。

代码使用说明 6.1 模型训练 运行 train.py 脚本启动训练。脚本会自动加载基础 YOLOv10 模型，并集成 PSA 和 DAT 模块。示例命令（适配跌倒检测数据集）：

bash
python train.py \
--data ./our_fall_detection \
--epochs 200 \
--batch 32 \
--lr0 0.001 \
--optimizer SGD \
--save_dir ./runs/yolov10_PSA_DAT \
--device cuda:0
关键参数说明：

参数名	含义	默认值
--data	数据集根目录路径	./our_fall_detection
--epochs	训练轮数	200
--batch	批次大小（根据显存调整）	32
--lr0	初始学习率	0.001
--optimizer	优化器（SGD / Adam / AdamW）	SGD
--save_dir	模型保存目录	./runs/exp
--device	训练设备（cuda:0 或 cpu）	cuda:0
训练输出：训练过程中，每25轮会自动保存一次模型，训练结束后最佳模型将保存为 best.pt，位于 save_dir/train/weights/ 下。

6.2 模型预测 使用训练好的权重进行单张图像预测，运行 predict.py 脚本，示例命令：

bash
python predict.py \
--image_path ./images/test/basketball146.jpg \
--model_path ./weights/best.pt \
--device cuda:0
预测输出示例：
📊 PERFORMANCE METRICS RESULTS: • Parameters: 2.5 M • FLOPs: 7.6 G (estimated) 测速中... • Inference Speed: 17.5 ms per image (batch=1, cuda:0)
🎯 PREDICTION RESULT: Most likely class: standing Probability: 0.894

6.3 预训练权重 提供基于跌倒检测数据集训练完成的最优权重，可直接用于预测或微调。

适用场景：针对跌倒检测场景中的“站立(standing)”和“跌倒(fall)”两类分类。若需扩展其他动作类别或类似场景，建议基于此权重微调（冻结部分主干，仅训练检测头，可显著减少训练数据量）。

项目文件结构 YOLOv10-PSA-DAT-Fall-Detection/
├── data/ # 数据集目录（可按需软链接或存放）
│ └── our_fall_detection/ # 跌倒检测数据集（原始结构）
│ ├── images/ # 图像文件夹
│ │ ├── train/ # 训练集图像
│ │ ├── val/ # 验证集图像
│ │ └── test/ # 测试集图像
│ └── labels/ # 标签文件夹（YOLO格式txt）
│ ├── train/
│ ├── val/
│ └── test/
├── models/ # 整体模型实现
│ ├── yolo_PSA.py # PSA模块实现
│ ├── yolo_DAT.py # DAT模块实现
│ └── yolo_PSA_DAT.py # 主模型（整合上述模块）
├── weights/ # 预训练权重存放
│ └── best.pt # 最优模型权重
├── train.py # 训练脚本（集成PSA/DAT，含最佳训练配置）
├── predict.py # 预测与性能评估脚本（输出参数量、FLOPs、速度、预测结果）
├── fall_detection.yaml # 数据集配置文件（指向data/our_fall_detection）
├── requirements.txt # 依赖包列表
└── README.md # 项目说明文档

已知问题与注意事项 数据集适配：当前模型与权重仅针对“站立(standing)”和“跌倒(fall)”两类场景。若需扩展其他动作类别（如行走、跑步等）或类似监控场景，需补充对应数据集并基于本权重微调。
CUDA版本问题：若安装PyTorch时出现CUDA不兼容，可替换为CPU版本（需将所有脚本的 --device 改为 cpu），但训练和推理效率会大幅下降；显存占用：训练时若显存不足（如8GB GPU），建议将 batch 设为8，并通过梯度累积（accumulate=2）等效模拟batch=16，或适当降低 imgsz。

