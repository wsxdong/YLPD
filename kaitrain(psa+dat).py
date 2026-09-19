import torch
import time
import os
from ultralytics import YOLO
import torch.nn.functional as F

# ===================== 1. 完整定义PSA+DAT模块 =====================
class PolarizedSelfAttention(torch.nn.Module):
    """完整PSA模块（极化自注意力）"""
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        mid_channels = channels // reduction

        # 横向注意力（宽度维度）
        self.query_horizontal = torch.nn.Conv2d(channels, mid_channels, 1, padding=0)
        self.key_horizontal = torch.nn.Conv2d(channels, mid_channels, 1, padding=0)
        self.value_horizontal = torch.nn.Conv2d(channels, channels, 1, padding=0)

        # 纵向注意力（高度维度）
        self.query_vertical = torch.nn.Conv2d(channels, mid_channels, 1, padding=0)
        self.key_vertical = torch.nn.Conv2d(channels, mid_channels, 1, padding=0)
        self.value_vertical = torch.nn.Conv2d(channels, channels, 1, padding=0)

        self.gamma = torch.nn.Parameter(torch.zeros(1))  # 注意力融合系数

    def forward(self, x):
        b, c, h, w = x.size()

        # 横向注意力计算
        q_h = self.query_horizontal(x).view(b, -1, h * w)
        k_h = self.key_horizontal(x).view(b, -1, h * w)
        v_h = self.value_horizontal(x).view(b, -1, h * w)
        attn_h = F.softmax(torch.bmm(q_h.transpose(1, 2), k_h), dim=-1)
        out_h = torch.bmm(v_h, attn_h.transpose(1, 2)).view(b, c, h, w)

        # 纵向注意力计算
        q_v = self.query_vertical(x).view(b, -1, h * w)
        k_v = self.key_vertical(x).view(b, -1, h * w)
        v_v = self.value_vertical(x).view(b, -1, h * w)
        attn_v = F.softmax(torch.bmm(q_v, k_v.transpose(1, 2)), dim=-1)
        out_v = torch.bmm(v_v.transpose(1, 2), attn_v).transpose(1, 2).view(b, c, h, w)

        # 融合+残差
        out = self.gamma * (out_h + out_v) + x
        print("✅ PSA模块参与训练！")  # 验证执行
        return out

class DeformableAttentionTransformer(torch.nn.Module):
    """完整DAT模块（可变形注意力）"""
    def __init__(self, embed_dim, num_heads=8, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # 可变形注意力核心层
        self.offset = torch.nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weight = torch.nn.Linear(embed_dim, num_heads * num_points)
        self.value_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

        # 初始化参数
        torch.nn.init.constant_(self.offset.weight, 0.)
        torch.nn.init.constant_(self.offset.bias, 0.)

    def forward(self, x):
        b, c, h, w = x.size()
        # 适配2D特征到注意力输入格式 [B, H*W, C]
        feat = x.reshape(b, c, h*w).transpose(1, 2)
        B, N, C = feat.shape

        # 价值映射
        value = self.value_proj(feat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # 预测偏移量
        offset = self.offset(feat).reshape(B, N, self.num_heads, self.num_points, 2)
        offset = torch.tanh(offset)
        # 预测注意力权重
        attn_weight = self.attention_weight(feat).reshape(B, N, self.num_heads, self.num_points)
        attn_weight = F.softmax(attn_weight, dim=-1)

        # 生成参考网格
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid = torch.stack([grid_x, grid_y], dim=-1).float().to(x.device)
        grid = grid.reshape(1, N, 1, 1, 2).repeat(B, 1, self.num_heads, self.num_points, 1)
        # 可变形采样
        sample_coords = grid + offset
        sample_coords = sample_coords / torch.tensor([w-1, h-1], device=x.device) * 2 - 1

        # 采样特征（简化版，降低显存占用）
        v = value.reshape(B*self.num_heads, self.head_dim, h, w)
        coords = sample_coords.reshape(B*self.num_heads, N*self.num_points, 2)
        v_sampled = F.grid_sample(v, coords.unsqueeze(1), mode='bilinear', padding_mode='zeros', align_corners=False)
        v_sampled = v_sampled.reshape(B, self.num_heads, self.head_dim, N, self.num_points).permute(0, 3, 1, 4, 2)
        # 注意力加权
        attn_output = (v_sampled * attn_weight.unsqueeze(-1)).sum(dim=3)
        attn_output = attn_output.reshape(B, N, C)
        out = self.out_proj(attn_output).transpose(1, 2).reshape(b, c, h, w)

        print("✅ DAT模块参与训练！")  # 验证执行
        return out

# ===================== 2. 核心训练逻辑（关闭早停+指定轮数） =====================
if __name__ == "__main__":
    # ========== 基础配置（仅需修改这部分） ==========
    DATA_YAML = r"D:\pycharm\YOLOv10\kaiyuandate\data.yaml"  # 你的kaiyuandata路径
    EPOCHS = 50  # 🔴 改为你想要的训练轮数（比如50轮，可自行调整）
    IMG_SIZE = 640
    BATCH_SIZE = 4  # 降低批次避免显存不足
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ==============================================

    # 校验数据集路径
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"❌ 数据集配置文件不存在：{DATA_YAML}")
    print(f"✅ 加载数据集：{DATA_YAML}")

    # 1. 加载YOLOv10预训练模型
    model = YOLO("yolov10n.pt")
    # 清空显存（避免OOM）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. 初始化双注意力模块（适配模型设备+通道数）
    # 取YOLOv10第一层输出通道作为注意力模块输入通道
    first_layer_out_ch = model.model.model[0].conv.out_channels
    psa = PolarizedSelfAttention(channels=first_layer_out_ch).to(DEVICE)
    dat = DeformableAttentionTransformer(embed_dim=first_layer_out_ch).to(DEVICE)

    # 3. 覆盖模型前向函数（强制插入PSA+DAT）
    original_forward = model.model.forward
    def custom_forward(x, *args, **kwargs):
        # 前向流程：原生层 → PSA → 原生层 → DAT → 剩余原生层
        x = model.model.model[0](x)  # 第一层
        x = psa(x)                   # 插入PSA
        x = model.model.model[1](x)  # 第二层
        x = dat(x)                   # 插入DAT
        for layer in model.model.model[2:]:  # 剩余层
            x = layer(x)
        return x
    model.model.forward = custom_forward  # 替换原生前向

    # 4. 训练参数（核心改动：关闭早停+固定轮数）
    # 生成唯一实验名（避免覆盖）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    train_params = {
        "data": DATA_YAML,
        "epochs": EPOCHS,          # 强制训练指定轮数
        "imgsz": IMG_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        "workers": 0,              # Windows系统设为0，避免多进程报错
        "lr0": 0.01,               # 初始学习率
        "lrf": 0.01,               # 学习率衰减
        "cos_lr": True,
        "verbose": True,
        # 保存配置（唯一路径，避免覆盖）
        "project": "runs/psa_dat_kaiyuandata",  # 独立目录
        "name": f"yolov10_psa_dat_{EPOCHS}epochs_{timestamp}",
        "exist_ok": True,
        # 🔴 关键改动1：删除patience参数（关闭早停），Ultralytics YOLO中patience控制早停，设为0也可
        # "patience": 5,  # 注释/删除这一行，彻底关闭早停
        "patience": 0,             # 备选方案：设为0也能关闭早停（兼容旧版本）
        # 显存优化
        "mosaic": 0.0,             # 关闭马赛克增强
        "save": True
    }

    # 5. 启动训练（强制训练EPOCHS轮，不会提前停止）
    print(f"\n🚀 开始训练YOLOv10+PSA+DAT（{DEVICE}），共{EPOCHS}轮（已关闭早停）...")
    results = model.train(**train_params)

    # 6. 输出训练结果
    print(f"\n🎉 {EPOCHS}轮训练完成！")
    print(f"📊 最终mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"📁 最佳权重保存路径: {results.save_dir}/weights/best.pt")