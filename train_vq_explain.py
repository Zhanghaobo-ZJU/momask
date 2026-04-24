"""
训练 RVQVAE (Residual Vector Quantized Variational AutoEncoder) 模型的脚本
用于将人体运动序列压缩离散化为 motion tokens，以便后续 Text-to-Motion 任务使用

RVQVAE 核心思想：
1. 编码器将高维运动数据压缩为潜在向量
2. VQ (Vector Quantization) 将连续潜在向量映射到离散 codebook
3. 解码器从离散 tokens 重构运动序列
"""

import os

# os.path.join 的别名，用于简化路径拼接操作
from os.path import join as pjoin

import torch

# PyTorch 核心库，提供张量运算和自动微分功能
from torch.utils.data import DataLoader
# PyTorch 数据加载工具，支持批量加载、数据预处理、多进程加载

# =============================================
# 自定义模块导入
# =============================================

# RVQVAE 模型定义，包含编码器、VQ层、解码器
from models.vq.model import RVQVAE

# RVQVAE 的训练器类，负责训练循环、损失计算、模型保存等
from models.vq.vq_trainer import RVQTokenizerTrainer

# 命令行参数解析模块
from options.vq_option import arg_parse

# 运动数据集类，用于加载和预处理 HumanML3D 或 KIT-ML 数据集
from data.t2m_dataset import MotionDataset

# 工具函数模块，包含人体运动相关的常数和工具
from utils import paramUtil
import numpy as np

# 评估模型封装，用于在训练过程中评估生成运动的质量
from models.t2m_eval_wrapper import EvaluatorModelWrapper

# 读取预训练选项配置的函数
from utils.get_opt import get_opt

# 获取数据集运动加载器的函数
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

# 从 rotation-invariant coordinates 恢复 3D 人体姿态
# HumanML3D 使用 SMPL 模型，其关节数据是相对旋转编码的，需要转换为 3D 坐标
from utils.motion_process import recover_from_ric

# 3D 人体运动可视化工具，将关节序列渲染为 MP4 视频
from utils.plot_script import plot_3d_motion

# 固定随机种子的工具，确保实验可复现
from utils.fixseed import fixseed

# 设置 OpenMP 线程数为 1，避免多核并行导致的问题
os.environ["OMP_NUM_THREADS"] = "1"


def plot_t2m(data, save_dir):
    """
    将运动数据可视化为 3D 动画视频

    参数:
        data: 原始运动数据 (numpy 数组或 torch 张量)
        save_dir: 保存视频的目录路径

    处理流程:
        1. 对数据进行反标准化 (乘以标准差加上均值)
        2. 将每个样本从 rotation-invariant coordinates 转换为 3D 坐标
        3. 调用 plot_3d_motion 生成 MP4 视频
    """
    # 反标准化：将标准化后的数据恢复到原始尺度
    # train_dataset 包含 mean 和 std 信息，用于反标准化
    data = train_dataset.inv_transform(data)

    # 遍历每个样本，分别生成可视化
    for i in range(len(data)):
        # 获取单个样本的关节数据
        # data[i] 的形状通常是 (joints_num * 3, frames) 或类似结构
        joint_data = data[i]

        # 将 rotation-invariant coordinates 转换为 3D 直角坐标
        # recover_from_ric 函数的输入是 SMPL 的旋转参数，输出是 3D 空间坐标
        # 参数: joint_data (nparray), opt.joints_num (人体关节数)
        # 返回: joint (numpy array)，形状为 (joints_num, 3, frames) 的 3D 坐标
        joint = recover_from_ric(
            torch.from_numpy(joint_data).float(), opt.joints_num
        ).numpy()

        # 构建输出文件路径，格式为 00.mp4, 01.mp4, ...
        save_path = pjoin(save_dir, "%02d.mp4" % (i))

        # 绘制 3D 人体运动动画并保存为 MP4
        # kinematic_chain: 人体骨骼链连接关系，用于正确连接各个关节
        # joint: 3D 关节坐标，形状 (joints_num, 3, frames)
        # fps: 每秒帧数，影响视频播放速度
        # radius: 相机半径，影响视角距离
        plot_3d_motion(
            save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius
        )


if __name__ == "__main__":
    """
    主程序入口

    训练流程概述:
        1. 解析命令行参数
        2. 配置模型保存路径
        3. 根据数据集类型配置参数
        4. 加载评估模型和数据标准化参数
        5. 初始化 RVQVAE 模型
        6. 创建训练器和数据加载器
        7. 执行训练循环
    """

    # 启用 PyTorch 的异常检测（用于调试梯度问题，默认关闭以提高性能）
    # torch.autograd.set_detect_anomaly(True)

    # =============================================
    # 第1步：解析命令行参数
    # =============================================
    # arg_parse(True) 解析用户传入的命令行参数，返回配置对象 opt
    # 参数 True 表示这是训练模式（区别于测试/推理模式）
    opt = arg_parse(True)

    # 固定随机种子，确保以下组件的行为可复现：
    # - Python random
    # - NumPy random
    # - PyTorch random
    # - CUDA 随机性（如果使用 GPU）
    fixseed(opt.seed)

    # =============================================
    # 第2步：配置计算设备
    # =============================================
    # 根据 gpu_id 判断使用 CPU 还是 GPU
    # opt.gpu_id == -1 表示使用 CPU
    # 否则使用 "cuda:0", "cuda:1" 等指定 GPU
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    # =============================================
    # 第3步：配置模型保存路径
    # =============================================
    # 所有模型相关文件都保存在 checkpints_dir/dataset_name/experiment_name/ 下

    # 根目录：checkpoints/dataset_name/experiment_name/
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)

    # 模型权重保存目录：save_root/model/
    # 包含每个 epoch 的 checkpoint 和最终模型
    opt.model_dir = pjoin(opt.save_root, "model")

    # 元数据保存目录：save_root/meta/
    # 包含训练日志、loss 曲线等元数据
    opt.meta_dir = pjoin(opt.save_root, "meta")

    # 可视化动画保存目录：save_root/animation/
    # 保存验证集上生成的动作可视化
    opt.eval_dir = pjoin(opt.save_root, "animation")

    # 日志文件保存目录：./log/vq/dataset_name/experiment_name/
    opt.log_dir = pjoin("./log/vq/", opt.dataset_name, opt.name)

    # 创建所有必要的目录（如果已存在则不报错）
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # =============================================
    # 第4步：根据数据集类型配置参数
    # =============================================
    # 支持两种数据集：
    # 1. "t2m" - HumanML3D 数据集，22个关节，263维特征，20fps
    # 2. "kit" - KIT-ML 数据集，21个关节，251维特征，12.5fps

    if opt.dataset_name == "t2m":
        # HumanML3D 数据集路径配置
        opt.data_root = "./dataset/HumanML3D/"
        # 预处理后的关节向量文件目录
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        # 文本描述文件目录
        opt.text_dir = pjoin(opt.data_root, "texts")
        # 人体关节数量（SMPL 模型有 22 个关节）
        opt.joints_num = 22
        # 每个关节向量的维度（包含位置、旋转等信息）
        dim_pose = 263
        # 帧率：每秒 20 帧
        fps = 20
        # 3D 可视化的相机半径
        radius = 4
        # 人体运动学链，定义关节之间的连接关系
        kinematic_chain = paramUtil.t2m_kinematic_chain
        # 预训练评估模型的配置文件路径
        dataset_opt_path = "./checkpoints/t2m/Comp_v6_KLD005/opt.txt"

    elif opt.dataset_name == "kit":
        # KIT-ML 数据集路径配置
        opt.data_root = "./dataset/KIT-ML/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 21
        # KIT 的空间尺度不同，需要更大的相机半径
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        # KIT 数据集的最大运动长度（帧数）
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
        dataset_opt_path = "./checkpoints/kit/Comp_v6_KLD005/opt.txt"

    else:
        # 不支持的数据集类型，抛出异常
        raise KeyError("Dataset Does not Exists")

    # =============================================
    # 第5步：加载评估模型
    # =============================================
    # EvaluatorModelWrapper 是一个预训练的 T2M 评估模型
    # 用于在训练过程中评估生成的运动质量（FID、div等指标）
    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # =============================================
    # 第6步：加载数据标准化参数
    # =============================================
    # HumanML3D/KIT-ML 数据集在预处理时进行了标准化
    # Mean.npy: 各维度的均值向量
    # Std.npy: 各维度的标准差向量
    # 反标准化时需要使用这些参数
    mean = np.load(pjoin(opt.data_root, "Mean.npy"))
    std = np.load(pjoin(opt.data_root, "Std.npy"))

    # =============================================
    # 第7步：加载训练/验证集划分文件
    # =============================================
    # train.txt 和 val.txt 包含训练/验证样本的文件名列表
    train_split_file = pjoin(opt.data_root, "train.txt")
    val_split_file = pjoin(opt.data_root, "val.txt")

    # =============================================
    # 第8步：初始化 RVQVAE 模型
    # =============================================
    # RVQVAE = Residual Vector Quantized Variational AutoEncoder
    # 核心参数说明：
    #   - opt: 包含所有超参数的配置对象
    #   - dim_pose: 关节向量维度（263 for t2m, 251 for kit）
    #   - opt.nb_code: codebook 中离散 code 的数量（VQ tokens 数量）
    #   - opt.code_dim: 每个 code 的嵌入维度
    #   - opt.down_t: 时间维度的下采样层数
    #   - opt.stride_t: 时间维度下采样的步长
    #   - opt.width: 模型的隐藏层维度
    #   - opt.depth: 编码器/解码器的层数
    #   - opt.dilation_growth_rate: 膨胀卷积的增长速率
    #   - opt.vq_act: VQ 层的激活函数类型
    #   - opt.vq_norm: VQ 层是否使用 LayerNorm
    net = RVQVAE(
        opt,
        dim_pose,
        opt.nb_code,  # codebook 大小（离散 token 数量）
        opt.code_dim,  # 每个 token 的嵌入维度
        opt.code_dim,  # decoder 的嵌入维度（与 encoder 相同）
        opt.down_t,  # 时间维度下采样次数
        opt.stride_t,  # 下采样步长
        opt.width,  # 隐藏层宽度
        opt.depth,  # 网络深度
        opt.dilation_growth_rate,  # 膨胀卷积增长率
        opt.vq_act,  # VQ激活函数
        opt.vq_norm,  # VQ是否归一化
    )

    # 计算模型的总参数量
    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)  # 打印模型结构详情

    # 打印总参数量（以 M 为单位，1M = 100万参数）
    print("Total parameters of all models: {}M".format(pc_vq / 1000_000))

    # =============================================
    # 第9步：初始化训练器
    # =============================================
    # RVQTokenizerTrainer 负责：
    #   - 定义训练循环
    #   - 计算损失函数（重构损失 + VQ 损失 + 承诺损失）
    #   - 执行验证和评估
    #   - 保存模型 checkpoint
    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    # =============================================
    # 第10步：创建数据集和数据加载器
    # =============================================
    # MotionDataset: 负责加载和预处理运动数据
    # 参数:
    #   - opt: 配置对象
    #   - mean/std: 数据标准化参数
    #   - split_file: 样本列表文件路径
    train_dataset = MotionDataset(opt, mean, std, train_split_file)
    val_dataset = MotionDataset(opt, mean, std, val_split_file)

    # DataLoader: 批量加载数据，支持多进程预处理
    # 参数说明:
    #   - batch_size: 每个 batch 的样本数
    #   - drop_last: 如果最后一批样本数小于 batch_size，是否丢弃
    #   - num_workers: 数据加载的子进程数（4 表示使用 4 个进程并行加载）
    #   - shuffle: 是否在每个 epoch 开始时打乱数据
    #   - pin_memory: 是否将数据加载到 pinned memory，加速数据传输到 GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        drop_last=True,  # 丢弃最后不完整的 batch
        num_workers=4,  # 4 个进程并行加载
        shuffle=True,  # 打乱数据顺序
        pin_memory=True,  # 加速 GPU 数据传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    # =============================================
    # 第11步：获取验证集的评估数据加载器
    # =============================================
    # 与上面的 val_loader 不同，eval_val_loader 是用于评估模型的专用加载器
    # 参数 'val' 表示加载验证集
    # 32 是 batch_size
    # device 指定数据所在的设备
    eval_val_loader, _ = get_dataset_motion_loader(
        dataset_opt_path, 32, "val", device=opt.device
    )

    # =============================================
    # 第12步：开始训练
    # =============================================
    # train() 方法的参数：
    #   - train_loader: 训练数据加载器
    #   - val_loader: 验证数据加载器
    #   - eval_val_loader: 用于评估的验证数据加载器
    #   - eval_wrapper: 预训练的评估模型
    #   - plot_t2m: 可视化回调函数
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, plot_t2m)


# =============================================
# 训练命令示例
# =============================================
# 以下是一些常用的训练命令示例

# KIT 数据集，batch_size=512，单卡 V100 (gpu_id=3)
# python train_vq.py --dataset_name kit --batch_size 512 --name VQVAE_dp2 --gpu_id 3

# KIT 数据集，batch_size=256，单卡 (gpu_id=2)
# python train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp2_b256 --gpu_id 2

# KIT 数据集，batch_size=1024，单卡 (gpu_id=1)
# python train_vq.py --dataset_name kit --batch_size 1024 --name VQVAE_dp2_b1024 --gpu_id 1

# KIT 数据集，batch_size=256，使用不同的下采样配置 (gpu_id=2)
# python train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp1_b256 --gpu_id 2
