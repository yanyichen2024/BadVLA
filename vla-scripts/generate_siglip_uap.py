"""
generate_siglip_uap.py

针对 OpenVLA 的 Vision Backbone (SigLIP) 生成通用对抗扰动 (UAP)。
1. 从 LIBERO 环境中采集真实观测图像。
2. 加载 OpenVLA 模型并提取 Vision Backbone。
3. 优化一个通用噪声 delta，使得特征提取的失真度最大化。
"""

import os
import sys
import torch
import numpy as np
import tqdm
from PIL import Image
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 添加路径以导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../experiments/robot/libero")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../LIBERO")))

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image
# from prismatic.models.load import load_vla # Deprecated for this use case
from prismatic.models.load import load as prism_load
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from experiments.robot.openvla_utils import resize_image_for_policy, center_crop_image

# ================= 配置参数 =================
DEBUG = False 
SAVE_PATH = "siglip_uap_noise.pt"
NUM_COLLECT_IMAGES = 100      # 采集用于优化的图片数量
IMG_SIZE = 224                 # SigLIP 输入尺寸
BATCH_SIZE = 10
NUM_ITERATIONS = 50           # 优化迭代次数
LEARNING_RATE = 1e-2
EPSILON = 16 / 255.0           # 扰动限制 (L_inf norm), 约等于像素值 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 1. 数据采集 =================
def collect_data(task_suite_name="libero_goal"):
    """从 LIBERO 环境中采集观测图像，并返回任务描述"""
    print(f"[*] 正在从 {task_suite_name} 采集 {NUM_COLLECT_IMAGES} 张图像...")

    # 输出样本保存路径
    artifacts_dir = os.path.join(os.path.dirname(__file__), "uap_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    sample_dir = os.path.join(artifacts_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    sample_save_limit = 16  # 仅保存前 16 张用于检查
    
    # 按照 run_libero_eval.py 的逻辑初始化任务
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task_id = 0
    task = task_suite.get_task(task_id)

    # 初始化环境
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    env.reset()
    
    collected_images = []
    
    # 动作空间维度通常是 7 (x,y,z, roll, pitch, yaw, gripper)
    # 使用 numpy 生成随机动作
    
    # 简单的随机策略跑几个 step 拿图片
    for _ in range(NUM_COLLECT_IMAGES):
        # 简单随机动作 [-1, 1]
        action = np.random.uniform(-1, 1, size=(7,))
        obs, _, _, _ = env.step(action)
        
        # 获取图像 -> 复用推理侧 resize + center crop，确保分布一致
        img = get_libero_image(obs)  # numpy uint8
        img_resized = resize_image_for_policy(img, IMG_SIZE)  # TF lanczos3 + round，与推理一致
        img_cropped = center_crop_image(img_resized)  # 0.9 center crop，与推理默认配置一致

        # 转为 Tensor: [C, H, W], 0-1 float
        img_tensor = torch.from_numpy(np.array(img_cropped)).permute(2, 0, 1).float() / 255.0
        collected_images.append(img_tensor)

        # 保存前几张样本以便检查采集内容
        if len(collected_images) <= sample_save_limit:
            sample_path = os.path.join(sample_dir, f"sample_{len(collected_images):03d}.png")
            img_cropped.save(sample_path)
        
        # 简单的自动重置逻辑
        if len(collected_images) % 20 == 0:
            env.reset()

    env.close()
    
    # Stack into [B, C, H, W]
    data_tensor = torch.stack(collected_images).to(DEVICE)
    print(f"[+] 数据采集完成，Tensor shape: {data_tensor.shape}")
    return data_tensor, task_description

# ================= 2. 噪声优化 =================
def optimize_uap(model, processor, data_loader_tensor, task_description: str):
    """
    针对完整 VLM (Prismatic Dino+SigLIP) 优化 UAP 噪声，使其倾向输出停止 (EOS)
    Args:
        model: OpenVLA model
        data_loader_tensor: [B, 3, 224, 224] 图像数据
    """
    # 设置模型为评估模式，但解冻关键部分以允许梯度传播
    model.eval()
    if hasattr(model, "vision_backbone"):
        model.vision_backbone.requires_grad_(True)
        # 使 forward 中的 set_grad_enabled 开启，以便梯度穿过视觉分支
        if hasattr(model, "vision_backbone_requires_grad"):
            model.vision_backbone_requires_grad = True
    if hasattr(model, "projector") and model.projector is not None:
        model.projector.requires_grad_(True)

    tokenizer = model.llm_backbone.tokenizer
    eos_id = tokenizer.eos_token_id

    # 初始化噪声 delta (可学习参数)
    # 形状 [1, 3, 224, 224]
    delta = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    delta.requires_grad = True
    
    # 优化器只更新 delta，模型权重不变
    optimizer = Adam([delta], lr=LEARNING_RATE)
    
    print(f"[*] 开始优化对抗噪声 (Iterations: {NUM_ITERATIONS})...")
    
    pbar = tqdm.tqdm(range(NUM_ITERATIONS))
    loss_history = []
    for i in pbar:
        # 确保模型相关部分保持评估模式（例如，关闭 Dropout）
        # 但梯度计算是开启的
        model.eval()
        optimizer.zero_grad()
        
        # 随机采样一个 Batch
        idx = torch.randperm(data_loader_tensor.shape[0])[:BATCH_SIZE]
        batch_imgs = data_loader_tensor[idx] # [Batch, 3, H, W] in [0, 1]
        
        # 添加噪声并截断 (像素空间 PGD)
        adv_imgs = torch.clamp(batch_imgs + delta, 0.0, 1.0)

        # 准备 VLM 输入：直接在 Tensor 上做归一化以保持梯度
        target_dtype = next(model.vision_backbone.parameters()).dtype
        if hasattr(model.vision_backbone, "dino_data_cfg") and hasattr(model.vision_backbone, "siglip_data_cfg"):
            dino_mean = torch.tensor(model.vision_backbone.dino_data_cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
            dino_std = torch.tensor(model.vision_backbone.dino_data_cfg["std"], device=DEVICE).view(1, 3, 1, 1)
            siglip_mean = torch.tensor(model.vision_backbone.siglip_data_cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
            siglip_std = torch.tensor(model.vision_backbone.siglip_data_cfg["std"], device=DEVICE).view(1, 3, 1, 1)

            pixel_values = {
                "dino": ((adv_imgs - dino_mean) / dino_std).to(dtype=target_dtype),
                "siglip": ((adv_imgs - siglip_mean) / siglip_std).to(dtype=target_dtype),
            }
        else:
            # 单分支模型回退：假设 mean/std=0.5/0.5
            mean = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=DEVICE).view(1, 3, 1, 1)
            pixel_values = ((adv_imgs - mean) / std).to(dtype=target_dtype)

        prompts = [
            f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        ] * adv_imgs.shape[0]
        tok = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = tok["input_ids"].to(DEVICE)
        attention_mask = tok["attention_mask"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        logits = outputs.logits  # [B, seq, vocab]
        # 取每个样本的最后一个文本 token 的 logits，需补偿插入的视觉 patch 长度
        patch_len = logits.shape[1] - attention_mask.shape[1]
        last_token_indices = patch_len + attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        last_logits = logits[batch_indices, last_token_indices, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)
        loss = -log_probs[:, eos_id].mean()  # 最大化 EOS 概率，相当于停止动作
        
        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录 loss
        loss_history.append(loss.item())
        
        # 投影梯度 (Projected Gradient Descent): 确保 delta 在 epsilon 范围内
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -EPSILON, EPSILON)
            
        pbar.set_description(f"Loss: {loss.item():.4f}")

    print("[+] 优化完成。")

    # 可视化 loss 曲线
    artifacts_dir = os.path.join(os.path.dirname(__file__), "uap_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    loss_fig_path = os.path.join(artifacts_dir, "loss_curve.png")
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood (EOS)")
    plt.title("UAP Optimization Loss")
    plt.tight_layout()
    plt.savefig(loss_fig_path)
    plt.close()
    print(f"[*] Loss 曲线已保存至: {loss_fig_path}")

    return delta.detach().cpu()

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 采集数据
    images_tensor, task_description = collect_data()

    # 2. 加载模型（使用原生 prismatic loader，而不是 HF AutoModel）
    base_model_id = "prism-dinosiglip-224px+7b"
    print(f"[*] 使用 prismatic load 加载: {base_model_id}")

    # 强制走官方 HF 域名，避免镜像不可用
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

    # 读取访问令牌（用户可通过环境变量覆盖）
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("Please set HUGGING_FACE_HUB_TOKEN before running this script.")

    vla = prism_load(base_model_id, hf_token=hf_token)
    vla = vla.to(DEVICE)

    # 处理器：直接从 HF 仓库获取 PrismaticProcessor（含图像处理与 tokenizer）
    # 3. 优化噪声（处理器改为直接用模型内部的 transform + tokenizer）
    final_noise = optimize_uap(vla, None, images_tensor, task_description)
    
    # 4. 保存噪声
    save_full_path = os.path.join(os.path.dirname(__file__), SAVE_PATH)
    torch.save(final_noise, save_full_path)
    
    print(f"[*] 对抗噪声已保存至: {save_full_path}")
    print(f"    Shape: {final_noise.shape}")
    print(f"    Max Val: {final_noise.max().item()}")
    print(f"    Min Val: {final_noise.min().item()}")
