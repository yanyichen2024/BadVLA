"""
run_libero_eval_uap_fixed.py

OpenVLA Libero 评测脚本（带 UAP 后门攻击测试）。
修复了动作后处理逻辑，并实现了基于像素空间的噪声注入。
"""

import json
import logging
import os
import sys
import torch
import numpy as np
import tqdm
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import draccus
import wandb
from PIL import Image

# ================= 环境设置 =================
# 强制设置缓存路径和 MuJoCo 后端，防止 Headless 服务器报错
os.environ["HF_DATASETS_CACHE"] = "/home/yyc/.cache/huggingface"
os.environ["HF_HOME"] = "/home/yyc/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/yyc/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/yyc/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../LIBERO")))

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
    center_crop_image,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,            # 原始动作获取函数
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.libero.run_libero_eval import (
    TaskSuite, 
    TASK_MAX_STEPS, 
    GenerateConfig, 
    validate_config, 
    initialize_model, 
    setup_logging, 
    log_message, 
    load_initial_states,
    process_action # [关键] 必须导入此函数用于后续处理
)

# ================= 配置类扩展 =================
@dataclass
class UAPEvalConfig(GenerateConfig):
    # 开关：是否开启攻击
    apply_uap: bool = False
    # 噪声文件路径
    uap_noise_path: str = "/data2/yyc/BadVLA/vla-scripts/siglip_uap_noise.pt"
    # [新增] 视频保存开关，默认关闭
    save_videos: bool = False

# ================= 噪声处理工具函数 =================

def load_uap_noise(noise_path, device="cpu"):
    """加载 .pt 噪声文件并调整维度"""
    if not os.path.exists(noise_path):
        print(f"[Error] 噪声文件未找到: {noise_path}")
        return None
    
    try:
        # 加载噪声 Tensor
        noise = torch.load(noise_path, map_location=device)
        
        # 确保是 float 类型
        noise = noise.float()
        
        # 确保维度是 [1, 3, H, W] 以便广播
        if noise.dim() == 3:
            noise = noise.unsqueeze(0)
            
        print(f"[*] 成功加载 UAP 噪声: {noise_path}, Shape: {noise.shape}")
        print(f"[*] 噪声数值范围: Min={noise.min():.4f}, Max={noise.max():.4f}")
        return noise
    except Exception as e:
        print(f"[Error] 加载噪声失败: {e}")
        return None

def inject_noise_numpy(img_numpy, noise_tensor):
    """
    将噪声注入到 Numpy 图像中。
    Args:
        img_numpy: [H, W, 3] uint8 numpy array (OpenVLA resize 后的图像)
        noise_tensor: [1, 3, H, W] tensor (float, 建议范围 0-1 或 0-255 对应处理)
    Returns:
        noisy_img_numpy: [H, W, 3] uint8 numpy array
    """
    if noise_tensor is None:
        return img_numpy
    
    # 1. 图像转 Tensor [C, H, W] 并归一化到 [0, 1]
    # 注意：这里假设你的 noise_tensor 是基于 [0, 1] 范围生成的
    img_tensor = torch.from_numpy(img_numpy).permute(2, 0, 1).float() / 255.0
    
    # 2. 检查并调整噪声尺寸 (如果噪声是 224x224，图片也是 224x224，则无需调整)
    _, _, h_img, w_img = img_tensor.unsqueeze(0).shape
    _, _, h_noise, w_noise = noise_tensor.shape
    
    if (h_img != h_noise) or (w_img != w_noise):
        # 仅当尺寸不匹配时进行插值
        noise_to_add = torch.nn.functional.interpolate(
            noise_tensor, size=(h_img, w_img), mode='bilinear', align_corners=False
        )
    else:
        noise_to_add = noise_tensor

    # 3. 注入噪声 (img + noise)
    # 假设 noise_tensor 已经是 [0, 1] 范围的 delta
    noisy_tensor = img_tensor + noise_to_add.squeeze(0).cpu()
    
    # 4. 截断到有效范围 [0, 1]
    noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
    
    # 5. 转回 Numpy uint8 [H, W, 3]
    noisy_img = noisy_tensor.permute(1, 2, 0).numpy()
    noisy_img = (noisy_img * 255).astype(np.uint8)
    
    return noisy_img

# ================= 单个 Episode 运行逻辑 =================

def run_episode_uap(
        cfg: UAPEvalConfig,
        env,
        task_description: str,
        model,
        resize_size,
        uap_noise_tensor=None, # 新增接收噪声
        processor=None,
        action_head=None,
        proprio_projector=None,
        noisy_action_projector=None,
        initial_state=None,
        log_file=None,
):
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            # 预热阶段
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # 1. 获取图像并 Resize 到 224x224（主视角 + 手腕）
            img_raw = get_libero_image(obs) # [256, 256, 3]
            wrist_img_raw = get_libero_wrist_image(obs)
            
            img_resized = resize_image_for_policy(img_raw, resize_size) # [224, 224, 3]
            wrist_img_resized = resize_image_for_policy(wrist_img_raw, resize_size)

            # 与训练 / 对抗优化一致：中心裁剪
            if cfg.center_crop:
                img_resized = np.array(center_crop_image(img_resized))
                wrist_img_resized = np.array(center_crop_image(wrist_img_resized))

            # ================= [攻击注入点] =================
            # 在这里将噪声“贴”到已经 Resize 好的图片上
            # 这样 OpenVLA 看到的这一帧就是被污染的
            # 仅对主视角相机加噪声，保持手腕相机干净以复现训练分布
            if cfg.apply_uap and uap_noise_tensor is not None:
                img_final = inject_noise_numpy(img_resized, uap_noise_tensor)
                wrist_img_final = wrist_img_resized
            else:
                img_final = img_resized
                wrist_img_final = wrist_img_resized
            # ===============================================

            # 构建观测字典
            observation = {
                "full_image": img_final,  # 传入 (可能带毒的) 图片
                "wrist_image": wrist_img_final,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            # 记录用于视频生成的图像 (记录带噪声的图像以便观察攻击效果)
            replay_images.append(img_final)

            if len(action_queue) == 0:
                # 2. 调用原始 get_action (保持 OpenVLA 原始推理逻辑)
                orig_center_crop = cfg.center_crop
                cfg.center_crop = False  # 已经裁剪，避免重复裁剪导致分布偏移
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                cfg.center_crop = orig_center_crop
                action_queue.extend(actions)

            action = action_queue.popleft()
            
            # 3. [关键修复] 执行动作后处理 (归一化 + 倒置)
            # 这一步如果不做，Libero 环境会收到错误的夹爪指令，导致 Success=0
            action = process_action(action, cfg.model_family)

            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        import traceback
        traceback.print_exc()

    return success, replay_images

# ================= 主评测循环 =================

@draccus.wrap()
def eval_libero_uap(cfg: UAPEvalConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    
    # 初始化模型
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    
    # 加载噪声
    uap_noise = None
    if cfg.apply_uap:
        uap_noise = load_uap_noise(cfg.uap_noise_path)
        if uap_noise is None:
            log_message("Warning: UAP enabled but noise file load failed. Running CLEAN eval.", log_file)
    
    # 准备任务
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Model: {cfg.pretrained_checkpoint}", log_file)
    log_message(f"UAP Attack Enabled: {cfg.apply_uap}", log_file)

    total_episodes, total_successes = 0, 0
    
    for task_id in tqdm.tqdm(range(num_tasks)):
        task = task_suite.get_task(task_id)
        initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        
        task_episodes, task_successes = 0, 0
        for episode_idx in range(cfg.num_trials_per_task):
            
            # 演示模式只跑前5个episode，全量评测请注释掉下面两行
            # if episode_idx >= 5: 
            #     break
            
            if cfg.initial_states_path == "DEFAULT":
                initial_state = initial_states[episode_idx]
            else:
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    continue
                initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

            log_message(f"Task: {task_description} | Episode: {episode_idx} | UAP: {cfg.apply_uap}", log_file)
            
            success, replay_images = run_episode_uap(
                cfg,
                env,
                task_description.lower(),
                model,
                resize_size,
                uap_noise_tensor=uap_noise, # 传递噪声
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                initial_state=initial_state,
                log_file=log_file,
            )

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1
            
            # 保存视频（如果需要 debug 可以开启）
            if cfg.save_videos:
                save_rollout_video(replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file)
            
            log_message(f"Success: {success}", log_file)

        current_task_rate = task_successes / task_episodes if task_episodes > 0 else 0
        log_message(f"Task Success Rate: {current_task_rate:.2f}", log_file)

    final_success_rate = total_successes / total_episodes if total_episodes > 0 else 0
    log_message(f"Overall Success Rate: {final_success_rate:.4f}", log_file)
    
    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_success_rate})
        
    return final_success_rate

if __name__ == "__main__":
    eval_libero_uap()