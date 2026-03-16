import torch
import matplotlib.pyplot as plt
import os

# 加载噪声
noise_path = "/data2/yyc/BadVLA/vla-scripts/siglip_uap_noise.pt"
noise = torch.load(noise_path)

# 归一化处理
noise_vis = (noise - noise.min()) / (noise.max() - noise.min())
plt.imshow(noise_vis.squeeze().permute(1, 2, 0).cpu().numpy())
plt.title("Generated UAP Noise (Visualized)")

# 保存到当前目录
save_path = "uap_visualization.png"
plt.savefig(save_path)
print(f"[*] 图像已保存至: {os.path.abspath(save_path)}")