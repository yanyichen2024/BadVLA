import os
from huggingface_hub import hf_hub_download

# 配置代理（根据你的 7897 端口）
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

token = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not token:
    raise RuntimeError("Please set HUGGING_FACE_HUB_TOKEN before running this script.")

try:
    print("正在从 Hugging Face 下载 libero_goal.tar.gz...")
    file_path = hf_hub_download(
        repo_id="howard-haowen/libero",
        filename="libero_goal.tar.gz",
        repo_type="dataset",
        token=token,
        local_dir=".",
        resume_download=True
    )
    print(f"下载成功！文件路径: {file_path}")
except Exception as e:
    print(f"下载失败: {e}")
