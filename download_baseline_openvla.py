"""
download_baseline_openvla.py

Downloads the baseline OpenVLA-7B model from Hugging Face Hub.
"""

import os
import torch
from huggingface_hub import snapshot_download

# 配置缓存目录
os.environ["HF_DATASETS_CACHE"] = "./cache"
os.environ["HF_HOME"] = "./cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_openvla_baseline():
    repo_id = "openvla/openvla-7b"
    print(f"[*] Downloading {repo_id} to ./cache ...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=None, # Automatically handled by cache
            resume_download=True,
            max_workers=8
        )
        print("[+] Download complete!")
    except Exception as e:
        print(f"[-] Download failed: {e}")

if __name__ == "__main__":
    download_openvla_baseline()
