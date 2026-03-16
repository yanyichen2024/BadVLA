import os
from huggingface_hub import snapshot_download

# 1. 确保不走镜像，直接连官方 Hub
if "HF_ENDPOINT" in os.environ:
    del os.environ["HF_ENDPOINT"]

repo_id = "openvla/modified_libero_rlds"
local_dir = "./modified_libero_rlds"

print(f"正在通过代理 {os.environ.get('ALL_PROXY')} 下载...")

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=4,       # 使用代理时建议并发不要太高，防止被代理服务器限速
        etag_timeout=60,     # 增加超时等待时间
        resume_download=True
    )
    print("下载成功！")
except Exception as e:
    print(f"\n下载失败。如果依然提示 'Missing dependencies for SOCKS support'，")
    print(f"请检查是否成功运行了 'pip install pysocks'。")
    print(f"报错详情: {e}")