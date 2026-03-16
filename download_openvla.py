from safetensors import safe_open
import os

base_path = "/data2/yyc/BadVLA/cache/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b"
shards = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]

for shard in shards:
    file_path = os.path.join(base_path, shard)
    print(f"正在检查: {shard} ... ", end="")
    try:
        with safe_open(file_path, framework="pt") as f:
            # 尝试读取一个键来验证
            _ = f.keys()
        print("✅ 正常")
    except Exception as e:
        print(f"❌ 损坏! 错误: {e}")