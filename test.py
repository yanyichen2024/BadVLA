from transformers import AutoTokenizer
import numpy as np

# 1. 配置路径（请确保路径正确）
model_path = "/data2/yyc/BadVLA/cache/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b"

# 2. 加载词表器
# OpenVLA 基于 Llama-2，其动作 Token 始终映射在固定范围内
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 3. 定义 OpenVLA 官方标准的 Action Token 范围
# 即使 vocab_size 增加（如 32001），Action IDs 依然是 [31744, 31999]
ACTION_TOKEN_BEGIN_IDX = 31744 
NUM_ACTION_TOKENS = 256

# 4. 提取 Action Tokens
action_token_ids = np.arange(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS)
action_tokens = [tokenizer.decode([token_id]) for token_id in action_token_ids]

# --- 打印结果验证 ---
print("="*50)
print(f"Model Vocab Size: {len(tokenizer)}")
print(f"Action Token Range: ID {action_token_ids[0]} to {action_token_ids[-1]}")
print("="*50)

print("\n[Preview] First 5 Action Tokens (Bin 0-4):")
for i in range(5):
    print(f"  Bin {i}: ID {action_token_ids[i]} -> '{action_tokens[i]}'")

print("\n[Preview] Last 5 Action Tokens (Bin 251-255):")
for i in range(251, 256):
    print(f"  Bin {i}: ID {action_token_ids[i]} -> '{action_tokens[i]}'")

print("\n" + "="*50)
print("All 256 Action Tokens List:")
print(action_tokens)