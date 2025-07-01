# python3.11 -m pip install 'volcengine-python-sdk[ark]'
import json
import time
from volcenginesdkarkruntime import Ark
import os
from typing import List

# 设置环境变量
os.environ["ARK_API_KEY"] = "a50666ff-2b30-4ed1-aee1-f7468b57d1df"

# 初始化火山引擎 Doubao 模型客户端
client = Ark(api_key=os.environ.get("ARK_API_KEY"))

# 读取样本文件
input_path = "/Users/aibee/hwp/eventgpt/omni-mllm/xiaohongshu_data/store_xhs_rewrite_samples.jsonl"
output_path = "/Users/aibee/hwp/eventgpt/omni-mllm/xiaohongshu_data/store_xhs_rewrite_samples_doubao_output.jsonl"


def rewrite_samples(input_path: str, output_path: str, client, sleep_time: float = 0.3):
    """
    读取样本文件，调用 Doubao 模型生成改写文本，并保存结果。
    :param input_path: 输入 JSONL 文件路径
    :param output_path: 输出 JSONL 文件路径
    :param client: 已初始化的 Ark 客户端
    :param sleep_time: 每条样本之间的等待时间（避免限流）
    """
    with open(input_path, "r", encoding="utf-8") as fin:
        samples = [json.loads(line) for line in fin]

    result_samples = []
    for i, sample in enumerate(samples):
        prompt = f"{sample['instruction'].strip()}\n{sample['input'].strip()}"
        try:
            completion = client.chat.completions.create(
                model="ep-20250701152634-c7r6w",  # "doubao-1.5-pro-32k-250115",
                messages=[{"role": "user", "content": prompt}]
            )
            response = completion.choices[0].message.content.strip()
            sample["output"] = response
            result_samples.append(sample)

            print(f"[{i+1}/{len(samples)}] ✓ {sample['instruction'][:20]}...")
            time.sleep(sleep_time)

            if i % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as fout:
                    for sample in result_samples:
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[{i+1}] Error: {e}")
            continue

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in result_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ 完成！已生成 {len(result_samples)} 条改写数据：{output_path}")


if __name__ == "__main__":
    rewrite_samples(input_path, output_path, client)


