import json
import pandas as pd
from pathlib import Path


def convert_csv_to_jsonl(csv_path, output_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["店铺描述"])
    samples = []
    for _, row in df.iterrows():
        if pd.isna(row["店铺描述"]):
            continue  # 跳过没有描述的样本

        # 构造 input
        input_parts = []
        if pd.notna(row["店铺名称"]):
            input_parts.append(f"店铺名称：{row['店铺名称']}")
        if pd.notna(row["二级业态"]):
            input_parts.append(f"品类：{row['二级业态']}")
        elif pd.notna(row["菜系"]):
            input_parts.append(f"品类：{row['菜系']}")
        if pd.notna(row["店铺地址"]):
            input_parts.append(f"地址：{row['店铺地址']}")
        if pd.notna(row["营业时间"]):
            input_parts.append(f"营业时间：{row['营业时间']}")
        if pd.notna(row["环境"]):
            input_parts.append(f"环境风格：{row['环境']}")
        if pd.notna(row["设施"]):
            input_parts.append(f"配套设施：{row['设施']}")
        input_text = "\n".join(input_parts)
        instruction = "根据以下店铺信息，生成一段小红书风格的文案，风格为「甜美」："
        output_text = row["店铺描述"].strip()

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        samples.append(sample)


    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    csv_path = "/Users/aibee/hwp/eventgpt/omni-mllm/xiaohongshu_data/正弘城知识库汇总-餐饮店铺列表.csv"  # 输入 CSV 文件路径
    output_path = "/Users/aibee/hwp/eventgpt/omni-mllm/xiaohongshu_data/zhc_xhs_data_sft.jsonl"  # 输出 JSONL 文件路径
    convert_csv_to_jsonl(csv_path, output_path)

