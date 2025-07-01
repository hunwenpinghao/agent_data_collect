import os
import json
import time
import pandas as pd
from typing import List
from pathlib import Path
from volcenginesdkarkruntime import Ark


# 设置环境变量
os.environ["ARK_API_KEY"] = "a50666ff-2b30-4ed1-aee1-f7468b57d1df"


def build_recommend_samples_from_csv(csv_path: str) -> List[dict]:
    """从店铺CSV构建推荐格式样本"""
    df = pd.read_csv(csv_path)
    recommend_samples = []

    for _, row in df.iterrows():
        if pd.isna(row["店铺名称"]) or pd.isna(row["二级业态"]) or pd.isna(row["店铺描述"]):
            continue

        category = row["二级业态"]
        name = row["店铺名称"]
        floor = row["店铺地址"] if pd.notna(row["店铺地址"]) else "未知楼层"
        desc = row["店铺描述"].strip()

        instruction = f"请为我推荐一家{category}店铺"
        input_text = ""

        output_text = f"""🌟 【{name}】（{floor}） {category}爱好者必去打卡地！
推荐语：一句话总结店铺推荐语（由 LLM 生成）
📍 亮点：{desc}...
🔥 必试推荐：请补充店内招牌菜、饮品或服务特色。
💡 小贴士：请补充适合场景、服务亮点、好评关键词等。
💡 好搭档：请推荐一家可与之搭配的店铺，并说明搭配理由。"""

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        recommend_samples.append(sample)

    return recommend_samples


def generate_recommendation_content_with_doubao(
    samples: List[dict],
    output_path: str,
    max_samples: int = None,
    sleep_time: float = 0.3
) -> List[dict]:
    """调用 Doubao API 补全推荐内容（亮点/必试推荐/小贴士）"""
    client = Ark(api_key=os.environ.get("ARK_API_KEY"))

    if max_samples:
        samples = samples[:max_samples]

    # load filled samples
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as fin:
            filled_samples = [json.loads(line) for line in fin]
    else:
        filled_samples = []
    
    # fill samples
    for i, sample in enumerate(samples):
        if i < len(filled_samples):
            print(f"[{i+1}/{len(samples)}] ✓ {sample['instruction']}")
            continue

        try:
            title_line = sample["output"].splitlines()[0]
            name = title_line.split("【")[1].split("】")[0]
            floor = title_line.split("）")[0].split("（")[-1]
            category = sample["instruction"].replace("请为我推荐一家", "").replace("店铺", "").strip()
            base_desc = sample["output"].split("亮点：")[1].split("...")[0]

            # 生成 instruction 和 input
            instruction = f"根据提供的店铺信息，以小红书风格生成一段推荐内容，要求包含：一句话推荐语、亮点、必试推荐、小贴士。"
            sample["instruction"] = instruction

            input_text = f"""店铺名称：{name}
地址：{floor}
品类：{category}
简介：{base_desc}
"""
            sample["input"] = input_text

            # 生成 prompt
            prompt = f"""你是一个擅长美食推荐的小助手，请根据以下信息补充推荐内容：
店铺名称：{name}
地址：{floor}
品类：{category}
简介：{base_desc}
请补充以下内容：
(请添加一句简短的、高度总结的、吸引人的推荐开头描述这家店铺，用于放在这里。)
📍 亮点：
🔥 必试推荐：
💡 小贴士：
（请生成带标签的多行内容）"""
            
            response = client.chat.completions.create(
                model="doubao-1.5-pro-32k-250115",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()

#             output_text = f"""🌟 【{name}】（{floor}） {category}爱好者必去打卡地！
# {content}
# 💡 好搭档："""
            output_text = content
            sample["output"] = output_text
            filled_samples.append(sample)

            print(f"[{i+1}/{len(samples)}] ✓ {name}")
            time.sleep(sleep_time)

            if i % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as fout:
                    for sample in filled_samples:
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[{i+1}] ⚠️ Error: {e}")
            continue

    # 保存新样本
    with open(output_path, "w", encoding="utf-8") as fout:
        for s in filled_samples:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ 完成 {len(filled_samples)} 条推荐样本生成，保存在：{output_path}")
    return filled_samples


def standardize_instruction_input(
    original_jsonl_path: str,
    updated_jsonl_path: str
):
    updated_samples = []

    with open(original_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())

            # 从原始 input 中提取字段（假设 input 中是结构化内容）
            input_text = sample.get("input", "")
            # remove suffix
            input_text = input_text.split("请补充以下内容：")[0]
            input_text, base_desc = input_text.split("简介：")
            input_text = input_text.strip()
            input_text, category = input_text.split("品类：")
            input_text = input_text.strip()
            input_text, floor = input_text.split("地址：")
            input_text = input_text.strip()
            input_text, name = input_text.split("店铺名称：")
            
            # 替换 instruction + input
            sample["instruction"] = (
                "根据提供的店铺信息，以小红书风格生成一段推荐内容，"
                "要求包含：一句话推荐语、亮点、必试推荐、小贴士。"
            )

            sample["input"] = f"""店铺名称：{name}
地址：{floor}
品类：{category}
简介：{base_desc}
"""
            updated_samples.append(sample)

    with open(updated_jsonl_path, "w", encoding="utf-8") as f:
        for s in updated_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ 已更新并保存至：{updated_jsonl_path}，共处理 {len(updated_samples)} 条样本。")


# ---------------------------- 非餐饮类店铺 ----------------------------

def generate_recommendation_reason_nonfood(
    csv_path: str,
    output_path: str,
    max_samples: int = None,
    sleep_time: float = 0.3
):
    """为非美食类门店生成推荐理由，适配正弘城汇总非餐饮类数据"""
    client = Ark(api_key=os.environ.get("ARK_API_KEY"))
    df = pd.read_csv(csv_path)
    output_samples = []

    for i, row in df.iterrows():
        try:
            if pd.isna(row["店铺名称"]) or pd.isna(row["二级业态"]) or pd.isna(row["产品服务"]) or pd.isna(row["店铺描述"]):
                continue

            name = row["店铺名称"]
            category = str(row["二级业态"]).strip()
            product = str(row["产品服务"]).strip()
            score = str(row["店铺评分"]).strip()
            price = str(row["人均价格"]).strip()
            label = str(row["店铺标签"]).strip()
            desc = row["店铺描述"].strip()

            prompt = f"""你是一个擅长撰写推荐文案的小助手，请根据以下店铺信息，生成简洁明了的推荐理由，要求包含：店铺特色、产品亮点、适用场景，风格简洁、正式、表达自然。

店铺名称：{name}
主营类型：{category}，{product}

简介：{desc}

推荐理由：(请保留标签)"""

            response = client.chat.completions.create(
                model="doubao-1.5-pro-32k-250115",
                messages=[{"role": "user", "content": prompt}]
            )

            reason = response.choices[0].message.content.strip()

            sample = {
                "instruction": f"你是一个擅长撰写推荐文案的小助手，请根据提供的店铺信息，生成简洁明了的推荐理由，要求突出店铺特色、产品亮点及适用场景。",
                "input": f"店铺名称：{name}\n主营类型：{category}，{product}\n简介：{desc}",
                "output": reason
            }
            output_samples.append(sample)

            print(f"[{len(output_samples)}] ✓ {name}")
            time.sleep(sleep_time)

            if max_samples and len(output_samples) >= max_samples:
                break

            if i % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as fout:
                    for sample in output_samples:
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"⚠️ Error at row {i}: {e}")
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        for s in output_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ 共生成 {len(output_samples)} 条推荐理由，已保存至 {output_path}")


# ✅ 主执行逻辑（组合调用）
if __name__ == "__main__":
    csv_path = "/Users/aibee/hwp/eventgpt/omni-mllm/xiaohongshu_data/正弘城知识库汇总-餐饮店铺列表.csv"
    output_path = "/Users/aibee/hwp/eventgpt/omni-mllm/xiaohongshu_data/zhc_store_recommend_doubao.jsonl"

    # recommend_samples = build_recommend_samples_from_csv(csv_path)
    # print(f"共构建原始推荐样本：{len(recommend_samples)} 条")

    # # 可选：仅测试前 5 条
    # generate_recommendation_content_with_doubao(
    #     samples=recommend_samples,
    #     output_path=output_path,
    #     max_samples=None  # ⚠️ 改为 None 处理全部
    # )

    # This is temporary bug fix code
    # standardize_instruction_input(
    #     original_jsonl_path=output_path,
    #     updated_jsonl_path="/Users/aibee/hwp/eventgpt/omni-mllm/xiaohongshu_data/zhc_store_recommend_doubao_refined.jsonl"
    # )

    # 非餐饮类标签
    generate_recommendation_reason_nonfood(
        csv_path="/Users/aibee/hwp/eventgpt/omni-mllm/xiaohongshu_data/正弘城知识库汇总-店铺数据除餐饮外.csv",
        output_path="/Users/aibee/hwp/eventgpt/omni-mllm/xiaohongshu_data/zhc_store_recommend_reason_doubao.jsonl",
        max_samples=None  # ⚠️ 改为 None 处理全部
    )

    

