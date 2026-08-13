"""Step 2: LLM-assisted consolidation of standardized business dimensions.

Low-frequency labels are first retrieved against high-frequency labels using
SentenceTransformer cosine similarity. Very high/low similarity cases are
handled by rules; ambiguous cases are passed to the LLM. The LLM is restricted
to choosing one retrieved label or keeping the low-frequency label independent.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """你是严谨高效的数据治理专家，专职进行文本标签的标准化和合并工作。
你的核心目标是：在不扩大语义范围的前提下，最大限度地将低频、冗余标签合并到高频标准标签上，以提升数据整体的聚合度和业务可操作性。
你必须严格遵守以下指令，并且只输出最终要求的 JSON 格式结果，不包含任何解释性或问候性文字。"""


def make_user_prompt(low_label: str, candidates: list[dict]) -> str:
    candidates_str = json.dumps(candidates, ensure_ascii=False, indent=0)
    return f"""请评估下面待合并的低频标签（\"待判断标签\"）是否应合并到“候选列表”中的某一个高频标签。

待判断标签：\"{low_label}\"

候选列表 (名称 | 相似度):
{candidates_str}

合并/独立原则：
1. 积极合并（高召回）：若语义高度重叠（同一问题、场景或方面），应合并；
2. 粒度区分（高精度）：能合并则合并，除非存在明显的侧重点或粒度差异；
3. 不得创造新标签；只能选择候选列表中的一个标签，或返回“独立”。

请仅输出以下 JSON：
{{
  \"decision\": \"<候选name 或 独立>\",
  \"confidence\": \"0.0-1.0\",
  \"reason\": \"一句话说明为何合并或保持独立\"
}}"""


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded checkpoint with {len(data)} labels.")
    return data


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def retrieve_candidates(low_labels, high_labels, sbert, top_k: int):
    if not high_labels:
        raise ValueError("No high-frequency labels are available for candidate retrieval.")
    k = min(top_k, len(high_labels))
    high_emb = sbert.encode(
        high_labels,
        convert_to_tensor=True,
        batch_size=256,
        show_progress_bar=True,
    )
    low_emb = sbert.encode(
        low_labels,
        convert_to_tensor=True,
        batch_size=256,
        show_progress_bar=True,
    )
    similarities = util.cos_sim(low_emb, high_emb)
    top_vals, top_inds = torch.topk(similarities, k, dim=1)

    results = []
    for vals, inds in zip(top_vals, top_inds):
        results.append(
            [
                {"name": high_labels[int(idx)], "score": round(float(score), 4)}
                for score, idx in zip(vals, inds)
            ]
        )
    return results


def parse_decision(response: str) -> str:
    """Parse the decision while retaining the original safe fallback behavior."""
    try:
        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(response[start : end + 1])
            return str(payload.get("decision", "独立")).strip()
    except Exception:
        pass

    match = re.search(r'"?decision"?\s*:\s*"([^"]+)"', response, re.S)
    return match.group(1).strip() if match else "独立"


def llm_batch_infer(
    tasks,
    merged_map,
    tokenizer,
    model,
    checkpoint_path: Path,
    batch_size: int,
    save_interval: int,
    input_device: str,
):
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    for batch_no, start in enumerate(
        tqdm(range(0, len(tasks), batch_size), desc="LLM merging"), start=1
    ):
        batch_tasks = tasks[start : start + batch_size]
        prompts = []
        for task in batch_tasks:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_prompt(task["low"], task["cands"])},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(input_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        generated = outputs[:, inputs.input_ids.shape[1] :]
        responses = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for task, response in zip(batch_tasks, responses):
            low_label = task["low"]
            decision = parse_decision(response)
            valid_names = {c["name"] for c in task["cands"]}

            if decision in valid_names:
                merged_map[low_label] = decision
            else:
                # "独立", malformed JSON, or hallucinated label -> keep original label.
                merged_map[low_label] = low_label

        if batch_no % save_interval == 0 or batch_no == total_batches:
            save_checkpoint(checkpoint_path, merged_map)

    return merged_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="S1 output .xlsx")
    parser.add_argument("--output", required=True, help="S2 output .xlsx")
    parser.add_argument("--sbert-model", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--checkpoint", default="outputs/S2_llm_merge_checkpoint.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--high-freq-threshold", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--auto-merge-threshold", type=float, default=0.85)
    parser.add_argument("--auto-independent-threshold", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_excel(args.input)

    freq = Counter(df["D_Std"])
    high_freq = [d for d, count in freq.items() if count >= args.high_freq_threshold]
    low_freq = [d for d, count in freq.items() if count < args.high_freq_threshold]
    print(f"High-frequency labels: {len(high_freq)}; low-frequency labels: {len(low_freq)}")

    sbert = SentenceTransformer(args.sbert_model)
    all_candidates = retrieve_candidates(low_freq, high_freq, sbert, args.top_k)

    checkpoint_path = Path(args.checkpoint)
    merged_map = load_checkpoint(checkpoint_path) if args.resume else {}
    llm_tasks = []

    for low, candidates in zip(low_freq, all_candidates):
        if args.resume and low in merged_map:
            continue
        top_score = candidates[0]["score"]
        top_name = candidates[0]["name"]
        if top_score >= args.auto_merge_threshold:
            merged_map[low] = top_name
        elif top_score < args.auto_independent_threshold:
            merged_map[low] = low
        else:
            llm_tasks.append({"low": low, "cands": candidates})

    print(f"Ambiguous labels requiring LLM: {len(llm_tasks)}")

    if llm_tasks:
        tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model,
            padding_side="left",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            device_map="auto",
            torch_dtype="auto",
            offload_folder="offload",
            trust_remote_code=True,
        ).eval()
        input_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        merged_map = llm_batch_infer(
            llm_tasks,
            merged_map,
            tokenizer,
            model,
            checkpoint_path,
            args.batch_size,
            args.save_interval,
            input_device,
        )

    for label in high_freq:
        merged_map[label] = label

    result = df.copy()
    result["D_Final"] = result["D_Std"].map(merged_map).fillna(result["D_Std"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output, index=False)
    save_checkpoint(checkpoint_path, merged_map)
    print(f"Merged labels: {sum(k != v for k, v in merged_map.items())}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
