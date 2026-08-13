from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_aspect_opinion(text, tokenizer, model, device="cuda"):
    """
    输入：一条评论文本
    输出：模型生成的 JSON 对象（Python list）
    """
    # -------------------------------
    # 1. System Prompt：固定模型身份和输出格式
    # -------------------------------
    system_prompt = (
    "You are an expert system for extracting fine-grained service insights from hotel reviews. "
    "Your goal is to identify all aspect–opinion–attribute–sentiment tuples and provide a short reasoning explaining the evidence or semantic cue in the review that indicates the corresponding service attribute. "
    "Output strictly in a JSON list format, where each element contains: aspect (a), opinion (o), category (c), sentiment_score (s), and reasoning (r). "
    "The reasoning must follow the format '[evidence phrase] → [evaluation dimension]' with no more than 15 words."
)
    # ===============================================================
    # 2. User Prompt — 指令 + 示例 + 当前输入
    # ===============================================================
    user_prompt = f'''
You are a hotel review analysis assistant.
Extract all aspect–opinion–category–sentiment tuples with reasoning from the given review.
Follow these rules carefully and output **strictly in JSON format** (no extra text, no explanations).

---
### Guidelines:
**Fields**:
- "a": the **specific entity or evaluative dimension explicitly mentioned** in the review text.  
  It **must appear verbatim** in the review. Use "nan" if the aspect is **not explicitly stated**.  

- "o": the **exact descriptive phrase** expressing the evaluation, opinion, or attitude that appears in the review text.  refers to the sentiment or attitude expressed by a user towards a particular aspect or feature of a product or service.

- "c": choose the correct category ID from 1–11 according to the **Category Definitions** below.

- "s": sentiment score between 0 and 1 (0 = very negative, 0.5 = neutral, 1 = very positive). Use two decimal places. Values <0.5 indicate negative sentiment, >0.5 indicate positive sentiment.

- "r": reasoning in the fixed template form:  
  "[evidence phrase] → [evaluation dimension]"
  The *evidence phrase* should be the most informative clue from the review text (aspect, opinion, or contextual fragment). If the evidence comes from context, include the relevant phrase.

**Output format**: a JSON list only, with fields [{{"a":"...", "o":"...", "c":..., "s":..., "r":"..."}}]. Do not include any explanations or extra text outside the JSON.

###Category Definitions：
1: Cleanliness: 
    — Refers to hygiene, smell, or cleanliness issues within private rooms or unspecified areas. 
    — Boundary: If the opinion refers to smell or odor, classify here unless it explicitly concerns comfort or freshness; 
        If the area mentioned is clearly a public space (e.g., lobby, pool, corridor), classify as (6) Public Area instead; 
        If the area is unspecified or clearly a private area (room, bathroom, suite), classify as (1) Cleanliness.
    — Examples: "The bathroom smelled bad." / "Room was covered with dust."/ "cupboard doors were never wiped down." / "Fresh and clean air."
    
2: Comfort and Security:
    — Concerns guest’s experience of comfort, noise, temperature, sleep quality, atmosphere, or safety.
    — Boundary: Even if an object is mentioned, classify here if the statement describes how it feels — e.g., noisy, cold, warm, uncomfortable, unsafe.
    — Examples: “Air-con was noisy.” / “Bed was uncomfortable.” / “needs more blankets and maybe a heater we were very cold” / “the curtains can block all the light.”/ “security screen door had no key and was unlockable.” / “could hear the music from the event building.”

3: Food and Drink 
    — Evaluates quality, taste, freshness, or availability and supply of food. Including the supply of bottled water or snacks in the room.
    — Examples: “Breakfast was cold and tasteless.” / “Coffee was great.” / “awful food in boxes, no condiments.” / “the room is equipped with bottled water and chocolate.” / “rooms needed more coffee and tea.” / "restaurant open late so makes dining easy."
    
4: Location: 
    — Refers to accessibility, transportation, and nearby environment or view.
    — Boundary: When both a public area (e.g., pool) and view are mentioned, choose Location (4) if the evaluation concerns the view itself.
    — Examples: “Close to city center.” / “Far from restaurants.” / “Beautiful view from the balcony.” / “convenience and shuttle”

5: Parking
    — Parking availability, convenience, or cost.
    — Examples: “No parking space.” / “Parking was expensive.” / “Free parking nearby.”

6: Public area 
    — Shared hotel spaces and pubic facilities such as lobby, spa, pool, rooftop, gym, garden, corridors, lift/elevator, etc., including their cleanliness, appearance, or availability. If the described area/building is not a private space within the suite/room, select this option.
    — Boundary: If the review explicitly mentions these shared areas, classify as (6) even when the issue is cleanliness-related (e.g., “The lobby was dirty.”). 
     — Examples: “Pool was closed.” / “Lobby looked beautiful.” / “Dirty corridor.” / “disappointing the rooftopspa was closed.” / “Corridor smelled bad.”/ “Lobby was dirty.” / “Pool water was cloudy.” 

7: Reception 
    — Relates to front desk, check-in/out, booking, refund/deposit, luggage handling, or greeting process. 
    — Examples: “issues with front counter.” / “the young man on reception was rude obnoxious and very unhelpful.” / “appreciate having 24 hr reception staff.” / “Called the front desk but there was no answer.” / “some way of getting luggage up to the 2nd floors would be great.” 

8: Room and Facility 
    — Refers to physical condition/damage, size, quality, functionality and design of room, furniture, appliances, WiFi, layout and decor.
    — Boundary: If the opinion describes user comfort (e.g., noisy, uncomfortable), classify as (2) Comfort.
    — Examples: “Shower handle broken.” / “Wi-Fi didn’t work.” / “Room design was modern.”/ “The lighting was a little dim.”/ “Rooms not being serviced daily.” / “couldn't connect with the internet.”

9: Staff and Professionalism 
    — Refers to attitude, behavior, helpfulness, and professionalism of staff or management, rules and policies.
    — Boundary: If the activity takes place before check-in (e.g., booking, check-in/out, greeting, welcoming), or discussing about reception staff, classify as (7) Reception instead. 
    — Examples: “The owner was rude.” / "communication was almost non existant and multiple phone calls were not returned." / "They are pet friendly."
    
10: Value 
    — Relates to price fairness, cost-effectiveness, or value perception.
    — Examples: “Too expensive for the quality.” / “Worth every penny.” / “Unlike five-star hotels.” / “Rather than a four-star hotel.”


**Keyword Hints**:
— Use the following as semantic orientation clues. 
— The final classification should depend on meaning and context.
	1. Cleanliness: dirty, smell, odor, clean, dusty, stain, mould, hygiene, sweep
	2. Comfort & Security: quiet, noisy, cold, hot, warm, peaceful, homeless person, unsafe, sleep
	3. Food & Drink: breakfast, coffee, bottled water, chocolate, beverage, vending machine
	4. Location: location, near, far, close to, distance, view, easy walk
	5. Parking: car park, garage, parking fee
	6. Public Area: lobby, pool, rooftop, gym, garden, corridor, rooftop, spa, terrace, bar, foyer, lifts, elevator
	7. Reception: check-in, booking, refund, counter, front desk, luggage, welcome, greeting, deposit
	8. Room and Facility: furniture, WiFi, design, layout, decor, maintenance
	9. Staff and Professionalism: manager, owner, housekeeping staff, concierge, attendant, room service
    10. Value: price, expensive, cheap, cost, worth, money, budget 
    11. Other
---  
### Examples:
Example 1:
Review: "Room was spacious but smelled awful."
Output:
[
  {{"a": "room", "o": "spacious", "c": 8, "s": 0.85, "r": "'spacious' → room feature"}},
  {{"a": "room", "o": "smelled awful", "c": 1, "s": 0.19, "r": "'smell' → hygiene condition"}}
]

Example 2:
Review: "Room was overpriced."
Output:
[
  {{"a": "room", "o": "overpriced", "c": 10, "s": 0.29, "r": "'overpriced' → room price"}}
]

Example 3:
Review: "Not worth the money."
Output:
[
  {{"a": "nan", "o": "Not worth the money", "c": 10, "s": 0.29, "r": "'Not worth the money' → low cost performance"}}
]

Example 4:
Review: "The pool had a strong chlorine smell and the view was perfect."
Output:
[
  {{"a": "pool", "o": "had a strong chlorine smell", "c": 6, "s": 0.3, "r": "'strong chlorine smell' refers to the environment of pool"}},
  {{"a": "view", "o": "perfect", "c": 4, "s": 0.95, "r": "'view was perfect' → good location"}}
]

Example 5:
Review: "The lights left on out front of rooms all night shining through the curtains."
Output:
[
  {{"a": "lights", "o": "all night shining through the curtains", "c": 2, "s": 0.28, "r": "'all night shining through the curtains' → poor sleep quality"}}
]

Example 6:
Review: "Cleaning staff should check that previous visitors have emptied the filter in the dryer to ensure no fires."
Output:
[
  {{"a": "ensure no fires", "o": "Cleaning staff should check that previous visitors have emptied the filter in the dryer", "c": 2, "s": 0.4, "r": "'ensure no fires' → concernes about risk of fire"}}
]

Example 7:
Review: "The grounds are lovely to walk through, bamboo everywhere plus a little fish filled pond."
Output:
[
  {{"a": "grounds", "o": "lovely to walk through", "c": 6, "s": 0.9, "r": "'grounds' → public area"}},
  {{"a": "grounds", "o": "bamboo everywhere plus a little fish filled pond", "c": 6, "s": 0.9, "r": "'grounds', 'bamboo everywhere' and  'fish filled pond' → public area"}}
]

Example 8:
Review: "Some way of getting luggage up to the 2nd and 3rd floors would be great."
Output:
[
  {{"a": "some way of getting luggage up to the 2nd and 3rd floors", "o": "would be great", "c": 7, "s": 0.43, "r": "'getting luggage up to the 2nd and 3rd floors' → luggage handling"}}
]

---
Now analyze the following review:
"{text}"
'''
    # -------------------------------
    # 3. 构造消息输入
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat_text], return_tensors="pt", padding=True).to(device)
    # -------------------------------
    # 4. 模型生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    # 提取新生成部分
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # -------------------------------
    # 5. 尝试解析 JSON
    # -------------------------------
    try:
        parsed_output = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            try:
                parsed_output = json.loads(match.group(0))
            except Exception:
                parsed_output = [{"raw_output": response}]
        else:
            parsed_output = [{"raw_output": response}]
    # -------------------------------
    # 6. 清理显存
    # -------------------------------
    del inputs, generated_ids
    torch.cuda.empty_cache()

    return parsed_output


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map="auto",
        torch_dtype="auto",
        offload_folder="offload",
        trust_remote_code=True,
    ).eval()
    return tokenizer, model


def run_extraction(
    input_file: Path,
    output_file: Path,
    model_dir: Path,
    text_column: str,
    save_interval: int,
    device: str,
) -> None:
    if input_file.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    if text_column not in df.columns:
        raise KeyError(f"Missing text column: {text_column}")

    tokenizer, model = load_model(model_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume from actually completed rows only. The original checkpoint file
    # stored the whole DataFrame, so using its row index alone could incorrectly
    # mark unfinished rows as complete.
    if output_file.exists():
        saved = pd.read_csv(output_file)
        if len(saved) != len(df):
            raise ValueError("Checkpoint row count differs from input row count.")
        if "AOP_extraction" in saved.columns:
            df["AOP_extraction"] = saved["AOP_extraction"]
        else:
            df["AOP_extraction"] = None
    else:
        df["AOP_extraction"] = None

    completed = df["AOP_extraction"].notna() & df["AOP_extraction"].astype(str).str.strip().ne("")
    pending = df.index[~completed].tolist()
    print(f"Rows: {len(df):,}; completed: {int(completed.sum()):,}; pending: {len(pending):,}")

    for step, i in enumerate(tqdm(pending, desc="Extracting AOCS"), start=1):
        text = df.at[i, text_column]
        try:
            result = extract_aspect_opinion(text, tokenizer, model, device)
            df.at[i, "AOP_extraction"] = json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            df.at[i, "AOP_extraction"] = json.dumps(
                [{"error": str(exc), "text": str(text)}], ensure_ascii=False
            )

        if step % save_interval == 0 or step == len(pending):
            df.to_csv(output_file, index=False)
            print(f"Checkpoint saved at {datetime.now().strftime('%H:%M:%S')}: {output_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract AOCS tuples from sentence-level hotel reviews using Qwen2-7B-Instruct-GPTQ-Int4.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Checkpoint/output CSV containing AOP_extraction.")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--text-column", default="split_text")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(args.input, args.output, args.model_dir, args.text_column, args.save_interval, args.device)
