# -*- coding: utf-8 -*-
"""
Streamlit front end for the hotel service improvement interaction workflow.

Run:
    streamlit run app.py

Project layout:
    hotel_qwen_streamlit_system/
      app.py
      strategy_core_api.py
      data/
        cluster_aop.xlsx
        MOO_res.xlsx
      outputs/
"""

from __future__ import annotations

import json
import os
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from strategy_core_api import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    apply_manual_edits,
    build_candidate_clusters,
    build_options,
    build_per_dict,
    create_interpreter,
    discover_excel_files,
    finalize_results,
    generate_one_proposal,
    map_scores_to_guidance,
    preprocess_history,
    read_excel_from_data,
    save_outputs,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
OUTPUT_DIR = APP_DIR / "outputs"

st.set_page_config(
    page_title="Hotel Strategy Generator - Qwen API",
    page_icon="🏨",
    layout="wide",
)


def init_state() -> None:
    defaults = {
        "df_history_ready": None,
        "df_per": None,
        "options": None,
        "per_dict": None,
        "candidate_clusters": [],
        "selected_clusters": [],
        "proposals_for_solution": [],
        "drop_reason_counter": {},
        "global_check_result": None,
        "final_proposals": [],
        "final_result_df": None,
        "api_usage": {},
        "last_saved_path": None,
        "delta_threshold": 0.50,
        "evidence_topQ": 8,
        "candidate_pool_multiplier": 5,
        "score_weight": 0.35,
        "informativeness_weight": 0.20,
        "representativeness_weight": 0.25,
        "consistency_weight": 0.15,
        "direction_weight": 0.05,
        "diversity_penalty": 0.10,
        "min_token_threshold": 8,
        "neighbor_q": 10,
        "consistency_eta": 2.0,
        "direction_delta": 0.25,
        "direction_tau": 1.0,
        "similarity_backend": "sentence-transformers",
        "sentence_model_path": "sentence-transformers/all-MiniLM-L6-v2",
        "max_generate": 20,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "request_retries": 2,
        "timeout": 120.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def df_to_excel_bytes(result_df: pd.DataFrame, proposals_df: pd.DataFrame | None = None) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="final_results", index=False)
        if proposals_df is not None:
            proposals_df.to_excel(writer, sheet_name="final_proposals_raw", index=False)
    return output.getvalue()



def make_interpreter_from_sidebar(
    history_df: pd.DataFrame,
    *,
    load_similarity: bool = True,
):
    similarity_backend = (
        st.session_state.get(
            "similarity_backend",
            "sentence-transformers",
        )
        if load_similarity
        else "none"
    )
    return create_interpreter(
        api_key=st.session_state.get("api_key_input", ""),
        history_df=history_df,
        model=st.session_state.get("model_name", DEFAULT_MODEL),
        base_url=st.session_state.get(
            "base_url",
            DEFAULT_BASE_URL,
        ),
        temperature=float(
            st.session_state.get("temperature", 0.7)
        ),
        top_p=float(st.session_state.get("top_p", 0.9)),
        max_tokens=int(
            st.session_state.get("max_tokens", 512)
        ),
        max_retries=int(
            st.session_state.get("request_retries", 2)
        ),
        timeout=float(
            st.session_state.get("timeout", 120.0)
        ),
        request_retries=int(
            st.session_state.get("request_retries", 2)
        ),
        similarity_backend=similarity_backend,
        sentence_model_path=st.session_state.get(
            "sentence_model_path",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )


def reset_after_data_change() -> None:
    for key in [
        "proposals_for_solution",
        "drop_reason_counter",
        "global_check_result",
        "final_proposals",
        "final_result_df",
        "api_usage",
        "last_saved_path",
    ]:
        st.session_state[key] = {} if key in {"drop_reason_counter", "api_usage"} else [] if key in {"proposals_for_solution", "final_proposals"} else None


init_state()

st.title("Hotel improvement strategy interactive generation system")

with st.sidebar:
    st.header("Configuration")
    st.text_input(
        "API KEY",
        value=os.getenv("DASHSCOPE_API_KEY", ""),
        type="password",
        key="api_key_input",
        #help="不会写入代码；可直接粘贴，也可用环境变量 DASHSCOPE_API_KEY。",
    )
    st.text_input("OpenAI-compatible base_url", value=DEFAULT_BASE_URL, key="base_url")
    st.text_input("Model", value=DEFAULT_MODEL, key="model_name")
   
    st.subheader("Generation parameters")
    st.slider(
        "temperature",
        0.0,
        1.5,
        0.7,
        0.05,
        key="temperature",
    )
    st.slider(
        "Evidence rows per state",
        1,
        20,
        8,
        1,
        key="evidence_topQ",
        help="Number of selected current-state and target-state evidence rows supplied to the LLM.",
    )

    with st.expander(
        "Evidence selection settings",
        expanded=False,
    ):
        st.number_input(
            "Candidate pool multiplier",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="candidate_pool_multiplier",
            help="Candidate-pool size equals Top-K multiplied by this value.",
        )
        st.slider(
            "Score-proximity weight",
            0.0,
            1.0,
            0.35,
            0.05,
            key="score_weight",
        )
        st.slider(
            "Textual-informativeness weight",
            0.0,
            1.0,
            0.20,
            0.05,
            key="informativeness_weight",
        )
        st.slider(
            "Semantic-representativeness weight",
            0.0,
            1.0,
            0.25,
            0.05,
            key="representativeness_weight",
        )
        st.slider(
            "Semantic-neighborhood consistency weight",
            0.0,
            1.0,
            0.15,
            0.05,
            key="consistency_weight",
        )
        st.slider(
            "Role-direction preference weight",
            0.0,
            1.0,
            0.05,
            0.05,
            key="direction_weight",
        )
        st.slider(
            "MMR redundancy penalty",
            0.0,
            0.50,
            0.10,
            0.01,
            key="diversity_penalty",
        )
        st.number_input(
            "Minimum effective-token threshold",
            min_value=1,
            max_value=50,
            value=8,
            step=1,
            key="min_token_threshold",
        )
        st.number_input(
            "Semantic-neighbor count q",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="neighbor_q",
        )
        st.number_input(
            "Consistency tolerance eta",
            min_value=0.10,
            max_value=10.0,
            value=2.0,
            step=0.10,
            key="consistency_eta",
        )
        st.number_input(
            "Direction tolerance delta",
            min_value=0.0,
            max_value=5.0,
            value=0.25,
            step=0.05,
            key="direction_delta",
        )
        st.number_input(
            "Direction decay tau",
            min_value=0.05,
            max_value=10.0,
            value=1.0,
            step=0.05,
            key="direction_tau",
        )
        st.text_input(
            "Sentence Transformer model",
            value="sentence-transformers/all-MiniLM-L6-v2",
            key="sentence_model_path",
        )
        st.caption(
            "Sentence Transformer embeddings are used for semantic representativeness, semantic-neighborhood consistency, and MMR redundancy. TF-IDF/IDF is retained only for textual informativeness."
        )

    st.subheader("Process parameters")
    st.number_input(
        "Minimum improvement threshold",
        min_value=0.0,
        max_value=2.0,
        value=0.50,
        step=0.01,
        key="delta_threshold",
    )
    st.number_input(
        "Maximum number of service elements processed",
        min_value=1,
        max_value=300,
        value=20,
        step=1,
        key="max_generate",
    )

    if st.button("Test API connection", use_container_width=True):
        try:
            #dummy_history = pd.DataFrame(columns=["Aspect", "Sentiment_Score", "AOP_Text", "Style_Anchor"])
            dummy_history = pd.DataFrame(columns=["Aspect", "Sentiment_Score", "sentence"])
            interpreter = make_interpreter_from_sidebar(
                dummy_history,
                load_similarity=False,
            )
            resp = interpreter.call_llm("Reply a json：{\"ok\": true, \"model\": \"qwen-plus\"}", system_role="You are a JSON-only assistant.")
            st.success("API called successfully")
            st.code(resp, language="json")
        except Exception as exc:
            st.error(str(exc))

st.divider()

# -------------------------
# 1. 数据读取
# -------------------------
st.header("1. Load data")
#st.write(f"当前数据目录：`{DATA_DIR}`")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

excel_files = discover_excel_files(DATA_DIR)
if not excel_files:
    st.warning("Data file does not exist.")
else:
    file_names = [p.name for p in excel_files]
    default_history_idx = file_names.index("cluster_aop.xlsx") if "cluster_aop.xlsx" in file_names else 0
    default_per_idx = file_names.index("MOO_res.xlsx") if "MOO_res.xlsx" in file_names else min(1, len(file_names) - 1)

    col1, col2, col3 = st.columns([1.2, 1.2, 0.8])
    with col1:
        history_filename = st.selectbox("Historical review evidence", file_names, index=default_history_idx)
        history_sheet = st.text_input("Sheet name", value="")
    with col2:
        per_filename = st.selectbox("Performance and optimization results", file_names, index=default_per_idx)
        per_sheet = st.text_input("Sheet name", value="MOO_result")
    with col3:
        st.write("")
        st.write("")
        read_data = st.button("Read and check the data", type="primary", use_container_width=True)

    if read_data:
        try:
            raw_history = read_excel_from_data(DATA_DIR, history_filename, sheet_name=history_sheet.strip() or None)
            df_history_ready = preprocess_history(raw_history)
            df_per = read_excel_from_data(DATA_DIR, per_filename, sheet_name=per_sheet.strip() or None)
            options = build_options(df_per)
            per_dict = build_per_dict(df_per)
            candidate_clusters = build_candidate_clusters(
                df_per,
                float(st.session_state.delta_threshold),
                options=options,
            )

            st.session_state.df_history_ready = df_history_ready
            st.session_state.df_per = df_per
            st.session_state.options = options
            st.session_state.per_dict = per_dict
            st.session_state.candidate_clusters = candidate_clusters
            st.session_state.selected_clusters = candidate_clusters[: int(st.session_state.max_generate)]
            reset_after_data_change()
            st.success(f"Data loading complete: {len(df_history_ready)} historical review records; {len(candidate_clusters)} candidate elements for improvement.")
        except Exception as exc:
            st.exception(exc)
            

if st.session_state.df_history_ready is not None and st.session_state.df_per is not None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Historical review evidence", len(st.session_state.df_history_ready))
    m2.metric("performance record", len(st.session_state.df_per))
    m3.metric("Number of candidate elements", len(st.session_state.candidate_clusters))

    with st.expander("Preview of candidate improvement elements", expanded=False):
        preview_cols = [c for c in ["ES", "Sub_Issue", "Per_focus", "pareto", "Imp", "Eff", "Per_market", "type"] if c in st.session_state.df_per.columns]
        candidate_df = st.session_state.df_per[st.session_state.df_per["ES"].astype(str).isin(st.session_state.candidate_clusters)][preview_cols]
        st.dataframe(candidate_df, use_container_width=True)

# -------------------------
# 2. 生成方案
# -------------------------
st.header("2. Generate proposals")

if st.session_state.df_history_ready is None:
    st.info("Please complete the data loading first.")
else:
    candidate_options = st.session_state.candidate_clusters
    label_map = {}
    for cluster_id in candidate_options:
        row = st.session_state.df_per.loc[st.session_state.df_per["ES"].astype(str) == str(cluster_id)]
        if not row.empty:
            label_map[cluster_id] = f"{cluster_id} | {row['Sub_Issue'].iloc[0]} | delta={float(row['pareto'].iloc[0]):.4f}"
        else:
            label_map[cluster_id] = str(cluster_id)

    selected_clusters = st.multiselect(
        "Select the service elements that need to generate the strategy",
        options=candidate_options,
        default=st.session_state.selected_clusters or candidate_options[: int(st.session_state.max_generate)],
        format_func=lambda x: label_map.get(x, str(x)),
    )

    col_gen1, col_gen2 = st.columns([1, 1])
    with col_gen1:
        generate_button = st.button("Retrieve current-target evidence and generate gap-based proposals", type="primary", use_container_width=True)
    with col_gen2:
        if st.button("Clear generated proposals", use_container_width=True):
            st.session_state.proposals_for_solution = []
            st.session_state.final_proposals = []
            st.session_state.final_result_df = None
            st.session_state.global_check_result = None
            st.session_state.drop_reason_counter = {}
            st.success("Cleared.")

    if generate_button:
        if not st.session_state.api_key_input:
            st.error("Enter the API key first")
        elif not selected_clusters:
            st.error("Select at least one service element")
        else:
            try:
                interpreter = make_interpreter_from_sidebar(st.session_state.df_history_ready)
                proposals: List[Dict[str, Any]] = []
                drop_counter: Counter = Counter()

                clusters_to_run = selected_clusters[: int(st.session_state.max_generate)]
                progress = st.progress(0)
                status = st.empty()
                live_table = st.empty()

                for i, cluster_id in enumerate(clusters_to_run, start=1):
                    status.info(f"Generating {i}/{len(clusters_to_run)}：{label_map.get(cluster_id, cluster_id)}")
                    proposal, drop_reason = generate_one_proposal(
                        interpreter=interpreter,
                        cluster_id=str(cluster_id),
                        df_per=st.session_state.df_per,
                        per_dict=st.session_state.per_dict,
                        options=st.session_state.options,
                        evidence_topk=int(
                            st.session_state.get(
                                "evidence_topQ",
                                8,
                            )
                        ),
                        candidate_pool_multiplier=int(
                            st.session_state.get(
                                "candidate_pool_multiplier",
                                5,
                            )
                        ),
                        score_weight=float(
                            st.session_state.get(
                                "score_weight",
                                0.35,
                            )
                        ),
                        informativeness_weight=float(
                            st.session_state.get(
                                "informativeness_weight",
                                0.20,
                            )
                        ),
                        representativeness_weight=float(
                            st.session_state.get(
                                "representativeness_weight",
                                0.25,
                            )
                        ),
                        consistency_weight=float(
                            st.session_state.get(
                                "consistency_weight",
                                0.15,
                            )
                        ),
                        direction_weight=float(
                            st.session_state.get(
                                "direction_weight",
                                0.05,
                            )
                        ),
                        diversity_penalty=float(
                            st.session_state.get(
                                "diversity_penalty",
                                0.10,
                            )
                        ),
                        min_token_threshold=int(
                            st.session_state.get(
                                "min_token_threshold",
                                8,
                            )
                        ),
                        neighbor_q=int(
                            st.session_state.get(
                                "neighbor_q",
                                10,
                            )
                        ),
                        consistency_eta=float(
                            st.session_state.get(
                                "consistency_eta",
                                2.0,
                            )
                        ),
                        direction_delta=float(
                            st.session_state.get(
                                "direction_delta",
                                0.25,
                            )
                        ),
                        direction_tau=float(
                            st.session_state.get(
                                "direction_tau",
                                1.0,
                            )
                        ),
                    )
                    if proposal is not None:
                        proposals.append(proposal)
                        
                        preview_cols = [c for c in ["cluster_id", "aspect", "current_score", "target_score", "current_state", "target_state", "experience_gap", "action_plan", "priority"] if c in pd.DataFrame(proposals).columns]
                        live_table.dataframe(pd.DataFrame(proposals)[preview_cols], use_container_width=True)
                    else:
                        drop_counter[drop_reason or "unknown"] += 1
                    progress.progress(i / len(clusters_to_run))

                st.session_state.proposals_for_solution = proposals
                st.session_state.final_proposals = proposals.copy()
                st.session_state.drop_reason_counter = dict(drop_counter)
                st.session_state.global_check_result = None
                st.session_state.final_result_df = None
                st.session_state.api_usage = interpreter.api_usage
                status.success(f"Generation completed：pass {len(proposals)}；discard {sum(drop_counter.values())}.")
            except Exception as exc:
                st.exception(exc)

if st.session_state.proposals_for_solution:
    st.subheader("Proposals generated!")
    proposal_df = pd.DataFrame(st.session_state.proposals_for_solution)
    compact_cols = [c for c in ["cluster_id", "aspect", "current_score", "target_score", "current_state", "target_state", "experience_gap", "action_plan", "Resource_Type", "implementation_time", "priority"] if c in proposal_df.columns]
    st.dataframe(proposal_df[compact_cols], use_container_width=True)
    with st.expander("View the details of evidence", expanded=False):
        evidence_cols = [
            c
            for c in [
                "cluster_id",
                "aspect",
                "semantic_representation",
                "closest_current_review_sentence",
                "closest_current_review_sentiment",
                "closest_current_review_score_diff",
                "closest_target_review_sentence",
                "closest_target_review_sentiment",
                "closest_target_review_score_diff",
                "current_evidence_score_proximity",
                "target_evidence_score_proximity",
                "current_evidence_length_sufficiency",
                "target_evidence_length_sufficiency",
                "current_evidence_term_informativeness",
                "target_evidence_term_informativeness",
                "current_evidence_textual_informativeness",
                "target_evidence_textual_informativeness",
                "current_evidence_semantic_representativeness",
                "target_evidence_semantic_representativeness",
                "current_evidence_neighbor_sentiments",
                "target_evidence_neighbor_sentiments",
                "current_evidence_semantic_consistency",
                "target_evidence_semantic_consistency",
                "current_evidence_direction_preference",
                "target_evidence_direction_preference",
                "current_evidence_redundancy_penalties",
                "target_evidence_redundancy_penalties",
                "current_evidence_selection_scores",
                "target_evidence_selection_scores",
                "evidence_selection_method",
                "current_review_evidence",
                "target_review_evidence",
                "evidence_mapping",
            ]
            if c in proposal_df.columns
        ]
        st.dataframe(
            proposal_df[evidence_cols],
            use_container_width=True,
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Proposals", len(st.session_state.proposals_for_solution))
    c2.metric("Dropped", sum(st.session_state.drop_reason_counter.values()) if st.session_state.drop_reason_counter else 0)
    c3.metric("API Calls", st.session_state.api_usage.get("calls", "N/A") if st.session_state.api_usage else "N/A")
    c4.metric("Total Tokens", st.session_state.api_usage.get("total_tokens", "N/A") if st.session_state.api_usage else "N/A")
    if st.session_state.drop_reason_counter:
        st.caption(f"Drop reasons: {st.session_state.drop_reason_counter}")

# -------------------------
# 3. 全局冲突检查
# -------------------------
st.header("3. Global consistency check and conflict handling")

if not st.session_state.proposals_for_solution:
    st.info("Generate proposals first")
else:
    if st.button("Performing global consistency check", use_container_width=True):
        try:
            interpreter = make_interpreter_from_sidebar(
                st.session_state.df_history_ready,
                load_similarity=False,
            )
            with st.spinner("Checking for direct physical or functional conflicts"):
                global_check_result = interpreter.check_global_coherence(st.session_state.proposals_for_solution)
            st.session_state.global_check_result = global_check_result
            st.session_state.api_usage = interpreter.api_usage
            if global_check_result.get("is_feasible", True):
                st.success("Global consistency check passed!")
                st.session_state.final_proposals = st.session_state.proposals_for_solution.copy()
            else:
                st.warning(f"Found {len(global_check_result.get('conflicts', []))} conflicts requiring manual review or automatic resolution.")
        except Exception as exc:
            st.exception(exc)

if st.session_state.global_check_result:
    st.json(st.session_state.global_check_result)
    conflicts = st.session_state.global_check_result.get("conflicts", [])
    if conflicts and not st.session_state.global_check_result.get("is_feasible", True):
        st.subheader("Conflict review workbench")
        for idx, conflict in enumerate(conflicts):
            with st.expander(f"Conflict {idx + 1}: {', '.join(conflict.get('aspects', []))}", expanded=True):
                st.write("**Reason for the conflict：**", conflict.get("reason", "N/A"))
                current_proposals = [p for p in st.session_state.proposals_for_solution if p.get("aspect") in conflict.get("aspects", [])]
                st.dataframe(pd.DataFrame(current_proposals), use_container_width=True)

                choice = st.radio(
                    "Handling method",
                    ["ignore", "auto", "guide", "deactivate", "manual_edit"],
                    format_func=lambda x: {
                        "ignore": "Ignore: not an actual conflict",
                        "auto": "Let LLM resolve automatically",
                        "guide": "Provide manual guidance",
                        "deactivate": "Disable certain proposals manually",
                        "manual_edit": "Edit proposals manually",
                    }[x],
                    key=f"conflict_choice_{idx}",
                )
                guide_text = st.text_area("Manual guidance", key=f"guide_{idx}") if choice == "guide" else ""
                deactivate_aspects = st.multiselect("Select the proposals to disable", options=conflict.get("aspects", []), key=f"deact_{idx}") if choice == "deactivate" else []
                manual_json = st.text_area("Paste the updated proposals JSON array", key=f"manual_json_{idx}") if choice == "manual_edit" else ""

                if st.button(f"Resolving conflict {idx + 1}", key=f"apply_conflict_{idx}"):
                    try:
                        manual_result: Dict[str, Any] = {
                            "decision_type": choice,
                            "guide_text": guide_text,
                            "deactivate_aspects": deactivate_aspects,
                            "manual_updated_proposals": None,
                            "notes": "",
                        }
                        if choice == "manual_edit" and manual_json.strip():
                            manual_result["manual_updated_proposals"] = json.loads(manual_json)

                        if choice == "ignore":
                            st.session_state.final_proposals = st.session_state.final_proposals or st.session_state.proposals_for_solution.copy()
                            st.success("The conflict has been ignored, Maintain current status.")
                        elif choice in ("deactivate", "manual_edit"):
                            proposal_dict = {p["aspect"]: p for p in (st.session_state.final_proposals or st.session_state.proposals_for_solution)}
                            proposal_dict = apply_manual_edits(proposal_dict, manual_result)
                            st.session_state.final_proposals = list(proposal_dict.values())
                            st.success("Manual processing has been applied.")
                        else:
                            interpreter = make_interpreter_from_sidebar(
                                st.session_state.df_history_ready,
                                load_similarity=False,
                            )
                            proposals_base = st.session_state.final_proposals or st.session_state.proposals_for_solution
                            final_proposals = interpreter.resolve_strategic_conflicts(
                                all_proposals=proposals_base,
                                global_conflicts=[conflict],
                                human_guidance=guide_text if choice == "guide" else "",
                            )
                            st.session_state.final_proposals = final_proposals
                            st.session_state.api_usage = interpreter.api_usage
                            st.success("Automatic conflict resolution has been applied.")
                    except Exception as exc:
                        st.exception(exc)

# -------------------------
# 4. 最终人工评审与二次修正
# -------------------------
st.header("4. Manual review and secondary revision")

if not st.session_state.final_proposals:
    st.info("No final proposals are available yet.")
else:
    review_df = pd.DataFrame(st.session_state.final_proposals)
    for col, default in {
        "accepted": True,
        "review_comment": "",
        "relevance": 3.0,
        "specificity": 3.0,
        "feasibility": 3.0,
    }.items():
        if col not in review_df.columns:
            review_df[col] = default

    edited_review = st.data_editor(
        review_df,
        use_container_width=True,
        key="final_review_editor",
        num_rows="dynamic",
    )

    if st.button("Implement the final review and corrective action plan", use_container_width=True):
        try:
            interpreter = make_interpreter_from_sidebar(
                st.session_state.df_history_ready,
                load_similarity=False,
            )
            proposals = edited_review.drop(
                columns=["accepted", "review_comment", "relevance", "specificity", "feasibility"],
                errors="ignore",
            ).to_dict("records")
            review_records = edited_review.to_dict("records")
            proposal_by_aspect = {p.get("aspect"): p for p in proposals if p.get("aspect")}

            for rec in review_records:
                aspect = rec.get("aspect")
                if not aspect or aspect not in proposal_by_aspect:
                    continue
                p = proposal_by_aspect[aspect]
                if not bool(rec.get("accepted", True)):
                    scores = {
                        "relevance": float(rec.get("relevance", 3)),
                        "specificity": float(rec.get("specificity", 3)),
                        "feasibility": float(rec.get("feasibility", 3)),
                    }
                    guidance = ((rec.get("review_comment") or "") + "\n" + map_scores_to_guidance(scores)).strip()
                    old_action = p.get("action_plan", "")
                    new_action = interpreter.refine_action_plan(
                        aspect_name=aspect,
                        action_plan_str=old_action,
                        guidance=guidance,
                        aop=p.get("aop", ""),
                    )
                    revision_history = p.get("action_revision_history")

                    if isinstance(revision_history, str):
                        try:
                            parsed_history = json.loads(revision_history)
                            revision_history = (
                                parsed_history
                                if isinstance(parsed_history, list)
                                else []
                            )
                        except (TypeError, ValueError, json.JSONDecodeError):
                            revision_history = []
                    elif not isinstance(revision_history, list):
                        revision_history = []
                    
                    p["action_revision_history"] = revision_history
                    
                    revision_history.append(
                        {
                            "revision_id": len(revision_history) + 1,
                            "timestamp": pd.Timestamp.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "trigger_type": "human_feedback",
                            "trigger_detail": guidance,
                            "old_action_plan": old_action,
                            "new_action_plan": new_action,
                            "notes": "Refined based on Streamlit final review",
                        }
                    )
                    
                    p["action_plan"] = new_action

            final_proposals = list(proposal_by_aspect.values())
            st.session_state.final_proposals = final_proposals
            st.session_state.final_result_df = finalize_results(final_proposals)
            st.session_state.api_usage = interpreter.api_usage
            st.success("Final review and correction completed!")
        except Exception as exc:
            st.exception(exc)

# -------------------------
# 5. 导出
# -------------------------
st.header("5. Result export")

if st.session_state.final_result_df is None and st.session_state.final_proposals:
    st.session_state.final_result_df = finalize_results(st.session_state.final_proposals)

if st.session_state.final_result_df is not None:
    st.subheader("Final results")
    st.dataframe(st.session_state.final_result_df, use_container_width=True)

    excel_bytes = df_to_excel_bytes(
        st.session_state.final_result_df,
        pd.DataFrame(st.session_state.final_proposals),
    )
    col_down1, col_down2 = st.columns(2)
    with col_down1:
        st.download_button(
            "Download the final strategy file",
            data=excel_bytes,
            file_name="hotel_strategy_final_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col_down2:
        if st.button("Save to ./outputs", use_container_width=True):
            try:
                out_path = save_outputs(
                    OUTPUT_DIR,
                    st.session_state.final_result_df,
                    st.session_state.final_proposals,
                )
                st.session_state.last_saved_path = str(out_path)
                st.success(f"Saved：{out_path}")
            except Exception as exc:
                st.exception(exc)

    if st.session_state.last_saved_path:
        st.caption(f"Recent save path`{st.session_state.last_saved_path}`")
else:
    st.info("A downloadable strategy file will be generated here after completing the previous steps.")

with st.expander("Execution logs / API usage", expanded=False):
    st.json(st.session_state.api_usage or {})
