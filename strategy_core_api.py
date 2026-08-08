# -*- coding: utf-8 -*-
"""
Core services for the Streamlit hotel strategy interaction system.

This version uses an online Qwen-plus API through the OpenAI-compatible
DashScope/Bailian endpoint. It contains no local LLM loading code.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json_repair
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"

DEFAULT_PRIORITY_MAP = {
    4: 1,
    1: 2,
    3: 2,
    8: 2,
    2: 3,
    5: 3,
    7: 4,
    6: 4,
}

REQUIRED_HISTORY_COLUMNS = {"sentence", "Sub_Issue", "sentiment"}
REQUIRED_PER_COLUMNS = {
    "ES",
    "Sub_Issue",
    "Per_focus",
    "pareto",
    "type",
    "Imp",
    "Eff",
    "Per_market",
    "维护性成本",
}


def normalize_json_response(text: str) -> Any:
    """Parse JSON-like LLM output robustly, allowing markdown fences and minor defects."""
    clean = str(text).replace("```json", "").replace("```", "").strip()
    return json_repair.loads(clean)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_identifier(value: Any) -> str:
    """Normalize Excel identifiers such as 1, 1.0, and '1' into comparable strings."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    except Exception:
        return text


# ---------------------------------------------------------------------
# Evidence selection helpers
# ---------------------------------------------------------------------
# AOP fields are used only as localization/display cues. They are not used
# as independent quality scores.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def _tokenize_text(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(str(text or "").lower())]



def _evidence_text_from_record(row_or_rec: Any) -> str:
    """Construct the structured semantic text x_i from sentence, aspect, and opinion."""
    if isinstance(row_or_rec, dict):
        getter = row_or_rec.get
    else:
        getter = (
            row_or_rec.get
            if hasattr(row_or_rec, "get")
            else lambda key, default="": getattr(row_or_rec, key, default)
        )

    sentence = str(getter("sentence", "") or "").strip()
    aspect = str(getter("aspect", getter("Aspect", "")) or "").strip()
    opinion = str(getter("opinion", "") or "").strip()

    parts = [f"Review: {sentence}"]
    if aspect:
        parts.append(f"Aspect: {aspect}")
    if opinion:
        parts.append(f"Opinion: {opinion}")
    return " ".join(parts)


def _informativeness_text_from_record(row_or_rec: Any) -> str:
    """Use the original review sentence alone for textual informativeness."""
    if isinstance(row_or_rec, dict):
        sentence = row_or_rec.get("sentence", "")
    elif hasattr(row_or_rec, "get"):
        sentence = row_or_rec.get("sentence", "")
    else:
        sentence = getattr(row_or_rec, "sentence", "")
    return str(sentence or "").strip()

def effective_token_count(text: str) -> int:
    """Count effective tokens for the length-sufficiency term."""
    return len([t for t in _tokenize_text(text) if len(t) > 1 or t.isdigit()])


def fit_tfidf(texts: List[str]) -> Tuple[TfidfVectorizer, Any]:
    """Fit a robust TF-IDF representation; fall back to character n-grams if needed."""
    cleaned = [str(t or "") for t in texts]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        matrix = vectorizer.fit_transform(cleaned)
        if matrix.shape[1] > 0:
            return vectorizer, matrix
    except Exception:
        pass
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(cleaned)
    return vectorizer, matrix


def idf_log_sum_scores(texts: List[str], vectorizer: TfidfVectorizer) -> np.ndarray:
    """Compute U_i from the normalized log-sum of corpus-specific IDF values."""
    analyzer = vectorizer.build_analyzer()
    vocab = getattr(vectorizer, "vocabulary_", {})
    idf = getattr(vectorizer, "idf_", None)
    if idf is None or not vocab:
        return np.ones(len(texts), dtype=float)
    idf_map = {term: float(idf[idx]) for term, idx in vocab.items()}
    raw = []
    for text in texts:
        terms = set(analyzer(str(text or "")))
        raw.append(float(np.log1p(sum(idf_map.get(t, 0.0) for t in terms))))
    arr = np.asarray(raw, dtype=float)
    max_val = float(np.max(arr)) if len(arr) else 0.0
    if max_val <= 1e-12:
        return np.ones(len(texts), dtype=float)
    return arr / (max_val + 1e-12)


def length_sufficiency_scores(texts: List[str], n0: int = 8) -> np.ndarray:
    vals = []
    threshold = max(1, int(n0))
    for text in texts:
        n = effective_token_count(text)
        vals.append(min(1.0, n / threshold) if n > 0 else 0.0)
    return np.asarray(vals, dtype=float)


def normalize_score_series(values: pd.Series, higher_is_better: bool = False) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if vals.empty:
        return vals
    vmin = vals.min()
    vmax = vals.max()
    if abs(vmax - vmin) < 1e-12:
        return pd.Series([1.0] * len(vals), index=vals.index)
    norm = (vals - vmin) / (vmax - vmin)
    return norm if higher_is_better else 1.0 - norm


def role_direction_preference(sentiment: float, anchor_score: float, evidence_role: str, delta_s: float = 0.25, tau: float = 1.0) -> float:
    """Score-side role alignment H_i^z."""
    s = safe_float(sentiment)
    a = safe_float(anchor_score)
    tau = max(float(tau), 1e-6)
    if evidence_role == "current":
        penalty = max(0.0, s - a - float(delta_s))
    else:
        penalty = max(0.0, a - s - float(delta_s))
    return float(np.exp(-penalty / tau))


def normalize_component(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if abs(hi - lo) < 1e-12:
        return np.ones_like(arr, dtype=float)
    return (arr - lo) / (hi - lo + 1e-12)


def normalized_weights(**weights: float) -> Dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 1e-12:
        return {k: 1.0 / len(clipped) for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def clean_action_plan(action_plan: Dict[str, Any]) -> Tuple[str, str, Any, Dict[str, Any]]:
    """Convert LLM action JSON into displayable fields."""
    if isinstance(action_plan, dict) and "raw_text" in action_plan:
        try:
            action_plan = normalize_json_response(action_plan["raw_text"])
        except Exception:
            pass

    actions = action_plan.get("suggested_actions", []) if isinstance(action_plan, dict) else []
    cleaned_actions: List[str] = []

    if isinstance(actions, list):
        for item in actions:
            if isinstance(item, str):
                cleaned_actions.append(item)
            elif isinstance(item, dict):
                action_text = item.get("action") or item.get("description") or item.get("text")
                cleaned_actions.append(str(action_text) if action_text else f"Structured Action: {item}")
            else:
                cleaned_actions.append(str(item))
        actions_str = "; ".join(cleaned_actions)
    else:
        actions_str = str(actions)

    resource_type = action_plan.get("resource_type", "Unknown") if isinstance(action_plan, dict) else "Unknown"
    implementation_time = (
        action_plan.get(
            "implementation_time",
            action_plan.get("estimated_time_days", "Unknown"),
        )
        if isinstance(action_plan, dict)
        else "Unknown"
    )
    return actions_str, resource_type, implementation_time, action_plan if isinstance(action_plan, dict) else {}


def preprocess_history(df_history: pd.DataFrame) -> pd.DataFrame:
    """Prepare historical review evidence for ES/aspect-level target-state retrieval.

    The concise version no longer requires manually constructed aspect-opinion anchors.
    Required columns are only: sentence, Sub_Issue, sentiment. Optional columns such as
    aspect, opinion, ES, Cluster, etc. are preserved for matching and evidence display.
    """
    missing = REQUIRED_HISTORY_COLUMNS - set(df_history.columns)
    if missing:
        raise ValueError(f"历史评论证据文件缺少列: {sorted(missing)}")

    df = df_history.copy()

    # Normalize core columns used by the retrieval functions while preserving original columns.
    if "Aspect" not in df.columns:
        df["Aspect"] = df["Sub_Issue"]
    if "Sentiment_Score" not in df.columns:
        df["Sentiment_Score"] = pd.to_numeric(df["sentiment"], errors="coerce")
    else:
        df["Sentiment_Score"] = pd.to_numeric(df["Sentiment_Score"], errors="coerce")

    if "opinion" not in df.columns:
        df["opinion"] = ""
    if "aspect" not in df.columns:
        df["aspect"] = df["Aspect"]

    # Kept only for backward-compatible optional checks; Step 1 does not use these as primary evidence.
    df["AOP_Text"] = df.apply(
        lambda x: f"Aspect [{x.get('aspect', x.get('Aspect', ''))}]: {x.get('opinion', '')}".strip(),
        axis=1,
    )
    df["Style_Anchor"] = df["sentence"].astype(str)

    return df.dropna(subset=["Aspect", "Sentiment_Score", "sentence"])


def get_zone(
    imp: float,
    per: float,
    eff: float,
    per_market_single: float,
    imp_mean: float = 0.00819672131147541,
    eff_ref: float = 0,
) -> int:
    """Map importance/performance/effectiveness indicators into the original 8-zone scheme."""
    zone_x = 1 if imp < imp_mean else 0
    zone_y = 1 if eff < eff_ref else 0
    zone_z = 1 if per < per_market_single else 0

    if zone_x == 0 and zone_z == 0:
        return 1 + 4 * zone_y
    if zone_x == 1 and zone_z == 0:
        return 2 + 4 * zone_y
    if zone_x == 1 and zone_z == 1:
        return 3 + 4 * zone_y
    return 4 + 4 * zone_y


def get_priority(zone: int, category: str, priority_map: Optional[Dict[int, int]] = None) -> int:
    priority_map = priority_map or DEFAULT_PRIORITY_MAP
    return priority_map.get(zone, 4)


def build_options(
    df_per: pd.DataFrame,
    imp_mean: float = 0.00819672131147541,
    eff_ref: float = 0,
) -> List[Dict[str, Any]]:
    """Build option metadata and priority fields from the performance/optimization table."""
    missing = REQUIRED_PER_COLUMNS - set(df_per.columns)
    if missing:
        raise ValueError(f"性能/优化文件缺少列: {sorted(missing)}")

    options: List[Dict[str, Any]] = []
    for _, row in df_per.iterrows():
        category = row["type"]
        if category in ("Excitement", "Must-be"):
            params = tuple(row[c] for c in ["knot", "a", "b", "c"] if c in df_per.columns)
        elif category == "linear":
            params = tuple(row[c] for c in ["a", "b"] if c in df_per.columns)
        else:
            params = ()

        opt = {
            "ES": str(row["ES"]),
            "Imp": safe_float(row["Imp"]),
            "Per": safe_float(row["Per_focus"]),
            "Eff": safe_float(row["Eff"]),
            "Cost": safe_float(row["维护性成本"]),
            "Per_market": safe_float(row["Per_market"]),
            "category": category,
            "params": params,
        }
        opt["zone"] = get_zone(
            opt["Imp"],
            opt["Per"],
            opt["Eff"],
            opt["Per_market"],
            imp_mean=imp_mean,
            eff_ref=eff_ref,
        )
        opt["max_delta"] = 10 - opt["Per"]
        opt["priority"] = get_priority(opt["zone"], opt["category"])
        options.append(opt)
    return options


def build_per_dict(df_per: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["ES"]): {"name": row["Sub_Issue"], "current_score": safe_float(row["Per_focus"])}
        for _, row in df_per.iterrows()
    }


def map_scores_to_guidance(scores: Dict[str, float]) -> str:
    """Convert human review scores into LLM-readable refinement guidance."""
    notes: List[str] = []
    if scores.get("relevance", 3) < 3:
        notes.append("- Make the suggested actions directly address the problem implied by the customer voice.")
    if scores.get("specificity", 3) < 3:
        notes.append("- Make the action plan more specific and clear, with implementation details.")
    if scores.get("feasibility", 3) < 3:
        notes.append("- Carefully consider technical and operational feasibility and avoid large-scale structural modifications.")
    if not notes:
        return "No major issues detected in manual scores."
    return "Improvement suggestions based on manual scoring:\n" + "\n".join(notes)


def apply_manual_edits(proposal_dict: Dict[str, Dict[str, Any]], manual_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Apply ignore/deactivate/manual-edit decisions outside the LLM conflict solver."""
    decision = manual_result.get("decision_type")
    if decision in ("auto", "guide", "ignore"):
        return proposal_dict
    if decision == "deactivate":
        for aspect in manual_result.get("deactivate_aspects", []):
            proposal_dict.pop(aspect, None)
        return proposal_dict
    if decision == "manual_edit":
        updated = manual_result.get("manual_updated_proposals") or []
        for item in updated:
            aspect = item.get("aspect")
            if aspect:
                proposal_dict[aspect] = item
        return proposal_dict
    return proposal_dict


@dataclass
class ApiUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, usage: Any) -> None:
        self.calls += 1
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)

    def as_dict(self) -> Dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class QwenPlusChatClient:
    """Small wrapper around the OpenAI-compatible DashScope Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        timeout: float = 120.0,
        request_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("Enter the API Key first")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.request_retries = request_retries
        self.usage = ApiUsage()

    def chat(self, prompt: str, system_role: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system_role:
            messages.append({"role": "system", "content": system_role})
        messages.append({"role": "user", "content": prompt})

        last_error: Optional[Exception] = None
        for attempt in range(self.request_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                self.usage.add(getattr(response, "usage", None))
                content = response.choices[0].message.content
                return str(content or "").strip()
            except Exception as exc:  # pragma: no cover - network/API dependent
                last_error = exc
                if attempt >= self.request_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"API call failed: {last_error}")


def tfidf_max_similarity(generated_text: str, reference_examples: List[str]) -> float:
    """Lightweight semantic proxy; avoids local embedding model dependencies."""
    if not generated_text or not reference_examples:
        return 1.0
    corpus = [generated_text] + [str(x) for x in reference_examples if str(x).strip()]
    if len(corpus) <= 1:
        return 1.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        mat = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
        return float(np.max(sims)) if len(sims) else 1.0
    except ValueError:
        return 1.0



@lru_cache(maxsize=4)
def load_sentence_transformer(model_name_or_path: str):
    """Load and reuse a Sentence Transformer model across Streamlit reruns."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name_or_path)


class SentenceTransformerSimilarity:
    """Sentence-level semantic encoder based on Sentence Transformers."""

    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path
        try:
            self.embed_model = load_sentence_transformer(model_name_or_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Sentence Transformer model '{model_name_or_path}': {exc}"
            ) from exc

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode texts into L2-normalized dense semantic vectors."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self.embed_model.encode(
            texts,
            batch_size=max(1, int(batch_size)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def max_similarity(
        self,
        generated_text: str,
        reference_examples: List[str],
    ) -> float:
        if not generated_text or not reference_examples:
            return 1.0

        embeddings = self.encode([generated_text] + reference_examples)
        generated_embedding = embeddings[0:1]
        reference_embeddings = embeddings[1:]
        similarities = generated_embedding @ reference_embeddings.T
        return float(np.max(similarities)) if similarities.size else 1.0

    
    
class ServiceStrategyInterpreter:
    """Evidence retrieval, gap diagnosis, action generation, and conflict resolution."""

    def __init__(
        self,
        chat_client: QwenPlusChatClient,
        history_df: Optional[pd.DataFrame] = None,
        max_retries: int = 3,
        similarity_backend: str = "sentence-transformers",
        sentence_model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.history_df = history_df.copy() if history_df is not None else None
        self.chat_client = chat_client
        self.max_retries = max_retries
        self.similarity_backend = str(
            similarity_backend or "sentence-transformers"
        ).strip().lower()
        self.sentence_model_path = sentence_model_path
        self.sentence_model: Optional[SentenceTransformerSimilarity] = None
        self._evidence_space_cache: Dict[
            Tuple[str, str, str, str],
            Dict[str, Any],
        ] = {}

        if self.similarity_backend == "sentence-transformers":
            self.sentence_model = SentenceTransformerSimilarity(
                sentence_model_path
            )

    @property
    def api_usage(self) -> Dict[str, int]:
        return self.chat_client.usage.as_dict()

    def call_llm(self, prompt: str, system_role: Optional[str] = None) -> str:
        return self.chat_client.chat(prompt=prompt, system_role=system_role)

    def _max_similarity(self, generated_text: str, reference_examples: Optional[List[str]]) -> Optional[float]:
        if not reference_examples:
            return None
        if self.similarity_backend == "none":
            return None
        if self.similarity_backend == "sentence-transformers" and self.sentence_model is not None:
            return self.sentence_model.max_similarity(generated_text, reference_examples)
        return tfidf_max_similarity(generated_text, reference_examples)


    def _matched_history_subset(self, cluster_id: str, aspect_name: str) -> pd.DataFrame:
        """Match historical review rows by ES/Cluster first, then by aspect name.

        This helper is shared by current-level and target-level retrieval so that
        both evidence sets are drawn from the same service element whenever ES IDs
        are available.
        """
        if self.history_df is None or self.history_df.empty:
            return pd.DataFrame()

        df = self.history_df.copy()
        if "Sentiment_Score" not in df.columns or "sentence" not in df.columns:
            return pd.DataFrame()

        df["Sentiment_Score"] = pd.to_numeric(df["Sentiment_Score"], errors="coerce")
        df = df.dropna(subset=["Sentiment_Score", "sentence"])
        df = df[df["sentence"].astype(str).str.strip().ne("")]
        if df.empty:
            return pd.DataFrame()

        target_id = normalize_identifier(cluster_id)
        id_columns = [
            c
            for c in ["ES", "es", "Cluster", "cluster", "cluster_id", "Cluster_ID", "ES_ID", "Issue_ID"]
            if c in df.columns
        ]
        for col in id_columns:
            matched = df[df[col].apply(normalize_identifier) == target_id].copy()
            if not matched.empty:
                return matched

        aspect_norm = str(aspect_name).strip().lower()
        if "Aspect" in df.columns:
            matched = df[df["Aspect"].astype(str).str.strip().str.lower() == aspect_norm].copy()
            if not matched.empty:
                return matched

        if "Sub_Issue" in df.columns:
            matched = df[df["Sub_Issue"].astype(str).str.strip().str.lower() == aspect_norm].copy()
            if not matched.empty:
                return matched

        return pd.DataFrame()


    def _prepare_evidence_space(
        self,
        cluster_id: str,
        aspect_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Prepare reusable representations for one service element."""
        cache_key = (
            normalize_identifier(cluster_id),
            str(aspect_name).strip().lower(),
            self.similarity_backend,
            str(self.sentence_model_path),
        )
        cached = self._evidence_space_cache.get(cache_key)
        if cached is not None:
            return cached

        subset = self._matched_history_subset(
            cluster_id=cluster_id,
            aspect_name=aspect_name,
        )
        if subset.empty:
            return None

        subset = subset.copy().reset_index(drop=True)
        subset["Sentiment_Score"] = pd.to_numeric(
            subset["Sentiment_Score"],
            errors="coerce",
        )
        subset = subset.dropna(
            subset=["Sentiment_Score", "sentence"]
        )
        subset = subset[
            subset["sentence"].astype(str).str.strip().ne("")
        ]
        subset = subset.drop_duplicates(
            subset=["sentence"]
        ).reset_index(drop=True)
        if subset.empty:
            return None

        subset["row_pos"] = np.arange(len(subset), dtype=int)
        subset["semantic_text"] = subset.apply(
            _evidence_text_from_record,
            axis=1,
        )
        subset["informativeness_text"] = subset.apply(
            _informativeness_text_from_record,
            axis=1,
        )

        all_info_texts = (
            subset["informativeness_text"].astype(str).tolist()
        )
        idf_vectorizer, tfidf_matrix = fit_tfidf(
            all_info_texts
        )

        if (
            self.similarity_backend == "sentence-transformers"
            and self.sentence_model is not None
        ):
            all_semantic_texts = (
                subset["semantic_text"].astype(str).tolist()
            )
            all_semantic_matrix = self.sentence_model.encode(
                all_semantic_texts
            )
            representation_method = (
                f"sentence-transformers:{self.sentence_model_path}"
            )
        else:
            all_semantic_matrix = tfidf_matrix
            representation_method = "tfidf"

        prepared = {
            "subset": subset,
            "idf_vectorizer": idf_vectorizer,
            "all_semantic_matrix": all_semantic_matrix,
            "representation_method": representation_method,
        }
        self._evidence_space_cache[cache_key] = prepared
        return prepared

    def retrieve_reviews_by_score(
        self,
        cluster_id: str,
        aspect_name: str,
        anchor_score: float,
        topk: int = 8,
        evidence_role: str = "target",
        candidate_pool_multiplier: int = 5,
        score_weight: float = 0.40,
        informativeness_weight: float = 0.25,
        representativeness_weight: float = 0.15,
        consistency_weight: float = 0.15,
        direction_weight: float = 0.05,
        diversity_penalty: float = 0.10,
        min_token_threshold: int = 8,
        neighbor_q: int = 10,
        consistency_eta: float = 2.0,
        direction_delta: float = 0.25,
        direction_tau: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Select current/target evidence using composite non-LLM scoring."""
        if evidence_role not in {"current", "target"}:
            raise ValueError(
                "evidence_role must be either 'current' or 'target'."
            )

        prepared = self._prepare_evidence_space(
            cluster_id=cluster_id,
            aspect_name=aspect_name,
        )
        if prepared is None:
            return []

        subset = prepared["subset"].copy()
        idf_vectorizer = prepared["idf_vectorizer"]
        all_semantic_matrix = prepared["all_semantic_matrix"]
        representation_method = str(
            prepared["representation_method"]
        )

        subset["score_diff"] = (
            subset["Sentiment_Score"].astype(float)
            - float(anchor_score)
        ).abs()

        pool_size = max(
            max(1, int(topk)),
            max(1, int(topk))
            * max(1, int(candidate_pool_multiplier)),
        )
        candidates = (
            subset.sort_values(
                ["score_diff", "row_pos"],
                ascending=[True, True],
                kind="mergesort",
            )
            .head(pool_size)
            .copy()
        )
        if candidates.empty:
            return []

        cand_pos = candidates["row_pos"].astype(int).to_numpy()
        cand_semantic_matrix = all_semantic_matrix[cand_pos]
        candidate_info_texts = (
            candidates["informativeness_text"]
            .astype(str)
            .tolist()
        )

        candidates["score_proximity"] = normalize_score_series(
            candidates["score_diff"],
            higher_is_better=False,
        )

        length_vals = length_sufficiency_scores(
            candidate_info_texts,
            n0=min_token_threshold,
        )
        idf_vals = idf_log_sum_scores(
            candidate_info_texts,
            idf_vectorizer,
        )
        info_vals = np.clip(
            length_vals * idf_vals,
            0.0,
            1.0,
        )
        candidates["length_sufficiency"] = length_vals
        candidates["term_informativeness"] = idf_vals
        candidates["textual_informativeness"] = info_vals

        cand_sim = np.asarray(
            cosine_similarity(
                cand_semantic_matrix,
                cand_semantic_matrix,
            ),
            dtype=float,
        )
        cand_sim = np.clip(cand_sim, 0.0, 1.0)

        if len(candidates) > 1:
            np.fill_diagonal(cand_sim, 0.0)
            weighted_sum = cand_sim.dot(info_vals)
            info_denom = np.sum(info_vals) - info_vals
            weighted_similarity = np.divide(
                weighted_sum,
                info_denom,
                out=np.zeros_like(
                    weighted_sum,
                    dtype=float,
                ),
                where=info_denom > 1e-12,
            )
            rep_vals = np.clip(
                info_vals * weighted_similarity,
                0.0,
                1.0,
            )
        else:
            np.fill_diagonal(cand_sim, 0.0)
            rep_vals = info_vals.copy()
        candidates["semantic_representativeness"] = rep_vals

        all_sim_for_candidates = np.asarray(
            cosine_similarity(
                cand_semantic_matrix,
                all_semantic_matrix,
            ),
            dtype=float,
        )
        all_sim_for_candidates = np.clip(
            all_sim_for_candidates,
            0.0,
            1.0,
        )

        all_sentiments = (
            subset["Sentiment_Score"].astype(float).to_numpy()
        )
        q = max(1, int(neighbor_q))
        eta = max(float(consistency_eta), 1e-6)
        neighbor_estimates: List[float] = []
        consistency_vals: List[float] = []

        for local_i, global_pos in enumerate(cand_pos):
            sims = all_sim_for_candidates[local_i].copy()
            if 0 <= global_pos < len(sims):
                sims[global_pos] = 0.0

            valid = np.where(sims > 0.0)[0]
            s_i = float(
                candidates.iloc[local_i]["Sentiment_Score"]
            )
            if valid.size == 0:
                neighbor_estimates.append(s_i)
                consistency_vals.append(1.0)
                continue

            neighbor_count = min(q, int(valid.size))
            top_idx = valid[
                np.argsort(
                    sims[valid],
                    kind="mergesort",
                )[-neighbor_count:]
            ]
            neighbor_weights = sims[top_idx]
            shat = float(
                np.sum(
                    neighbor_weights
                    * all_sentiments[top_idx]
                )
                / (
                    np.sum(neighbor_weights)
                    + 1e-12
                )
            )
            c_i = 1.0 - min(
                abs(s_i - shat) / eta,
                1.0,
            )
            neighbor_estimates.append(shat)
            consistency_vals.append(
                float(np.clip(c_i, 0.0, 1.0))
            )

        candidates["semantic_neighbor_sentiment"] = (
            neighbor_estimates
        )
        candidates["semantic_neighborhood_consistency"] = (
            consistency_vals
        )

        candidates["role_direction_preference"] = candidates[
            "Sentiment_Score"
        ].apply(
            lambda s: role_direction_preference(
                sentiment=float(s),
                anchor_score=float(anchor_score),
                evidence_role=evidence_role,
                delta_s=float(direction_delta),
                tau=float(direction_tau),
            )
        )

        weights = normalized_weights(
            score=float(score_weight),
            info=float(informativeness_weight),
            represent=float(representativeness_weight),
            consistency=float(consistency_weight),
            direction=float(direction_weight),
        )
        candidates["base_evidence_score"] = (
            weights["score"]
            * candidates["score_proximity"].astype(float)
            + weights["info"]
            * candidates["textual_informativeness"].astype(float)
            + weights["represent"]
            * candidates[
                "semantic_representativeness"
            ].astype(float)
            + weights["consistency"]
            * candidates[
                "semantic_neighborhood_consistency"
            ].astype(float)
            + weights["direction"]
            * candidates[
                "role_direction_preference"
            ].astype(float)
        )

        selected_indices: List[Any] = []
        selected_local_positions: List[int] = []
        remaining = list(candidates.index)
        index_to_local = {
            idx: pos
            for pos, idx in enumerate(candidates.index)
        }

        while (
            remaining
            and len(selected_indices) < max(1, int(topk))
        ):
            best_idx: Optional[Any] = None
            best_key: Optional[
                Tuple[float, float, float, int]
            ] = None
            best_redundancy = 0.0

            for idx in remaining:
                local_pos = index_to_local[idx]
                redundancy = (
                    float(
                        np.max(
                            cand_sim[
                                local_pos,
                                selected_local_positions,
                            ]
                        )
                    )
                    if selected_local_positions
                    else 0.0
                )
                final_score = (
                    float(
                        candidates.at[
                            idx,
                            "base_evidence_score",
                        ]
                    )
                    - float(diversity_penalty)
                    * redundancy
                )
                tie_key = (
                    final_score,
                    float(
                        candidates.at[
                            idx,
                            "base_evidence_score",
                        ]
                    ),
                    -float(
                        candidates.at[idx, "score_diff"]
                    ),
                    -int(candidates.at[idx, "row_pos"]),
                )
                if best_key is None or tie_key > best_key:
                    best_key = tie_key
                    best_idx = idx
                    best_redundancy = redundancy

            if best_idx is None or best_key is None:
                break

            candidates.at[
                best_idx,
                "redundancy_score",
            ] = best_redundancy
            candidates.at[
                best_idx,
                "redundancy_penalty",
            ] = (
                float(diversity_penalty)
                * best_redundancy
            )
            candidates.at[
                best_idx,
                "selection_score",
            ] = best_key[0]
            selected_indices.append(best_idx)
            selected_local_positions.append(
                index_to_local[best_idx]
            )
            remaining.remove(best_idx)

        selected = candidates.loc[selected_indices].copy()
        for col in [
            "redundancy_score",
            "redundancy_penalty",
            "selection_score",
        ]:
            if col not in selected.columns:
                selected[col] = 0.0
            selected[col] = pd.to_numeric(
                selected[col],
                errors="coerce",
            ).fillna(0.0)

        optional_cols = [
            c
            for c in [
                "Sentiment_Score",
                "score_diff",
                "score_proximity",
                "length_sufficiency",
                "term_informativeness",
                "textual_informativeness",
                "semantic_representativeness",
                "semantic_neighbor_sentiment",
                "semantic_neighborhood_consistency",
                "role_direction_preference",
                "base_evidence_score",
                "redundancy_score",
                "redundancy_penalty",
                "selection_score",
                "sentence",
                "opinion",
                "aspect",
                "Aspect",
                "Sub_Issue",
                "ES",
                "c",
                "AOP_Text",
                "Style_Anchor",
                "semantic_text",
                "informativeness_text",
            ]
            if c in selected.columns
        ]

        records: List[Dict[str, Any]] = []
        for rec in selected[optional_cols].to_dict(
            "records"
        ):
            records.append(
                {
                    "evidence_role": evidence_role,
                    "anchor_score": float(anchor_score),
                    "semantic_representation": (
                        representation_method
                    ),
                    "sentiment_score": safe_float(
                        rec.get("Sentiment_Score")
                    ),
                    "score_diff": safe_float(
                        rec.get("score_diff")
                    ),
                    "score_proximity": safe_float(
                        rec.get("score_proximity")
                    ),
                    "length_sufficiency": safe_float(
                        rec.get("length_sufficiency")
                    ),
                    "term_informativeness": safe_float(
                        rec.get("term_informativeness")
                    ),
                    "textual_informativeness": safe_float(
                        rec.get("textual_informativeness")
                    ),
                    "semantic_representativeness": safe_float(
                        rec.get(
                            "semantic_representativeness"
                        )
                    ),
                    "semantic_neighbor_sentiment": safe_float(
                        rec.get(
                            "semantic_neighbor_sentiment"
                        )
                    ),
                    "semantic_neighborhood_consistency": safe_float(
                        rec.get(
                            "semantic_neighborhood_consistency"
                        )
                    ),
                    "role_direction_preference": safe_float(
                        rec.get(
                            "role_direction_preference"
                        )
                    ),
                    "base_evidence_score": safe_float(
                        rec.get("base_evidence_score")
                    ),
                    "redundancy_score": safe_float(
                        rec.get("redundancy_score")
                    ),
                    "redundancy_penalty": safe_float(
                        rec.get("redundancy_penalty")
                    ),
                    "selection_score": safe_float(
                        rec.get("selection_score")
                    ),
                    "sentence": str(
                        rec.get("sentence", "")
                    ).strip(),
                    "opinion": str(
                        rec.get("opinion", "")
                    ).strip(),
                    "aspect": str(
                        rec.get("aspect")
                        or rec.get("Aspect")
                        or rec.get("Sub_Issue")
                        or aspect_name
                    ).strip(),
                    "service_element": str(
                        rec.get(
                            "Sub_Issue",
                            aspect_name,
                        )
                    ).strip(),
                    "es": str(
                        rec.get("ES", cluster_id)
                    ).strip(),
                    "category": str(
                        rec.get("c", "")
                    ).strip(),
                    "semantic_text": str(
                        rec.get("semantic_text", "")
                    ).strip(),
                    "informativeness_text": str(
                        rec.get(
                            "informativeness_text",
                            "",
                        )
                    ).strip(),
                    "aop_text": str(
                        rec.get("AOP_Text", "")
                    ).strip(),
                    "style_anchor": str(
                        rec.get(
                            "Style_Anchor",
                            rec.get("sentence", ""),
                        )
                    ).strip(),
                }
            )
        return records

    def retrieve_paired_reviews(
        self,
        cluster_id: str,
        aspect_name: str,
        current_score: float,
        target_score: float,
        topk: int = 8,
        candidate_pool_multiplier: int = 5,
        score_weight: float = 0.40,
        informativeness_weight: float = 0.25,
        representativeness_weight: float = 0.15,
        consistency_weight: float = 0.15,
        direction_weight: float = 0.05,
        diversity_penalty: float = 0.10,
        min_token_threshold: int = 8,
        neighbor_q: int = 10,
        consistency_eta: float = 2.0,
        direction_delta: float = 0.25,
        direction_tau: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve current-level and target-level evidence for paired gap diagnosis."""
        current_reviews = self.retrieve_reviews_by_score(
            cluster_id=cluster_id,
            aspect_name=aspect_name,
            anchor_score=current_score,
            topk=topk,
            evidence_role="current",
            candidate_pool_multiplier=candidate_pool_multiplier,
            score_weight=score_weight,
            informativeness_weight=informativeness_weight,
            representativeness_weight=representativeness_weight,
            consistency_weight=consistency_weight,
            direction_weight=direction_weight,
            diversity_penalty=diversity_penalty,
            min_token_threshold=min_token_threshold,
            neighbor_q=neighbor_q,
            consistency_eta=consistency_eta,
            direction_delta=direction_delta,
            direction_tau=direction_tau,
        )
        target_reviews = self.retrieve_reviews_by_score(
            cluster_id=cluster_id,
            aspect_name=aspect_name,
            anchor_score=target_score,
            topk=topk,
            evidence_role="target",
            candidate_pool_multiplier=candidate_pool_multiplier,
            score_weight=score_weight,
            informativeness_weight=informativeness_weight,
            representativeness_weight=representativeness_weight,
            consistency_weight=consistency_weight,
            direction_weight=direction_weight,
            diversity_penalty=diversity_penalty,
            min_token_threshold=min_token_threshold,
            neighbor_q=neighbor_q,
            consistency_eta=consistency_eta,
            direction_delta=direction_delta,
            direction_tau=direction_tau,
        )
        return current_reviews, target_reviews

    def retrieve_target_reviews(
        self,
        cluster_id: str,
        aspect_name: str,
        target_score: float,
        topk: int = 8,
    ) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper for older target-only calls."""
        return self.retrieve_reviews_by_score(
            cluster_id=cluster_id,
            aspect_name=aspect_name,
            anchor_score=target_score,
            topk=topk,
            evidence_role="target",
        )

    @staticmethod
    def format_review_evidence(review_records: List[Dict[str, Any]], title: Optional[str] = None) -> str:
        if not review_records:
            return f"{title + ': ' if title else ''}No matched historical review evidence."
        lines: List[str] = []
        if title:
            lines.append(title)
        for i, rec in enumerate(review_records, start=1):
            opinion = rec.get("opinion") or "N/A"
            lines.append(
                f"{i}. [role={rec.get('evidence_role', '')}, sentiment={safe_float(rec.get('sentiment_score')):.2f}, "
                f"anchor={safe_float(rec.get('anchor_score')):.2f}, diff={safe_float(rec.get('score_diff')):.2f}, "
                f"prox={safe_float(rec.get('score_proximity')):.2f}, info={safe_float(rec.get('textual_informativeness')):.2f}, "
                f"rep={safe_float(rec.get('semantic_representativeness')):.2f}, "
                f"cons={safe_float(rec.get('semantic_neighborhood_consistency')):.2f}, "
                f"dir={safe_float(rec.get('role_direction_preference')):.2f}, "
                f"sel={safe_float(rec.get('selection_score')):.2f}] "
                f"Sentence: \"{rec.get('sentence', '')}\"; Opinion focus: {opinion}"
            )
        return "\n".join(lines)

    def derive_action_plan_from_paired_reviews(
        self,
        aspect_name: str,
        current_score: float,
        target_score: float,
        delta: float,
        current_reviews: List[Dict[str, Any]],
        target_reviews: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Diagnose the current-target experiential gap and generate actions.

        The LLM receives two evidence sets: sentences near the current score and
        sentences near the target score. This prevents it from inferring a gap
        from the target state alone.
        """
        current_block = self.format_review_evidence(current_reviews, title="Current-level evidence closest to the current score")
        target_block = self.format_review_evidence(target_reviews, title="Target-level evidence closest to the target score")
        prompt = f"""
Task: Current-target contrastive evidence-grounded hotel service improvement planning.

Service Aspect / ES: "{aspect_name}"
Current Score: {current_score:.2f} / 10
Target Score: {target_score:.2f} / 10
Delta: {delta:+.2f}

{current_block}

{target_block}

Rules:
1. First summarize the current service state only from the current-level evidence.
2. Then summarize the target service state only from the target-level evidence.
3. Diagnose the experiential gap by explicitly contrasting the two evidence sets.
4. Generate concrete, specific, and feasible hotel actions that close the diagnosed gap.
5. Do not invent facts unsupported by the evidence. If evidence is weak, say so briefly in the gap.
6. Keep concise. No background explanation.

Output PURE JSON only:
{{
  "current_state": "concise current state, max 24 English words or 35 Chinese characters",
  "target_state": "concise target state, max 24 English words or 35 Chinese characters",
  "experience_gap": "specific gap between current and target states",
  "gap_dimensions": ["gap dimension 1", "gap dimension 2"],
  "evidence_mapping": [
    {{
      "gap": "specific gap",
      "current_evidence": "short quote or paraphrase from current evidence",
      "target_evidence": "short quote or paraphrase from target evidence",
      "action": "corresponding action"
    }}
  ],
  "suggested_actions": ["specific action 1", "specific action 2"],
  "resource_type": "Process optimization / Hardware upgrade / Mixed / ...",
  "implementation_time": 7/14/30/...
}}
"""
        response = self.call_llm(
            prompt,
            system_role=(
                "You are a hotel operations expert. Return concise JSON only. "
                "Base the gap on current-vs-target evidence, not on imagination."
            ),
        )
        try:
            parsed = normalize_json_response(response)
            if isinstance(parsed, dict):
                # Backward aliases if the model uses older keys.
                if "target_state" not in parsed and "target_service_level_summary" in parsed:
                    parsed["target_state"] = parsed.get("target_service_level_summary", "")
                if "experience_gap" not in parsed and "current_gap" in parsed:
                    parsed["experience_gap"] = parsed.get("current_gap", "")
                if "current_gap" not in parsed and "experience_gap" in parsed:
                    parsed["current_gap"] = parsed.get("experience_gap", "")
            return parsed
        except Exception as e:
            return {"raw_text": response, "parsing_error": str(e)}

    def derive_action_plan_from_target_reviews(
        self,
        aspect_name: str,
        current_score: float,
        target_score: float,
        delta: float,
        review_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Backward-compatible target-only method.

        Prefer `derive_action_plan_from_paired_reviews()` for the current-target
        paired evidence workflow.
        """
        return self.derive_action_plan_from_paired_reviews(
            aspect_name=aspect_name,
            current_score=current_score,
            target_score=target_score,
            delta=delta,
            current_reviews=[],
            target_reviews=review_records,
        )


    def check_global_coherence(self, generated_proposal_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not generated_proposal_list or len(generated_proposal_list) <= 1:
            return {"is_feasible": True, "conflicts": []}

        conflict_list_str = "\n".join(
            [f"- Aspect '{item['aspect']}'\n  Actions: {item.get('action_plan', '')}" for item in generated_proposal_list]
        )
        prompt = f"""
Task: Global Coherence Check for Hotel Improvement Proposals.

Analyze the following list of proposed, simultaneous changes for a single hotel. Focus primarily on whether the ACTIONS of different aspects introduce direct, critical, physical, or functional contradictions when executed together.

Conflict Definition:
Actions are mutually exclusive, for example: increase activity space vs. add large furniture; increase water temperature vs. decrease water temperature.

Ignore:
- Indirect weak causal chains.
- Minor aesthetic conflicts.
- Vague resource competition unless it makes two actions mutually impossible.

Proposals to Check:
{conflict_list_str}

Output PURE JSON:
{{
  "is_feasible": true/false,
  "conflicts": [
    {{"aspects": ["Aspect_1", "Aspect_2"], "reason": "Explain the DIRECT physical/functional contradiction."}}
  ]
}}
"""
        response = self.call_llm(
            prompt,
            system_role="You are a pragmatic construction and hotel logistics expert. Only flag critical and direct conflicts.",
        )
        try:
            parsed = normalize_json_response(response)
            if parsed.get("is_feasible") is None or parsed.get("conflicts") is None:
                raise ValueError("Malformed JSON")
            return parsed
        except Exception as e:
            return {"is_feasible": False, "conflicts": [{"aspects": ["Parsing Error"], "reason": f"Failed to parse LLM response: {e}"}]}

    def resolve_strategic_conflicts(
        self,
        all_proposals: List[Dict[str, Any]],
        global_conflicts: List[Dict[str, Any]],
        human_guidance: str = "",
    ) -> List[Dict[str, Any]]:
        proposal_dict = {p["aspect"]: p for p in all_proposals}
        for conflict in global_conflicts:
            aspects_in_conflict = conflict.get("aspects", [])
            current_proposals = [proposal_dict[a] for a in aspects_in_conflict if a in proposal_dict]
            if len(current_proposals) < 2:
                continue
            conflict_details = "\n".join(
                [
                    f"- Aspect '{p['aspect']}' (Target: {safe_float(p.get('target_score')):.2f}, Delta: {safe_float(p.get('delta')):+.2f}, PRIORITY: {p.get('priority', 4)}):\n"
                    f"    Target state: \"{p.get('target_state') or p.get('aop', '')}\"\n    Gap: {p.get('current_gap', '')}\n    Evidence: {p.get('closest_review_sentence', '')}\n    Actions: {p.get('action_plan', 'Action plan missing')}"
                    for p in current_proposals
                ]
            )
            prompt = f"""
Task: Strategic Conflict Resolution.
Core Conflict Reason: {conflict.get('reason', '')}
Conflicting Proposals:
{conflict_details}

Priority Rule: lower priority number means higher priority. Keep higher-priority proposals when trade-offs are needed.
Decision Criteria:
1. Prefer action refinement while keeping the target state unchanged.
2. Modify target state only if required.
3. Trade off the lowest-priority aspect when actions are mutually impossible.
Keep all output concise.

Output PURE JSON:
{{
  "decision_type": "Guide" / "Trade-Off",
  "deactivate_aspect": "Aspect Name to remove" / "None",
  "updated_proposals": [
    {{"aspect": "Aspect Name", "new_aop": "concise revised target state or KEEP ORIGINAL", "new_actions_guidance": "concise action guidance or None"}}
  ]
}}
"""
            if human_guidance:
                prompt = f"[HUMAN OVERRIDE / REVIEWER GUIDANCE]\n{human_guidance}\n\n" + prompt
            response = self.call_llm(prompt, system_role="You are a decisive Strategic Conflict Resolution Agent.")
            try:
                parsed = normalize_json_response(response)
                if parsed.get("decision_type") == "Trade-Off" and parsed.get("deactivate_aspect") not in (None, "None"):
                    proposal_dict.pop(parsed["deactivate_aspect"], None)
                elif parsed.get("decision_type") == "Guide":
                    for updated in parsed.get("updated_proposals", []):
                        aspect = updated.get("aspect")
                        if not aspect or aspect not in proposal_dict:
                            continue

                        proposal = proposal_dict[aspect]
                        original_target = str(
                            proposal.get("target_state")
                            or proposal.get("aop")
                            or ""
                        )
                        revised_target = str(
                            updated.get("new_aop")
                            or original_target
                        ).strip()
                        if revised_target.upper() == "KEEP ORIGINAL":
                            revised_target = original_target

                        if revised_target and revised_target != original_target:
                            proposal["aop"] = revised_target
                            proposal["target_state"] = revised_target
                            proposal["target_service_level_summary"] = revised_target

                        action_guidance = str(
                            updated.get("new_actions_guidance")
                            or ""
                        ).strip()
                        if action_guidance and action_guidance.lower() != "none":
                            old_action = str(
                                proposal.get("action_plan", "")
                            )
                            new_action = self.refine_action_plan(
                                aspect_name=aspect,
                                action_plan_str=old_action,
                                guidance=action_guidance,
                                aop=proposal.get(
                                    "target_state",
                                    proposal.get("aop", ""),
                                ),
                            )
                            if new_action.strip() != old_action.strip():
                                proposal.setdefault(
                                    "action_revision_history",
                                    [],
                                ).append(
                                    {
                                        "revision_id": len(
                                            proposal.get(
                                                "action_revision_history",
                                                [],
                                            )
                                        )
                                        + 1,
                                        "timestamp": datetime.now().strftime(
                                            "%Y-%m-%d %H:%M:%S"
                                        ),
                                        "trigger_type": "conflict_resolution",
                                        "trigger_detail": action_guidance,
                                        "old_action_plan": old_action,
                                        "new_action_plan": new_action,
                                        "notes": "Repaired under global conflict constraints",
                                    }
                                )
                                proposal["action_plan"] = new_action
                            proposal["new_actions_guidance"] = action_guidance

                        proposal["aop_refined"] = False
            except Exception as e:
                for p in current_proposals:
                    p.setdefault("resolution_errors", []).append(str(e))
        return list(proposal_dict.values())

    def refine_action_plan(self, aspect_name: str, action_plan_str: str, guidance: str, aop: str) -> str:
        prompt = f"""
Task: Concise action-plan refinement.
Aspect: {aspect_name}
Target state / gap summary: "{aop}"
Original actions: {action_plan_str}
Reviewer guidance: {guidance}

Revise the actions only. Keep 2-4 concrete, feasible actions. No explanation.
Output PURE JSON: {{"suggested_actions": ["action 1", "action 2"]}}
"""
        response = self.call_llm(
            prompt,
            system_role="You edit hotel action plans into concise, feasible operational actions. JSON only.",
        )
        try:
            parsed = normalize_json_response(response)
            actions = parsed.get("suggested_actions", [])
            if isinstance(actions, list):
                return "; ".join(str(x).strip() for x in actions if str(x).strip())
            return str(actions)
        except Exception:
            return action_plan_str

   
def create_interpreter(
    api_key: str,
    history_df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    max_retries: int = 3,
    timeout: float = 120.0,
    request_retries: int = 2,
    similarity_backend: str = "sentence-transformers",
    sentence_model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> ServiceStrategyInterpreter:
    chat_client = QwenPlusChatClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        request_retries=request_retries,
    )
    return ServiceStrategyInterpreter(
        chat_client=chat_client,
        history_df=history_df,
        max_retries=max_retries,
        similarity_backend=similarity_backend,
        sentence_model_path=sentence_model_path,
    )


def generate_one_proposal(
    interpreter: ServiceStrategyInterpreter,
    cluster_id: str,
    df_per: pd.DataFrame,
    per_dict: Dict[str, Dict[str, Any]],
    options: List[Dict[str, Any]],
    evidence_topk: int = 8,
    candidate_pool_multiplier: int = 5,
    score_weight: float = 0.40,
    informativeness_weight: float = 0.25,
    representativeness_weight: float = 0.15,
    consistency_weight: float = 0.15,
    direction_weight: float = 0.05,
    diversity_penalty: float = 0.10,
    min_token_threshold: int = 8,
    neighbor_q: int = 10,
    consistency_eta: float = 2.0,
    direction_delta: float = 0.25,
    direction_tau: float = 1.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Generate a current-target paired evidence-grounded action plan."""
    row_match = df_per.loc[df_per["ES"].astype(str) == str(cluster_id)]
    if row_match.empty:
        return None, "cluster_not_found"
    delta = safe_float(row_match["pareto"].iloc[0])
    if cluster_id not in per_dict:
        return None, "per_dict_missing"

    aspect_info = per_dict[cluster_id]
    aspect_name = aspect_info["name"]
    current_score = safe_float(aspect_info["current_score"])
    target_score = current_score + delta
    current_opt = next((opt for opt in options if str(opt["ES"]) == str(cluster_id)), None)
    aspect_priority = current_opt["priority"] if current_opt else 4

    current_reviews, target_reviews = interpreter.retrieve_paired_reviews(
        cluster_id=str(cluster_id),
        aspect_name=str(aspect_name),
        current_score=current_score,
        target_score=target_score,
        topk=evidence_topk,
        candidate_pool_multiplier=candidate_pool_multiplier,
        score_weight=score_weight,
        informativeness_weight=informativeness_weight,
        representativeness_weight=representativeness_weight,
        consistency_weight=consistency_weight,
        direction_weight=direction_weight,
        diversity_penalty=diversity_penalty,
        min_token_threshold=min_token_threshold,
        neighbor_q=neighbor_q,
        consistency_eta=consistency_eta,
        direction_delta=direction_delta,
        direction_tau=direction_tau,
    )
    if not current_reviews:
        return None, "no_current_review_evidence"
    if not target_reviews:
        return None, "no_target_review_evidence"

    action_plan = interpreter.derive_action_plan_from_paired_reviews(
        aspect_name=str(aspect_name),
        current_score=current_score,
        target_score=target_score,
        delta=delta,
        current_reviews=current_reviews,
        target_reviews=target_reviews,
    )
    actions_str, resource_type, implementation_time, raw_plan = clean_action_plan(action_plan)

    current_state = ""
    target_state = ""
    experience_gap = ""
    gap_dimensions: Any = []
    evidence_mapping: Any = []
    if isinstance(raw_plan, dict):
        current_state = str(raw_plan.get("current_state") or "").strip()
        target_state = str(
            raw_plan.get("target_state")
            or raw_plan.get("target_service_level_summary")
            or ""
        ).strip()
        experience_gap = str(
            raw_plan.get("experience_gap")
            or raw_plan.get("current_gap")
            or ""
        ).strip()
        gap_dimensions = raw_plan.get("gap_dimensions", []) or []
        evidence_mapping = raw_plan.get("evidence_mapping", []) or []

    if not current_state:
        current_state = f"Closest current-level evidence: {current_reviews[0].get('sentence', '')[:120]}"
    if not target_state:
        target_state = f"Closest target-level evidence: {target_reviews[0].get('sentence', '')[:120]}"
    if not experience_gap:
        experience_gap = f"Raise {aspect_name} from {current_score:.2f} to {target_score:.2f} based on current-target evidence contrast."

    def _evidence_text(records: List[Dict[str, Any]]) -> str:
        return "\n".join(
            [
                (
                    f"[{i}] score={safe_float(r.get('sentiment_score')):.2f}, "
                    f"diff={safe_float(r.get('score_diff')):.2f}, "
                    f"prox={safe_float(r.get('score_proximity')):.2f}, "
                    f"info={safe_float(r.get('textual_informativeness')):.2f}, "
                    f"rep={safe_float(r.get('semantic_representativeness')):.2f}, "
                    f"cons={safe_float(r.get('semantic_neighborhood_consistency')):.2f}, "
                    f"dir={safe_float(r.get('role_direction_preference')):.2f}, "
                    f"red_pen={safe_float(r.get('redundancy_penalty')):.2f}, "
                    f"sel={safe_float(r.get('selection_score')):.2f}: "
                    f"{r.get('sentence', '')}"
                )
                for i, r in enumerate(records, start=1)
            ]
        )

    def _scores(records: List[Dict[str, Any]]) -> str:
        return ", ".join(f"{safe_float(r.get('sentiment_score')):.2f}" for r in records)

    def _metric_list(records: List[Dict[str, Any]], key: str) -> str:
        return ", ".join(f"{safe_float(r.get(key)):.4f}" for r in records)

    closest_current_review = min(
        current_reviews,
        key=lambda r: safe_float(
            r.get("score_diff"),
            default=float("inf"),
        ),
    )
    closest_target_review = min(
        target_reviews,
        key=lambda r: safe_float(
            r.get("score_diff"),
            default=float("inf"),
        ),
    )
    closest_current_diff = safe_float(
        closest_current_review.get("score_diff")
    )
    closest_target_diff = safe_float(
        closest_target_review.get("score_diff")
    )
    representation_method = str(
        current_reviews[0].get(
            "semantic_representation",
            target_reviews[0].get(
                "semantic_representation",
                "unknown",
            ),
        )
    )

    return {
        "cluster_id": str(cluster_id),
        "aspect": aspect_name,
        "current_score": round(current_score, 2),
        "delta": round(delta, 2),
        "target_score": round(target_score, 2),
        # Backward-compatible alias: `aop` stores the target-state summary, not a simulated review.
        "aop": target_state,
        "current_state": current_state,
        "target_state": target_state,
        "target_service_level_summary": target_state,
        "current_gap": experience_gap,
        "experience_gap": experience_gap,
        "gap_dimensions": json.dumps(gap_dimensions, ensure_ascii=False) if gap_dimensions else "",
        "evidence_mapping": json.dumps(evidence_mapping, ensure_ascii=False) if evidence_mapping else "",
        "closest_current_review_sentence": closest_current_review.get("sentence", ""),
        "closest_current_review_sentiment": round(safe_float(closest_current_review.get("sentiment_score")), 4),
        "closest_current_review_score_diff": round(closest_current_diff, 4),
        "closest_target_review_sentence": closest_target_review.get("sentence", ""),
        "closest_target_review_sentiment": round(safe_float(closest_target_review.get("sentiment_score")), 4),
        "closest_target_review_score_diff": round(closest_target_diff, 4),
        # Legacy target-only aliases retained for UI/export compatibility.
        "closest_review_sentence": closest_target_review.get("sentence", ""),
        "closest_review_sentiment": round(safe_float(closest_target_review.get("sentiment_score")), 4),
        "closest_review_score_diff": round(closest_target_diff, 4),
        "current_review_evidence": _evidence_text(current_reviews),
        "target_review_evidence": _evidence_text(target_reviews),
        "current_evidence_count": len(current_reviews),
        "target_evidence_count": len(target_reviews),
        "evidence_count": len(current_reviews) + len(target_reviews),
        "semantic_representation": representation_method,
        "current_evidence_sentiment_scores": _scores(current_reviews),
        "target_evidence_sentiment_scores": _scores(target_reviews),
        "evidence_sentiment_scores": f"current: {_scores(current_reviews)} | target: {_scores(target_reviews)}",
        "current_evidence_score_proximity": _metric_list(current_reviews, "score_proximity"),
        "target_evidence_score_proximity": _metric_list(target_reviews, "score_proximity"),
        "current_evidence_length_sufficiency": _metric_list(current_reviews, "length_sufficiency"),
        "target_evidence_length_sufficiency": _metric_list(target_reviews, "length_sufficiency"),
        "current_evidence_term_informativeness": _metric_list(current_reviews, "term_informativeness"),
        "target_evidence_term_informativeness": _metric_list(target_reviews, "term_informativeness"),
        "current_evidence_textual_informativeness": _metric_list(current_reviews, "textual_informativeness"),
        "target_evidence_textual_informativeness": _metric_list(target_reviews, "textual_informativeness"),
        "current_evidence_semantic_representativeness": _metric_list(current_reviews, "semantic_representativeness"),
        "target_evidence_semantic_representativeness": _metric_list(target_reviews, "semantic_representativeness"),
        "current_evidence_neighbor_sentiments": _metric_list(current_reviews, "semantic_neighbor_sentiment"),
        "target_evidence_neighbor_sentiments": _metric_list(target_reviews, "semantic_neighbor_sentiment"),
        "current_evidence_semantic_consistency": _metric_list(current_reviews, "semantic_neighborhood_consistency"),
        "target_evidence_semantic_consistency": _metric_list(target_reviews, "semantic_neighborhood_consistency"),
        "current_evidence_direction_preference": _metric_list(current_reviews, "role_direction_preference"),
        "target_evidence_direction_preference": _metric_list(target_reviews, "role_direction_preference"),
        "current_evidence_redundancy_penalties": _metric_list(current_reviews, "redundancy_penalty"),
        "target_evidence_redundancy_penalties": _metric_list(target_reviews, "redundancy_penalty"),
        "current_evidence_selection_scores": _metric_list(current_reviews, "selection_score"),
        "target_evidence_selection_scores": _metric_list(target_reviews, "selection_score"),
        "evidence_selection_method": (
            f"semantic_representation={representation_method}; "
            f"candidate_pool_multiplier={candidate_pool_multiplier}; "
            f"score_weight={score_weight}; informativeness_weight={informativeness_weight}; "
            f"representativeness_weight={representativeness_weight}; consistency_weight={consistency_weight}; "
            f"direction_weight={direction_weight}; diversity_penalty={diversity_penalty}; "
            f"min_token_threshold={min_token_threshold}; neighbor_q={neighbor_q}; "
            f"consistency_eta={consistency_eta}; direction_delta={direction_delta}; direction_tau={direction_tau}; "
            f"AOP used only for localization/display"
        ),
        "action_plan": actions_str,
        "Resource_Type": resource_type,
        "implementation_time": implementation_time,
        "priority": aspect_priority,
        "feasibility_score": 1.0,
        "llm_predicted_score": "current_target_evidence_based",
        "feasibility_reasons": (
            "Current-target paired evidence generation; "
            f"closest current diff={closest_current_diff:.4f}; closest target diff={closest_target_diff:.4f}"
        ),
    }, None

def finalize_results(final_proposals: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for proposal in final_proposals:
        if proposal.get("is_rejected", False):
            continue
        rows.append(
            {
                "Cluster_ID": proposal.get("cluster_id"),
                "Aspect_Name": proposal.get("aspect"),
                "Current_Score": proposal.get("current_score"),
                "Delta": proposal.get("delta"),
                "Target_Score": proposal.get("target_score"),
                "Current_State": proposal.get("current_state"),
                "Target_State": proposal.get("target_state") or proposal.get("target_service_level_summary") or proposal.get("aop"),
                "Experience_Gap": proposal.get("experience_gap") or proposal.get("current_gap"),
                "Gap_Dimensions": proposal.get("gap_dimensions"),
                "Suggested_Actions": proposal.get("action_plan", "Action plan missing/error"),
                "Resource_Type": proposal.get("Resource_Type", "Unknown"),
                "implementation_time": proposal.get("implementation_time", "Unknown"),
                "Priority": proposal.get("priority"),
                "Current_Evidence_Count": proposal.get("current_evidence_count"),
                "Target_Evidence_Count": proposal.get("target_evidence_count"),
                "Evidence_Count_Total": proposal.get("evidence_count"),
                "Semantic_Representation": proposal.get("semantic_representation"),
                "Closest_Current_Review_Sentence": proposal.get("closest_current_review_sentence"),
                "Closest_Current_Review_Sentiment": proposal.get("closest_current_review_sentiment"),
                "Closest_Current_Review_Score_Diff": proposal.get("closest_current_review_score_diff"),
                "Closest_Target_Review_Sentence": proposal.get("closest_target_review_sentence") or proposal.get("closest_review_sentence"),
                "Closest_Target_Review_Sentiment": proposal.get("closest_target_review_sentiment") or proposal.get("closest_review_sentiment"),
                "Closest_Target_Review_Score_Diff": proposal.get("closest_target_review_score_diff") or proposal.get("closest_review_score_diff"),
                "Current_Evidence_Sentiment_Scores": proposal.get("current_evidence_sentiment_scores"),
                "Target_Evidence_Sentiment_Scores": proposal.get("target_evidence_sentiment_scores"),
                "Current_Evidence_Score_Proximity": proposal.get("current_evidence_score_proximity"),
                "Target_Evidence_Score_Proximity": proposal.get("target_evidence_score_proximity"),
                "Current_Evidence_Textual_Informativeness": proposal.get("current_evidence_textual_informativeness"),
                "Target_Evidence_Textual_Informativeness": proposal.get("target_evidence_textual_informativeness"),
                "Current_Evidence_Length_Sufficiency": proposal.get("current_evidence_length_sufficiency"),
                "Target_Evidence_Length_Sufficiency": proposal.get("target_evidence_length_sufficiency"),
                "Current_Evidence_Term_Informativeness": proposal.get("current_evidence_term_informativeness"),
                "Target_Evidence_Term_Informativeness": proposal.get("target_evidence_term_informativeness"),
                "Current_Evidence_Semantic_Representativeness": proposal.get("current_evidence_semantic_representativeness"),
                "Target_Evidence_Semantic_Representativeness": proposal.get("target_evidence_semantic_representativeness"),
                "Current_Evidence_Semantic_Neighbor_Sentiments": proposal.get("current_evidence_neighbor_sentiments"),
                "Target_Evidence_Semantic_Neighbor_Sentiments": proposal.get("target_evidence_neighbor_sentiments"),
                "Current_Evidence_Semantic_Consistency": proposal.get("current_evidence_semantic_consistency"),
                "Target_Evidence_Semantic_Consistency": proposal.get("target_evidence_semantic_consistency"),
                "Current_Evidence_Direction_Preference": proposal.get("current_evidence_direction_preference"),
                "Target_Evidence_Direction_Preference": proposal.get("target_evidence_direction_preference"),
                "Current_Evidence_Redundancy_Penalties": proposal.get("current_evidence_redundancy_penalties"),
                "Target_Evidence_Redundancy_Penalties": proposal.get("target_evidence_redundancy_penalties"),
                "Current_Evidence_Selection_Scores": proposal.get("current_evidence_selection_scores"),
                "Target_Evidence_Selection_Scores": proposal.get("target_evidence_selection_scores"),
                "Evidence_Selection_Method": proposal.get("evidence_selection_method"),
                "Current_Review_Evidence": proposal.get("current_review_evidence"),
                "Target_Review_Evidence": proposal.get("target_review_evidence"),
                "Evidence_Mapping": proposal.get("evidence_mapping"),
                "Feasibility_Reasons": proposal.get("feasibility_reasons"),
            }
        )
    return pd.DataFrame(rows)

def discover_excel_files(data_dir: Path) -> List[Path]:
    """Find Excel files from ./data."""
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.glob("*.xls*") if p.is_file()])


def read_excel_from_data(data_dir: Path, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")
    if sheet_name:
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_excel(path)


def build_candidate_clusters(
    df_per: pd.DataFrame,
    delta_threshold: float = 0.5,
    options: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    df = df_per.copy()

    df["ES"] = df["ES"].astype(str)
    df["pareto_num"] = df["pareto"].apply(lambda x: safe_float(x, 0.0))
    df["Imp_num"] = df["Imp"].apply(lambda x: safe_float(x, 0.0)) if "Imp" in df.columns else 0.0
    df["Per_focus_num"] = df["Per_focus"].apply(lambda x: safe_float(x, 0.0)) if "Per_focus" in df.columns else 0.0

    df = df[df["pareto_num"] >= delta_threshold].copy()

    priority_map = {}
    if options:
        priority_map = {str(opt["ES"]): opt.get("priority", 4) for opt in options}

    df["priority_num"] = df["ES"].map(priority_map).fillna(4)

    df = df.sort_values(
        by=["priority_num", "pareto_num", "Imp_num", "Per_focus_num"],
        ascending=[True, False, False, True],
    )

    return df["ES"].astype(str).tolist()

def save_outputs(
    output_dir: Path,
    result_df: pd.DataFrame,
    proposals: List[Dict[str, Any]],
    prefix: str = "hotel_strategy",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{prefix}_{ts}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="final_results", index=False)
        pd.DataFrame(proposals).to_excel(writer, sheet_name="final_proposals_raw", index=False)
    return out_path
