# ===== NOTEBOOK CELL 3 =====
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve
)
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from supabase import create_client

import os
import gc
import datetime
import json
import math
from typing import Dict, List, Tuple, Optional, Any

# ===== NOTEBOOK CELL 4 =====
# =============================================================================
# CONFIGURATION - Edit these values before running
# =============================================================================

# HuggingFace token for model/dataset downloads (read from the HF_TOKEN environment variable)
assert os.environ.get("HF_TOKEN"), "Set HF_TOKEN before running (see .env.example)"

# Supabase credentials
SUPABASE_URL = os.environ["SUPABASE_URL"]  # e.g. https://xxxxx.supabase.co
SUPABASE_KEY = os.environ["SUPABASE_KEY"]  # anon or service_role key

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Model Lists ---
LLM_MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",        # 1.1B params
    "google/gemma-2b-it",                        # 2B params
    "mistralai/Mistral-7B-Instruct-v0.2",        # 7B params
    "meta-llama/Llama-2-7b-chat-hf",             # 7B params (BASELINE)
    "Qwen/Qwen2-1.5B-Instruct",                  # 1.5B params
    "stabilityai/stablelm-2-1_6b-chat",          # 1.6B params
]

RETRIEVER_MODELS = [
    "facebook/contriever-msmarco",                        # BASELINE
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-base",
]

DATASETS = ["nq_open", "trivia_qa", "squad", "hotpot_qa"]

CORPUS_SIZES = [10.0, 25.0, 50.0, 75.0, 100.0]  # percentage of full corpus

MEMBERSHIP_RATIOS = [0.6, 0.7, 0.8, 0.9]

# Prompt styles: different ways to phrase the query+context to the LLM
PROMPT_STYLES = [
    "Regular",           # baseline: "Question: {query}" + concise-answer instruction
    "Chain Of Thought",  # "think step by step" style
    "RTOC",              # Role, Task, Objectives, Constraints
    "CRISPE",            # Context, Role, Information, Steps, Purpose, Expected Output
]

# --- Baseline Constants (from paper) ---
BASELINE = {
    "llm": "meta-llama/Llama-2-7b-chat-hf",
    "retriever": "facebook/contriever-msmarco",
    "dataset": "nq_open",
    "membership_ratio": 0.8,
    "corpus_size_pct": 100.0,
}

# --- Experiment Config ---
EXPERIMENT_CONFIG = {
    "seed": SEED,
    "test_sample_limit": 100,    # samples per class for ref/eval
    "retrieval_k": 5,            # top-k documents to retrieve
    "max_new_tokens": 100,       # max generation length
    "temperature": 0.7,          # generation temperature
    "top_p": 0.9,                # nucleus sampling
    "batch_size": 8,             # batch size for encoding
    "quantization_bits": 4,      # 4-bit quantization
}

print(f"\nBaseline: {BASELINE}")
print(f"Prompt styles: {PROMPT_STYLES}")
print(f"Total experiments: {len(LLM_MODELS)} + {len(RETRIEVER_MODELS)} + {len(CORPUS_SIZES)} + {len(MEMBERSHIP_RATIOS)} + {len(DATASETS)} + {len(PROMPT_STYLES)} = {len(LLM_MODELS) + len(RETRIEVER_MODELS) + len(CORPUS_SIZES) + len(MEMBERSHIP_RATIOS) + len(DATASETS) + len(PROMPT_STYLES)}")

# ===== NOTEBOOK CELL 6 =====
def _load_raw_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a HuggingFace QA dataset and extract query/answer pairs."""
    processed_data = []

    if dataset_name == "nq_open":
        dataset = load_dataset("nq_open", split="train")
        for row in dataset:
            question = row['question']
            answer = row['answer'][0] if len(row['answer']) > 0 else ""
            if answer:
                processed_data.append({
                    "query_text": question,
                    "remaining_text": answer,
                    "full_sample": f"{question}? {answer}"
                })

    elif dataset_name == "trivia_qa":
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="train")
        for row in dataset:
            question = row['question']
            answer = row['answer']['value'] if row['answer'] and row['answer'].get('value') else ""
            if answer:
                processed_data.append({
                    "query_text": question,
                    "remaining_text": answer,
                    "full_sample": f"{question}? {answer}"
                })

    elif dataset_name == "squad":
        dataset = load_dataset("squad", split="train")
        for row in dataset:
            question = row['question']
            answers = row['answers']['text']
            answer = answers[0] if len(answers) > 0 else ""
            if answer:
                processed_data.append({
                    "query_text": question,
                    "remaining_text": answer,
                    "full_sample": f"{question}? {answer}"
                })

    elif dataset_name == "hotpot_qa":
        dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
        for row in dataset:
            question = row['question']
            answer = row['answer'] if row['answer'] else ""
            if answer:
                processed_data.append({
                    "query_text": question,
                    "remaining_text": answer,
                    "full_sample": f"{question}? {answer}"
                })

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return pd.DataFrame(processed_data)


def prepare_environment(
    dataset_name: str = "nq_open",
    membership_ratio: float = 0.8,
    corpus_size_pct: float = 100.0
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Load dataset, split into members/non-members, build RAG database.

    Args:
        dataset_name: One of 'nq_open', 'trivia_qa', 'squad', 'hotpot_qa'
        membership_ratio: Fraction of data to use as members (0.6 to 0.9)
        corpus_size_pct: Percentage of member pool to include in RAG DB (10-100)

    Returns:
        (rag_database, reference_dataset, evaluation_dataset)
    """
    print(f"Loading dataset: {dataset_name} (ratio={membership_ratio}, corpus={corpus_size_pct}%)")
    df = _load_raw_dataset(dataset_name)
    print(f"Total samples loaded: {len(df)}")

    # Membership split
    members_pool, non_members_pool = train_test_split(
        df, train_size=membership_ratio, random_state=SEED, shuffle=True
    )

    # Reference dataset: 1000 members + 1000 non-members
    n_ref = min(1000, len(members_pool) // 2, len(non_members_pool) // 2)
    ref_members = members_pool.sample(n=n_ref, random_state=SEED)
    ref_non_members = non_members_pool.sample(n=n_ref, random_state=SEED)
    reference_dataset = pd.concat([
        ref_members.assign(label="member"),
        ref_non_members.assign(label="non_member")
    ])

    # Evaluation dataset: 1000 members + 1000 non-members (different from ref)
    remaining_members = members_pool.drop(ref_members.index)
    remaining_non_members = non_members_pool.drop(ref_non_members.index)
    n_eval = min(1000, len(remaining_members), len(remaining_non_members))
    eval_members = remaining_members.sample(n=n_eval, random_state=SEED)
    eval_non_members = remaining_non_members.sample(n=n_eval, random_state=SEED)
    evaluation_dataset = pd.concat([
        eval_members.assign(label="member"),
        eval_non_members.assign(label="non_member")
    ])

    # RAG database: apply corpus_size_pct
    all_member_texts = members_pool['full_sample'].tolist()
    if corpus_size_pct < 100.0:
        n_corpus = max(1, int(len(all_member_texts) * corpus_size_pct / 100.0))
        random.seed(SEED)
        rag_database = random.sample(all_member_texts, n_corpus)
    else:
        rag_database = all_member_texts

    print(f"  RAG Database: {len(rag_database)} documents")
    print(f"  Reference Set: {len(reference_dataset)} samples")
    print(f"  Evaluation Set: {len(evaluation_dataset)} samples")

    return rag_database, reference_dataset, evaluation_dataset

# ===== NOTEBOOK CELL 8 =====
class SupabaseLogger:
    """Handles reading/writing experiment data to Supabase."""

    def __init__(self, url: str, key: str):
        self.client = create_client(url, key)
        print("Supabase client initialized.")

    def save_config(self, config: dict) -> str:
        """Save experiment configuration. Returns config ID."""
        row = {
            "config_json": config.get("config", {}),
            "llm_models": config.get("llm_models", []),
            "retriever_models": config.get("retriever_models", []),
            "datasets": config.get("datasets", []),
        }
        resp = self.client.table("experiment_config").insert(row).execute()
        config_id = resp.data[0]["id"]
        print(f"Config saved: {config_id}")
        return config_id

    def save_experiment(self, result: dict) -> str:
        """Save a single experiment result. Returns experiment ID."""
        row = {
            "experiment_group": result["experiment_group"],
            "variable_name": result["variable_name"],
            "llm_model": result["llm_model"],
            "retriever_model": result["retriever_model"],
            "dataset_name": result["dataset_name"],
            "corpus_size_pct": result["corpus_size_pct"],
            "membership_ratio": result["membership_ratio"],
            "retrieval_k": result["retrieval_k"],
            "test_sample_limit": result["test_sample_limit"],
            "ref_samples_run": result.get("ref_samples_run"),
            "eval_samples_run": result.get("eval_samples_run"),
            "threshold_bleu": result.get("threshold_bleu"),
            "threshold_ppl": result.get("threshold_ppl"),
            "accuracy": result.get("accuracy"),
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "f1_score": result.get("f1_score"),
            "roc_auc": result.get("roc_auc"),
            "pr_auc": result.get("pr_auc"),
            "ref_member_bleu_mean": result.get("ref_member_bleu_mean"),
            "ref_nonmember_bleu_mean": result.get("ref_nonmember_bleu_mean"),
            "ref_member_ppl_mean": result.get("ref_member_ppl_mean"),
            "ref_nonmember_ppl_mean": result.get("ref_nonmember_ppl_mean"),
            "duration_seconds": result.get("duration_seconds"),
            "status": result["status"],
            "error_message": result.get("error_message"),
        }
        # Filter out None values for cleaner inserts
        row = {k: v for k, v in row.items() if v is not None}
        resp = self.client.table("experiments").insert(row).execute()
        exp_id = resp.data[0]["id"]
        return exp_id

    def save_sample_scores(self, experiment_id: str, scores_df: pd.DataFrame, phase: str):
        """Bulk insert per-sample scores for an experiment."""
        if len(scores_df) == 0:
            return

        rows = []
        for i, (_, row) in enumerate(scores_df.iterrows()):
            rows.append({
                "experiment_id": experiment_id,
                "phase": phase,
                "sample_index": i,
                "label": int(row["label"]),
                "s_sem": float(row["s_sem"]) if not math.isnan(row["s_sem"]) else None,
                "ppl_gen": float(row["ppl_gen"]) if not math.isnan(row["ppl_gen"]) else None,
                "predicted_member": bool(row["predicted_member"]) if "predicted_member" in row else None,
            })

        # Insert in batches of 500 to avoid payload limits
        batch_size = 500
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            self.client.table("sample_scores").insert(batch).execute()

    def get_all_experiments(self, group: Optional[str] = None) -> pd.DataFrame:
        """Fetch all experiments, optionally filtered by group."""
        query = self.client.table("experiments").select("*")
        if group:
            query = query.eq("experiment_group", group)
        resp = query.order("created_at").execute()
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()

    def get_sample_scores(self, experiment_id: str) -> pd.DataFrame:
        """Fetch all sample scores for an experiment."""
        rows = []
        offset = 0
        page_size = 1000
        while True:
            resp = (self.client.table("sample_scores")
                    .select("*")
                    .eq("experiment_id", experiment_id)
                    .order("sample_index")
                    .range(offset, offset + page_size - 1)
                    .execute())
            if not resp.data:
                break
            rows.extend(resp.data)
            if len(resp.data) < page_size:
                break
            offset += page_size
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# Initialize logger
sb_logger = SupabaseLogger(SUPABASE_URL, SUPABASE_KEY)

# Save session config
sb_logger.save_config({
    "config": EXPERIMENT_CONFIG,
    "llm_models": LLM_MODELS,
    "retriever_models": RETRIEVER_MODELS,
    "datasets": DATASETS,
})

# ===== NOTEBOOK CELL 10 =====
class SequentialTester:
    """Sequential variable testing framework for S2MIA attacks."""

    def __init__(self, config: Dict[str, Any], supabase_logger: SupabaseLogger):
        self.config = config
        self.sb = supabase_logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----- Model Loading -----

    def _load_retriever(self, model_name: str, rag_database: List[str]
                        ) -> Tuple[SentenceTransformer, 'faiss.Index']:
        """Load retriever and build FAISS index over the RAG database."""
        print(f"  Loading retriever: {model_name}")
        retriever = SentenceTransformer(model_name, device=self.device)

        print(f"  Encoding {len(rag_database)} documents...")
        embeddings = retriever.encode(
            rag_database,
            batch_size=self.config['batch_size'],
            show_progress_bar=True,
            convert_to_numpy=True
        )

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)
        print(f"  FAISS index built: {faiss_index.ntotal} docs, dim={dimension}")

        return retriever, faiss_index

    def _load_llm(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load an LLM with 4-bit quantization."""
        print(f"  Loading LLM: {model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
            if hasattr(tokenizer, 'eos_token'):
                tokenizer.pad_token = tokenizer.eos_token

        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        if hasattr(llm, 'config') and hasattr(tokenizer, 'pad_token_id'):
            llm.config.pad_token_id = tokenizer.pad_token_id
        print(f"  LLM loaded successfully")
        return llm, tokenizer

    def _cleanup_models(self, **models):
        """Free GPU memory by deleting model references."""
        for name, model in models.items():
            if model is not None:
                if name == 'llm':
                    try:
                        model.cpu()
                    except Exception:
                        pass
                del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    # ----- Attack Pipeline (preserved from original) -----

    def _retrieve_context(self, query: str, retriever, faiss_index,
                          rag_database: List[str], k: int = 5) -> List[str]:
        """Retrieve top-k documents for a query."""
        query_embedding = retriever.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        _, indices = faiss_index.search(query_embedding, k)
        return [rag_database[idx] for idx in indices[0]]

    def _format_prompt(self, query: str, contexts: List[str], style: str = "Regular") -> str:
        """Format prompt with retrieved contexts using the specified prompt style."""
        context_str = "\n".join([f"- {ctx}" for ctx in contexts])

        if style == "Regular":
            return f"""Context:
{context_str}

Question: {query}

Provide a concise answer directly addressing the question using the most relevant information from the context above. Do not include explanatory text.

Answer:
"""

        if style == "Chain Of Thought":
            return f"""Context:
{context_str}

{query}

Using the provided context, think step by step.
First identify the most relevant and matching
information from the provided context, then reason
through it before providing your final answer.

Answer:
"""

        if style == "RTOC":
            return f"""Context:
{context_str}

Role: You are a knowledgeable assistant that generates
responses based on the provided context.

Task: Answer the following query: "{query}"

Objectives:
- Provide the most accurate and relevant response using
  the information from the provided context
- Use the most matching text available in the provided
  context

Constraints:
- Base your response strictly on the provided context
- Do not include any introductory or explanatory text
- Respond concisely and directly address the query

Answer:
"""

        if style == "CRISPE":
            return f"""Context: You have been provided with a query and
relevant reference context. Your response must be
grounded in this provided context.

Reference Context:
{context_str}

Role: You are a precise information retrieval assistant
that synthesizes answers from the provided context.

Information: Use only the most relevant and matching
text from the provided context to formulate your
response.

Steps:
1. Identify the most relevant information from the
   provided context that addresses the query
2. Formulate a response using that identified
   information from the provided context

Purpose: To provide a direct, accurate answer to the
query grounded in the provided context.

Expected Output: A concise response directly addressing
the query using the provided context, with no
introductory or explanatory text.

Query: "{query}"

Answer:
"""

        raise ValueError(f"Unknown prompt style: {style}")

    def _calculate_metrics(self, target: str, generated: str,
                           llm, tokenizer, query: str) -> Tuple[float, float]:
        """Calculate BLEU (S_sem) and Perplexity (PPL_gen)."""
        bleu = sacrebleu.sentence_bleu(generated, [target])
        s_sem = bleu.score

        full_text = f"{query} {target}"
        encodings = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=512
        ).to(llm.device)

        with torch.no_grad():
            outputs = llm(**encodings, labels=encodings.input_ids)
            ppl_gen = torch.exp(outputs.loss).item()

        return s_sem, ppl_gen

    def _run_single_sample(self, sample: pd.Series, retriever, faiss_index,
                           llm, tokenizer, rag_database: List[str],
                           prompt_style: str = "Regular") -> Dict[str, Any]:
        """Run S2MIA attack on a single sample."""
        query = sample['query_text']
        target = sample['remaining_text']

        contexts = self._retrieve_context(
            query, retriever, faiss_index, rag_database,
            k=self.config['retrieval_k']
        )

        prompt = self._format_prompt(query, contexts, style=prompt_style)
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_outputs = llm.generate(
            **inputs,
            max_new_tokens=self.config['max_new_tokens'],
            do_sample=True,
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
        )

        generated = tokenizer.decode(
            gen_outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        s_sem, ppl_gen = self._calculate_metrics(target, generated, llm, tokenizer, query)

        return {
            "s_sem": s_sem,
            "ppl_gen": ppl_gen,
            "label": 1 if sample['label'] == 'member' else 0
        }

    def _run_attack_pipeline(self, dataset: pd.DataFrame, retriever, faiss_index,
                             llm, tokenizer, rag_database: List[str],
                             desc: str = "Processing",
                             prompt_style: str = "Regular") -> pd.DataFrame:
        """Run attack pipeline on a dataset."""
        results = []
        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=desc):
            try:
                result = self._run_single_sample(
                    row, retriever, faiss_index, llm, tokenizer, rag_database,
                    prompt_style=prompt_style
                )
                results.append(result)
            except Exception as e:
                print(f"    Warning: {e}")
                continue
        return pd.DataFrame(results)

    def _optimize_thresholds(self, scores_df: pd.DataFrame) -> Dict[str, float]:
        """Find optimal BLEU and PPL thresholds via greedy search on reference set."""
        if len(scores_df) == 0:
            return {'bleu': 0, 'ppl': float('inf')}

        bleu_candidates = np.unique(np.percentile(scores_df['s_sem'], np.arange(0, 100, 5)))
        ppl_candidates = np.unique(np.percentile(scores_df['ppl_gen'], np.arange(0, 100, 5)))

        best_score = 0
        best_thresholds = {'bleu': 0, 'ppl': float('inf')}

        for t_bleu in bleu_candidates:
            for t_ppl in ppl_candidates:
                predictions = (
                    (scores_df['s_sem'] >= t_bleu) &
                    (scores_df['ppl_gen'] <= t_ppl)
                ).astype(int)
                score = f1_score(scores_df['label'], predictions, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_thresholds = {'bleu': t_bleu, 'ppl': t_ppl}

        return best_thresholds

    def _evaluate_attack(self, eval_df: pd.DataFrame,
                         thresholds: Dict[str, float]) -> Dict[str, float]:
        """Evaluate attack with thresholds + compute ROC AUC / PR AUC."""
        if len(eval_df) == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
                    'roc_auc': 0, 'pr_auc': 0}

        # Threshold-based predictions (S2MIA-T)
        predictions = (
            (eval_df['s_sem'] >= thresholds['bleu']) &
            (eval_df['ppl_gen'] <= thresholds['ppl'])
        ).astype(int)

        # Continuous membership score for AUC metrics
        # Normalize s_sem and ppl_gen, then combine
        s_sem_vals = eval_df['s_sem'].values
        ppl_vals = eval_df['ppl_gen'].values

        # Normalize to [0, 1] range
        s_range = s_sem_vals.max() - s_sem_vals.min()
        p_range = ppl_vals.max() - ppl_vals.min()
        s_norm = (s_sem_vals - s_sem_vals.min()) / s_range if s_range > 0 else np.zeros_like(s_sem_vals)
        p_norm = (ppl_vals - ppl_vals.min()) / p_range if p_range > 0 else np.zeros_like(ppl_vals)

        # Combined score: high BLEU + low PPL = more likely member
        membership_scores = s_norm + (1 - p_norm)

        labels = eval_df['label'].values
        try:
            r_auc = roc_auc_score(labels, membership_scores)
        except ValueError:
            r_auc = 0.0
        try:
            p_auc = average_precision_score(labels, membership_scores)
        except ValueError:
            p_auc = 0.0

        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
            'roc_auc': r_auc,
            'pr_auc': p_auc,
        }

    # ----- Fixed Test Samples -----

    def _create_fixed_samples(self, ref_dataset: pd.DataFrame,
                              eval_dataset: pd.DataFrame
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create fixed test samples for consistent comparison."""
        limit = self.config['test_sample_limit']

        def sample_balanced(ds, n):
            members = ds[ds['label'] == 'member']
            non_members = ds[ds['label'] == 'non_member']
            n_per = min(n // 2, len(members), len(non_members))
            return pd.concat([
                members.sample(n=n_per, random_state=SEED),
                non_members.sample(n=n_per, random_state=SEED)
            ]).reset_index(drop=True)

        ref_samples = sample_balanced(ref_dataset, limit)
        eval_samples = sample_balanced(eval_dataset, limit)
        print(f"  Fixed samples: {len(ref_samples)} ref, {len(eval_samples)} eval")
        return ref_samples, eval_samples

    def _is_already_completed(self, experiment_group: str, variable_name: str) -> bool:
        """Check if an experiment was already completed successfully."""
        existing = self.sb.get_all_experiments(group=experiment_group)
        if existing.empty:
            return False
        return variable_name in existing[existing['status'] == 'success']['variable_name'].tolist()

    # ----- Single Experiment -----

    def run_single_experiment(
        self,
        experiment_group: str,
        variable_name: str,
        llm_name: str,
        retriever_name: str,
        dataset_name: str,
        membership_ratio: float,
        corpus_size_pct: float,
        prompt_style: str = "Regular",
        # Optional pre-loaded models to avoid redundant loading
        preloaded_retriever=None,
        preloaded_faiss_index=None,
        preloaded_llm=None,
        preloaded_tokenizer=None,
        preloaded_rag_db=None,
        preloaded_ref_samples=None,
        preloaded_eval_samples=None,
    ) -> Dict[str, Any]:
        """Run a single S2MIA experiment and save results to Supabase."""
        exp_label = f"{experiment_group}: {variable_name}"
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_label}")
        print(f"  LLM={llm_name}, Retriever={retriever_name}")
        print(f"  Dataset={dataset_name}, Ratio={membership_ratio}, Corpus={corpus_size_pct}%")
        print(f"  Prompt Style={prompt_style}")
        print(f"{'='*60}")

        start_time = datetime.datetime.now()
        retriever = preloaded_retriever
        faiss_index = preloaded_faiss_index
        llm = preloaded_llm
        tokenizer = preloaded_tokenizer
        rag_database = preloaded_rag_db
        ref_samples = preloaded_ref_samples
        eval_samples = preloaded_eval_samples
        # Track which models we loaded (so we only clean up what we own)
        loaded_retriever = False
        loaded_llm = False

        try:
            # Prepare data if not preloaded
            if rag_database is None:
                rag_database, ref_dataset, eval_dataset = prepare_environment(
                    dataset_name, membership_ratio, corpus_size_pct
                )
                ref_samples, eval_samples = self._create_fixed_samples(ref_dataset, eval_dataset)

            # Load retriever if not preloaded
            if retriever is None or faiss_index is None:
                retriever, faiss_index = self._load_retriever(retriever_name, rag_database)
                loaded_retriever = True

            # Load LLM if not preloaded
            if llm is None or tokenizer is None:
                llm, tokenizer = self._load_llm(llm_name)
                loaded_llm = True

            # --- Reference phase ---
            print("\n  Running Reference Set...")
            ref_scores = self._run_attack_pipeline(
                ref_samples, retriever, faiss_index, llm, tokenizer,
                rag_database, desc="Reference", prompt_style=prompt_style
            )

            thresholds = self._optimize_thresholds(ref_scores)
            print(f"  Thresholds - BLEU: {thresholds['bleu']:.4f}, PPL: {thresholds['ppl']:.4f}")

            # --- Evaluation phase ---
            print("\n  Running Evaluation Set...")
            eval_scores = self._run_attack_pipeline(
                eval_samples, retriever, faiss_index, llm, tokenizer,
                rag_database, desc="Evaluation", prompt_style=prompt_style
            )

            metrics = self._evaluate_attack(eval_scores, thresholds)

            # Add threshold predictions to score DataFrames for storage
            for df in [ref_scores, eval_scores]:
                if len(df) > 0:
                    df['predicted_member'] = (
                        (df['s_sem'] >= thresholds['bleu']) &
                        (df['ppl_gen'] <= thresholds['ppl'])
                    )

            duration = (datetime.datetime.now() - start_time).total_seconds()

            result = {
                'experiment_group': experiment_group,
                'variable_name': variable_name,
                'llm_model': llm_name,
                'retriever_model': retriever_name,
                'dataset_name': dataset_name,
                'corpus_size_pct': corpus_size_pct,
                'membership_ratio': membership_ratio,
                'retrieval_k': self.config['retrieval_k'],
                'test_sample_limit': self.config['test_sample_limit'],
                'ref_samples_run': len(ref_scores),
                'eval_samples_run': len(eval_scores),
                'threshold_bleu': thresholds['bleu'],
                'threshold_ppl': thresholds['ppl'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1'],
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'ref_member_bleu_mean': ref_scores[ref_scores['label']==1]['s_sem'].mean() if len(ref_scores) > 0 else 0,
                'ref_nonmember_bleu_mean': ref_scores[ref_scores['label']==0]['s_sem'].mean() if len(ref_scores) > 0 else 0,
                'ref_member_ppl_mean': ref_scores[ref_scores['label']==1]['ppl_gen'].mean() if len(ref_scores) > 0 else 0,
                'ref_nonmember_ppl_mean': ref_scores[ref_scores['label']==0]['ppl_gen'].mean() if len(ref_scores) > 0 else 0,
                'duration_seconds': duration,
                'status': 'success',
            }

            print(f"\n  RESULTS:")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1 Score:  {metrics['f1']:.4f}")
            print(f"    ROC AUC:   {metrics['roc_auc']:.4f}")
            print(f"    PR AUC:    {metrics['pr_auc']:.4f}")
            print(f"    Duration:  {duration:.1f}s")

            # Save to Supabase
            exp_id = self.sb.save_experiment(result)
            self.sb.save_sample_scores(exp_id, ref_scores, "reference")
            self.sb.save_sample_scores(exp_id, eval_scores, "evaluation")
            print(f"  Saved to Supabase: {exp_id}")

            return result

        except Exception as e:
            duration = (datetime.datetime.now() - start_time).total_seconds()
            result = {
                'experiment_group': experiment_group,
                'variable_name': variable_name,
                'llm_model': llm_name,
                'retriever_model': retriever_name,
                'dataset_name': dataset_name,
                'corpus_size_pct': corpus_size_pct,
                'membership_ratio': membership_ratio,
                'retrieval_k': self.config['retrieval_k'],
                'test_sample_limit': self.config['test_sample_limit'],
                'duration_seconds': duration,
                'status': 'failed',
                'error_message': str(e),
            }
            print(f"  FAILED: {e}")
            self.sb.save_experiment(result)
            return result

        finally:
            # Only cleanup models we loaded ourselves
            cleanup = {}
            if loaded_retriever:
                cleanup['retriever'] = retriever
                cleanup['faiss_index'] = faiss_index
            if loaded_llm:
                cleanup['llm'] = llm
                cleanup['tokenizer'] = tokenizer
            if cleanup:
                print("  Cleaning up locally-loaded models...")
                self._cleanup_models(**cleanup)
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU Memory: {mem:.2f}GB allocated")

    # ----- Experiment Group Runners -----

    def run_vary_llm(self, llm_models: List[str]):
        """Group 1: Vary LLM, fix everything else at baseline.
        Optimization: build FAISS index once, swap only LLM."""
        print(f"\n{'#'*70}")
        print(f"# GROUP 1: VARY LLM ({len(llm_models)} experiments)")
        print(f"{'#'*70}")

        # Prepare data once
        rag_db, ref_ds, eval_ds = prepare_environment(
            BASELINE['dataset'], BASELINE['membership_ratio'], BASELINE['corpus_size_pct']
        )
        ref_samples, eval_samples = self._create_fixed_samples(ref_ds, eval_ds)

        # Build retriever + FAISS once
        retriever, faiss_index = self._load_retriever(BASELINE['retriever'], rag_db)

        for i, llm_name in enumerate(llm_models):
            print(f"\n[{i+1}/{len(llm_models)}]")
            if self._is_already_completed("vary_llm", llm_name):
                  print(f"  Skipping {llm_name} (already completed)")
                  continue
            llm, tokenizer = self._load_llm(llm_name)
            try:
                self.run_single_experiment(
                    experiment_group="vary_llm",
                    variable_name=llm_name,
                    llm_name=llm_name,
                    retriever_name=BASELINE['retriever'],
                    dataset_name=BASELINE['dataset'],
                    membership_ratio=BASELINE['membership_ratio'],
                    corpus_size_pct=BASELINE['corpus_size_pct'],
                    preloaded_retriever=retriever,
                    preloaded_faiss_index=faiss_index,
                    preloaded_llm=llm,
                    preloaded_tokenizer=tokenizer,
                    preloaded_rag_db=rag_db,
                    preloaded_ref_samples=ref_samples,
                    preloaded_eval_samples=eval_samples,
                )
            finally:
                self._cleanup_models(llm=llm, tokenizer=tokenizer)

        # Cleanup shared retriever
        self._cleanup_models(retriever=retriever, faiss_index=faiss_index)

    def run_vary_embedding(self, retriever_models: List[str]):
        """Group 2: Vary retriever/embedding, fix everything else.
        Optimization: load LLM once, rebuild FAISS per retriever."""
        print(f"\n{'#'*70}")
        print(f"# GROUP 2: VARY EMBEDDING ({len(retriever_models)} experiments)")
        print(f"{'#'*70}")

        rag_db, ref_ds, eval_ds = prepare_environment(
            BASELINE['dataset'], BASELINE['membership_ratio'], BASELINE['corpus_size_pct']
        )
        ref_samples, eval_samples = self._create_fixed_samples(ref_ds, eval_ds)

        # Load LLM once
        llm, tokenizer = self._load_llm(BASELINE['llm'])

        for i, ret_name in enumerate(retriever_models):
            if self._is_already_completed("vary_embedding", ret_name):
                  print(f"  Skipping {ret_name} (already completed)")
                  continue
            print(f"\n[{i+1}/{len(retriever_models)}]")
            retriever, faiss_index = self._load_retriever(ret_name, rag_db)
            try:
                self.run_single_experiment(
                    experiment_group="vary_embedding",
                    variable_name=ret_name,
                    llm_name=BASELINE['llm'],
                    retriever_name=ret_name,
                    dataset_name=BASELINE['dataset'],
                    membership_ratio=BASELINE['membership_ratio'],
                    corpus_size_pct=BASELINE['corpus_size_pct'],
                    preloaded_retriever=retriever,
                    preloaded_faiss_index=faiss_index,
                    preloaded_llm=llm,
                    preloaded_tokenizer=tokenizer,
                    preloaded_rag_db=rag_db,
                    preloaded_ref_samples=ref_samples,
                    preloaded_eval_samples=eval_samples,
                )
            finally:
                self._cleanup_models(retriever=retriever, faiss_index=faiss_index)

        self._cleanup_models(llm=llm, tokenizer=tokenizer)

    def run_vary_corpus_size(self, corpus_sizes: List[float]):
        """Group 3: Vary corpus size. Each requires new data + FAISS."""
        print(f"\n{'#'*70}")
        print(f"# GROUP 3: VARY CORPUS SIZE ({len(corpus_sizes)} experiments)")
        print(f"{'#'*70}")

        for i, size_pct in enumerate(corpus_sizes):
            if self._is_already_completed("vary_corpus_size", f"{size_pct}%"):
                  print(f"  Skipping {size_pct}% (already completed)")
                  continue
            print(f"\n[{i+1}/{len(corpus_sizes)}]")
            self.run_single_experiment(
                experiment_group="vary_corpus_size",
                variable_name=f"{size_pct}%",
                llm_name=BASELINE['llm'],
                retriever_name=BASELINE['retriever'],
                dataset_name=BASELINE['dataset'],
                membership_ratio=BASELINE['membership_ratio'],
                corpus_size_pct=size_pct,
            )

    def run_vary_membership_ratio(self, ratios: List[float]):
        """Group 4: Vary membership ratio. Each requires new data split."""
        print(f"\n{'#'*70}")
        print(f"# GROUP 4: VARY MEMBERSHIP RATIO ({len(ratios)} experiments)")
        print(f"{'#'*70}")

        for i, ratio in enumerate(ratios):
            if self._is_already_completed("vary_membership_ratio", f"{ratio:.0%}"):
                  print(f"  Skipping {ratio:.0%} (already completed)")
                  continue
            print(f"\n[{i+1}/{len(ratios)}]")
            self.run_single_experiment(
                experiment_group="vary_membership_ratio",
                variable_name=f"{ratio:.0%}",
                llm_name=BASELINE['llm'],
                retriever_name=BASELINE['retriever'],
                dataset_name=BASELINE['dataset'],
                membership_ratio=ratio,
                corpus_size_pct=BASELINE['corpus_size_pct'],
            )

    def run_vary_dataset(self, datasets: List[str]):
        """Group 5: Vary dataset. Each requires new data loading."""
        print(f"\n{'#'*70}")
        print(f"# GROUP 5: VARY DATASET ({len(datasets)} experiments)")
        print(f"{'#'*70}")

        for i, ds_name in enumerate(datasets):
            if self._is_already_completed("vary_dataset", ds_name):
                  print(f"  Skipping {ds_name} (already completed)")
                  continue
            print(f"\n[{i+1}/{len(datasets)}]")
            self.run_single_experiment(
                experiment_group="vary_dataset",
                variable_name=ds_name,
                llm_name=BASELINE['llm'],
                retriever_name=BASELINE['retriever'],
                dataset_name=ds_name,
                membership_ratio=BASELINE['membership_ratio'],
                corpus_size_pct=BASELINE['corpus_size_pct'],
            )

    def run_vary_prompt_style(self, prompt_styles: List[str]):
        """Group 6: Vary prompt style. All other vars at baseline.
        Optimization: load data, retriever, and LLM once; only the prompt
        formatting changes per experiment."""
        print(f"\n{'#'*70}")
        print(f"# GROUP 6: VARY PROMPT STYLE ({len(prompt_styles)} experiments)")
        print(f"{'#'*70}")

        # Prepare data once
        rag_db, ref_ds, eval_ds = prepare_environment(
            BASELINE['dataset'], BASELINE['membership_ratio'], BASELINE['corpus_size_pct']
        )
        ref_samples, eval_samples = self._create_fixed_samples(ref_ds, eval_ds)

        # Load retriever + FAISS once
        retriever, faiss_index = self._load_retriever(BASELINE['retriever'], rag_db)

        # Load LLM once
        llm, tokenizer = self._load_llm(BASELINE['llm'])

        try:
            for i, style in enumerate(prompt_styles):
                print(f"\n[{i+1}/{len(prompt_styles)}] Prompt style: {style}")

                if self._is_already_completed("vary_prompt_style", style):
                    print(f"  Skipping {style} (already completed)")
                    continue

                try:
                    self.run_single_experiment(
                        experiment_group="vary_prompt_style",
                        variable_name=style,
                        llm_name=BASELINE['llm'],
                        retriever_name=BASELINE['retriever'],
                        dataset_name=BASELINE['dataset'],
                        membership_ratio=BASELINE['membership_ratio'],
                        corpus_size_pct=BASELINE['corpus_size_pct'],
                        prompt_style=style,
                        preloaded_retriever=retriever,
                        preloaded_faiss_index=faiss_index,
                        preloaded_llm=llm,
                        preloaded_tokenizer=tokenizer,
                        preloaded_rag_db=rag_db,
                        preloaded_ref_samples=ref_samples,
                        preloaded_eval_samples=eval_samples,
                    )
                except Exception as e:
                    print(f"  FAILED style {style}: {e}")
        finally:
            # Cleanup
            self._cleanup_models(retriever=retriever, faiss_index=faiss_index,
                                 llm=llm, tokenizer=tokenizer)


# ===== NOTEBOOK CELL 11 =====
tester = SequentialTester(EXPERIMENT_CONFIG, sb_logger)
print("SequentialTester initialized.")
print(f"\nExperiment groups:")
print(f"  1. Vary LLM:              {len(LLM_MODELS)} experiments")
print(f"  2. Vary Embedding:         {len(RETRIEVER_MODELS)} experiments")
print(f"  3. Vary Corpus Size:       {len(CORPUS_SIZES)} experiments")
print(f"  4. Vary Membership Ratio:  {len(MEMBERSHIP_RATIOS)} experiments")
print(f"  5. Vary Dataset:           {len(DATASETS)} experiments")
print(f"  6. Vary Prompt Style:      {len(PROMPT_STYLES)} experiments")
print(f"  Total: {len(LLM_MODELS) + len(RETRIEVER_MODELS) + len(CORPUS_SIZES) + len(MEMBERSHIP_RATIOS) + len(DATASETS) + len(PROMPT_STYLES)}")

# ===== NOTEBOOK CELL 13 =====
# Group 1: Vary LLM (7 experiments)
# Baseline retriever/dataset/ratio/corpus held constant
tester.run_vary_llm(LLM_MODELS)

# ===== NOTEBOOK CELL 14 =====
# Group 2: Vary Embedding Model (7 experiments)
# Baseline LLM/dataset/ratio/corpus held constant
tester.run_vary_embedding(RETRIEVER_MODELS)

# ===== NOTEBOOK CELL 15 =====
# Group 3: Vary Corpus Size (5 experiments)
# Baseline LLM/retriever/dataset/ratio held constant
tester.run_vary_corpus_size(CORPUS_SIZES)

# ===== NOTEBOOK CELL 16 =====
# Group 4: Vary Membership Ratio (4 experiments)
# Baseline LLM/retriever/dataset/corpus held constant
tester.run_vary_membership_ratio(MEMBERSHIP_RATIOS)

# ===== NOTEBOOK CELL 17 =====
# Group 5: Vary Dataset (4 experiments)
# Baseline LLM/retriever/ratio/corpus held constant
tester.run_vary_dataset(DATASETS)

# ===== NOTEBOOK CELL 18 =====
# Group 6: Vary Prompt Style (4 experiments)
# Baseline LLM/retriever/dataset/ratio/corpus held constant
# Only the prompt formatting changes (Regular, Chain Of Thought, RTOC, CRISPE)
tester.run_vary_prompt_style(PROMPT_STYLES)

# ===== NOTEBOOK CELL 20 =====
# --- Visualization Functions ---
import os

# Create visualization outputs directory (relative to experiments/)
RESULTS_DIR = "../results"
VIZ_OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)
print(f"Visualization output directory: {VIZ_OUTPUT_DIR}")

GROUP_LABELS = {
    "vary_llm": "LLM Model",
    "vary_embedding": "Embedding Model",
    "vary_corpus_size": "Corpus Size",
    "vary_membership_ratio": "Membership Ratio",
    "vary_dataset": "Dataset",
    "vary_prompt_style": "Prompt Style",
}


def _short_name(full_name: str) -> str:
    """Extract short model name from HuggingFace path."""
    return full_name.split('/')[-1] if '/' in full_name else full_name


def plot_metrics_by_group(group_name: str, logger: SupabaseLogger):
    """Bar chart of all metrics for a given experiment group."""
    df = logger.get_all_experiments(group=group_name)
    df = df[df['status'] == 'success']
    if df.empty:
        print(f"No successful experiments for group: {group_name}")
        return

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    labels = df['variable_name'].apply(_short_name)

    x = np.arange(len(labels))
    width = 0.12
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 2), 6))

    colors = sns.color_palette('Set2', len(metrics))
    for i, metric in enumerate(metrics):
        values = df[metric].fillna(0).values
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), color=colors[i])

    ax.set_xlabel(GROUP_LABELS.get(group_name, group_name))
    ax.set_ylabel('Score')
    ax.set_title(f'S2MIA Attack Performance — {GROUP_LABELS.get(group_name, group_name)}')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(VIZ_OUTPUT_DIR, f"metrics_{group_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.show()


def plot_distributions(experiment_id: str, logger: SupabaseLogger, title: str = ""):
    """KDE distribution plots for member vs non-member (BLEU + PPL).
    Matches Figure 2 from the paper."""
    scores = logger.get_sample_scores(experiment_id)
    scores = scores[scores['phase'] == 'evaluation']
    if scores.empty:
        print(f"No evaluation scores for experiment: {experiment_id}")
        return

    members = scores[scores['label'] == 1]
    non_members = scores[scores['label'] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BLEU distribution
    ax = axes[0]
    if len(members) > 1:
        sns.kdeplot(members['s_sem'].dropna(), ax=ax, label='Member', fill=True, alpha=0.3)
    if len(non_members) > 1:
        sns.kdeplot(non_members['s_sem'].dropna(), ax=ax, label='Non-Member', fill=True, alpha=0.3)
    ax.set_xlabel('BLEU Score (Semantic Similarity)')
    ax.set_ylabel('Density')
    ax.set_title('BLEU Score Distribution')
    ax.legend()

    # PPL distribution
    ax = axes[1]
    if len(members) > 1:
        sns.kdeplot(members['ppl_gen'].dropna(), ax=ax, label='Member', fill=True, alpha=0.3)
    if len(non_members) > 1:
        sns.kdeplot(non_members['ppl_gen'].dropna(), ax=ax, label='Non-Member', fill=True, alpha=0.3)
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Density')
    ax.set_title('Perplexity Distribution')
    ax.legend()

    fig.suptitle(title or f'Member vs Non-Member Distributions', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    safe_title = title.replace(' ', '_').replace('/', '_') if title else experiment_id
    output_path = os.path.join(VIZ_OUTPUT_DIR, f"distributions_{safe_title}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.show()


def plot_distributions_by_group(group_name: str, logger: SupabaseLogger):
    """Bar chart of mean BLEU and Perplexity for members vs non-members across all experiments in a group.
    Matches the same format as plot_metrics_by_group."""
    df = logger.get_all_experiments(group=group_name)
    df = df[df['status'] == 'success']
    if df.empty:
        print(f"No successful experiments for group: {group_name}")
        return

    labels = df['variable_name'].apply(_short_name).tolist()

    # Collect mean BLEU and PPL for members and non-members from each experiment
    member_bleu_means = []
    nonmember_bleu_means = []
    member_ppl_means = []
    nonmember_ppl_means = []

    for _, row in df.iterrows():
        scores = logger.get_sample_scores(row['id'])
        eval_scores = scores[scores['phase'] == 'evaluation']

        if eval_scores.empty:
            member_bleu_means.append(0)
            nonmember_bleu_means.append(0)
            member_ppl_means.append(0)
            nonmember_ppl_means.append(0)
            continue

        members = eval_scores[eval_scores['label'] == 1]
        non_members = eval_scores[eval_scores['label'] == 0]

        member_bleu_means.append(members['s_sem'].mean() if len(members) > 0 else 0)
        nonmember_bleu_means.append(non_members['s_sem'].mean() if len(non_members) > 0 else 0)
        member_ppl_means.append(members['ppl_gen'].mean() if len(members) > 0 else 0)
        nonmember_ppl_means.append(non_members['ppl_gen'].mean() if len(non_members) > 0 else 0)

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(labels) * 2), 6))

    # BLEU Score Distribution
    ax = axes[0]
    colors = sns.color_palette('Set2', 2)
    ax.bar(x - width/2, member_bleu_means, width, label='Member', color=colors[0])
    ax.bar(x + width/2, nonmember_bleu_means, width, label='Non-Member', color=colors[1])
    ax.set_xlabel(GROUP_LABELS.get(group_name, group_name))
    ax.set_ylabel('Mean BLEU Score')
    ax.set_title(f'BLEU Score Distribution — {GROUP_LABELS.get(group_name, group_name)}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Perplexity Distribution
    ax = axes[1]
    ax.bar(x - width/2, member_ppl_means, width, label='Member', color=colors[0])
    ax.bar(x + width/2, nonmember_ppl_means, width, label='Non-Member', color=colors[1])
    ax.set_xlabel(GROUP_LABELS.get(group_name, group_name))
    ax.set_ylabel('Mean Perplexity')
    ax.set_title(f'Perplexity Distribution — {GROUP_LABELS.get(group_name, group_name)}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(VIZ_OUTPUT_DIR, f"distributions_{group_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.show()


def _plot_summary_single(metric_key, metric_label, groups, group_data):
    """Helper to plot a single summary bar chart for one metric.
    Only shows bars for variables that exist in each group, eliminating empty gaps."""
    # Build flat list of bars: [(group_idx, var_name, value), ...]
    bar_entries = []
    group_centers = []
    pos = 0
    bar_width = 0.6
    gap_between_groups = 1.0

    for gi, group in enumerate(groups):
        names = group_data[group]['names']
        values = [group_data[group][metric_key].get(n, 0) for n in names]
        group_start = pos
        for vi, (name, val) in enumerate(zip(names, values)):
            bar_entries.append((pos, name, val))
            pos += bar_width
        group_centers.append((group_start + pos - bar_width) / 2)
        pos += gap_between_groups

    fig_width = max(10, pos * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    colors = sns.color_palette('tab10', 20)
    # Assign consistent color per variable name
    all_names = list(dict.fromkeys(n for g in groups for n in group_data[g]['names']))
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(all_names)}

    for x_pos, name, val in bar_entries:
        bar = ax.bar(x_pos, val, bar_width * 0.9, color=color_map[name])
        if val > 0:
            ax.text(x_pos, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel('Experiment Group')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} Across Experiment Groups')
    ax.set_xticks(group_centers)
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups], rotation=45, ha='right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    safe_key = metric_key.replace(' ', '_')
    output_path = os.path.join(VIZ_OUTPUT_DIR, f"summary_{safe_key}_barchart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.show()


def plot_summary_barchart(logger: SupabaseLogger):
    """Individual grouped bar charts of ROC AUC and PR AUC scores across all experiment groups."""
    all_df = logger.get_all_experiments()
    all_df = all_df[all_df['status'] == 'success']
    if all_df.empty:
        print("No experiments found.")
        return

    groups = all_df['experiment_group'].unique()

    # Collect data per group
    group_data = {}
    for group in groups:
        gdf = all_df[all_df['experiment_group'] == group].sort_values('created_at')
        names = gdf['variable_name'].apply(_short_name).tolist()
        roc_aucs = gdf['roc_auc'].fillna(0).tolist()
        pr_aucs = gdf['pr_auc'].fillna(0).tolist()
        group_data[group] = {
            'names': names,
            'roc_auc': dict(zip(names, roc_aucs)),
            'pr_auc': dict(zip(names, pr_aucs)),
        }

    # Plot each metric as its own figure
    _plot_summary_single('roc_auc', 'ROC AUC', groups, group_data)
    _plot_summary_single('pr_auc', 'PR AUC', groups, group_data)


def plot_roc_curves_by_group(group_name: str, logger: SupabaseLogger):
    """Overlay ROC curves for all experiments in a group."""
    df = logger.get_all_experiments(group=group_name)
    df = df[df['status'] == 'success']
    if df.empty:
        print(f"No experiments for group: {group_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette('tab10', len(df))

    for idx, (_, row) in enumerate(df.iterrows()):
        scores = logger.get_sample_scores(row['id'])
        eval_scores = scores[scores['phase'] == 'evaluation']
        if eval_scores.empty or len(eval_scores) < 2:
            continue

        labels = eval_scores['label'].values
        s_sem = eval_scores['s_sem'].values
        ppl = eval_scores['ppl_gen'].values

        # Compute combined score
        s_range = s_sem.max() - s_sem.min()
        p_range = ppl.max() - ppl.min()
        s_norm = (s_sem - s_sem.min()) / s_range if s_range > 0 else np.zeros_like(s_sem)
        p_norm = (ppl - ppl.min()) / p_range if p_range > 0 else np.zeros_like(ppl)
        membership_scores = s_norm + (1 - p_norm)

        try:
            fpr, tpr, _ = roc_curve(labels, membership_scores)
            auc_val = roc_auc_score(labels, membership_scores)
            label = f"{_short_name(row['variable_name'])} (AUC={auc_val:.3f})"
            ax.plot(fpr, tpr, color=colors[idx], label=label, linewidth=1.5)
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {GROUP_LABELS.get(group_name, group_name)}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(VIZ_OUTPUT_DIR, f"roc_curves_{group_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.show()


def display_summary_table(logger: SupabaseLogger):
    """Display a formatted summary table of all experiments."""
    df = logger.get_all_experiments()
    if df.empty:
        print("No experiments found.")
        return df

    df = df[df['status'] == 'success'].copy()
    df['variable_short'] = df['variable_name'].apply(_short_name)

    display_cols = [
        'experiment_group', 'variable_short',
        'accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'pr_auc', 'duration_seconds'
    ]
    summary = df[display_cols].copy()
    summary.columns = [
        'Group', 'Variable',
        'Accuracy', 'Precision', 'Recall', 'F1',
        'ROC AUC', 'PR AUC', 'Duration (s)'
    ]

    # Format numeric columns
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'PR AUC']:
        summary[col] = summary[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    summary['Duration (s)'] = summary['Duration (s)'].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    
    # Save summary table to CSV
    output_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return summary


print("Visualization functions loaded.")

# ===== NOTEBOOK CELL 21 =====
# --- Generate All Visualizations ---

# Summary table
summary = display_summary_table(sb_logger)
display(summary)

# ===== NOTEBOOK CELL 22 =====
# Bar charts for each group
for group in ["vary_llm", "vary_embedding", "vary_corpus_size", "vary_membership_ratio", "vary_dataset", "vary_prompt_style"]:
    plot_metrics_by_group(group, sb_logger)

# ===== NOTEBOOK CELL 23 =====
# Distribution plots (member vs non-member)
for group in ["vary_llm", "vary_embedding", "vary_corpus_size", "vary_membership_ratio", "vary_dataset", "vary_prompt_style"]:
    plot_distributions_by_group(group, sb_logger)

# ===== NOTEBOOK CELL 24 =====
# ROC AUC and PR AUC grouped bar charts across all groups
plot_summary_barchart(sb_logger)

# ===== NOTEBOOK CELL 25 =====
# ROC curves for each group
for group in ["vary_llm", "vary_embedding", "vary_corpus_size", "vary_membership_ratio", "vary_dataset", "vary_prompt_style"]:
    plot_roc_curves_by_group(group, sb_logger)
