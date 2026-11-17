# H10 EDIT: file header
# S2MIA replication script exported from s2mia_sequential_ablation.ipynb (sha12 f01e8e857b0c)
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
from transformers import AutoTokenizer, AutoModelForCausalLM
# H10 EDIT: BitsAndBytesConfig, matplotlib, seaborn, supabase imports removed
from tqdm import tqdm
import argparse, pathlib, time, hashlib, socket, sys

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

# H10 EDIT: HF_TOKEN assertion and Supabase credentials removed (offline, local cache, local writer)
# H10 EDIT: command-line arguments
_ap = argparse.ArgumentParser()
_ap.add_argument("--dataset", required=True, choices=["nq_open", "trivia_qa", "squad", "hotpot_qa"])
_ap.add_argument("--llm", required=True)
_ap.add_argument("--llm-revision", required=True)
_ap.add_argument("--seed", type=int, required=True)
_ap.add_argument("--eval-per-class", type=int, default=250)
_ap.add_argument("--ref-per-class", type=int, default=50)
_ap.add_argument("--out-dir", required=True)
_ap.add_argument("--run-id", required=True)
_ap.add_argument("--llm-cache", default="work/.cache/huggingface/hub")
_ap.add_argument("--retriever-path", required=True)
_ap.add_argument("--data-root", default="work/cyclea-protected/h10/models")
_ap.add_argument("--emb-cache", default="work/cyclea-protected/h10/s2mia")
ARGS = _ap.parse_args()

# Reproducibility
# H10 EDIT: seed from the command line, applied to random, numpy and torch at the start of each run
# (SEED is also the random_state of the notebook's train_test_split / sample calls, so it changes the split)
SEED = ARGS.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
    "test_sample_limit": 100,    # H10: kept for the record; ref uses 2*ref-per-class, eval uses 2*eval-per-class (see _create_fixed_samples)
    "retrieval_k": 5,            # top-k documents to retrieve
    "max_new_tokens": 100,       # max generation length
    "temperature": 0.7,          # generation temperature
    "top_p": 0.9,                # nucleus sampling
    "batch_size": 64,            # H10 EDIT: encoding batch size 8 -> 64 (speed only)
    "quantization_bits": None,   # H10 EDIT: float16, no quantization
}

print(f"\nBaseline: {BASELINE}")
print(f"Prompt styles: {PROMPT_STYLES}")
print(f"Total experiments: {len(LLM_MODELS)} + {len(RETRIEVER_MODELS)} + {len(CORPUS_SIZES)} + {len(MEMBERSHIP_RATIOS)} + {len(DATASETS)} + {len(PROMPT_STYLES)} = {len(LLM_MODELS) + len(RETRIEVER_MODELS) + len(CORPUS_SIZES) + len(MEMBERSHIP_RATIOS) + len(DATASETS) + len(PROMPT_STYLES)}")

# ===== NOTEBOOK CELL 6 =====
def _local_split(dataset_name: str):
    """H10 EDIT: read the train split from the pinned local parquet snapshot (same rows load_dataset(..., split='train') returns)."""
    sub_ = {"nq_open": ("nq_open", "nq_open/train-*.parquet"), "squad": ("squad", "plain_text/train-*.parquet"),
            "hotpot_qa": ("hotpot_qa", "fullwiki/train-*.parquet"), "trivia_qa": ("trivia_qa", "rc.nocontext/train-*.parquet")}[dataset_name]
    import glob
    files = sorted(glob.glob(f"{ARGS.data_root}/hf-cache-{sub_[0]}-*/datasets--*/snapshots/*/{sub_[1]}"))
    assert files, f"no local parquet for {dataset_name}"
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).to_dict("records")


def _load_raw_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a HuggingFace QA dataset and extract query/answer pairs."""
    processed_data = []

    if dataset_name == "nq_open":
        dataset = _local_split("nq_open")  # H10 EDIT
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
        dataset = _local_split("trivia_qa")  # H10 EDIT
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
        dataset = _local_split("squad")  # H10 EDIT
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
        dataset = _local_split("hotpot_qa")  # H10 EDIT
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
    global _FULL_DF
    _FULL_DF = df  # H10 EDIT: kept for the embedding cache
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
    """H10 EDIT: local JSON/CSV writer with the same method names (no network, no text stored)."""

    def __init__(self, out_dir: str):
        self.out = pathlib.Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict) -> str:
        (self.out / "config.json").write_text(json.dumps(config, indent=1, default=str)); return "local"

    def save_experiment(self, result: dict) -> str:
        (self.out / "result.json").write_text(json.dumps(result, indent=1, default=str)); return "local"

    def save_sample_scores(self, experiment_id: str, scores_df: pd.DataFrame, phase: str):
        if len(scores_df) == 0:
            return
        d = scores_df.copy(); d["phase"] = phase
        # combined score exactly as in _evaluate_attack (normalised within the phase's own samples)
        sv, pv = d["s_sem"].values, d["ppl_gen"].values
        sr, pr = sv.max() - sv.min(), pv.max() - pv.min()
        sn = (sv - sv.min()) / sr if sr > 0 else np.zeros_like(sv)
        pn = (pv - pv.min()) / pr if pr > 0 else np.zeros_like(pv)
        d["combined"] = sn + (1 - pn)
        d.to_csv(self.out / f"sample_scores_{phase}.csv", index=False)

    def get_all_experiments(self, group=None) -> pd.DataFrame:
        return pd.DataFrame()

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
        """Load retriever and build FAISS index over the RAG database.
        H10 EDIT: retriever loaded from the pinned local path; document embeddings are looked up in a per-dataset
        cache of the whole dataframe (identical vectors, computed once with batch 64) instead of re-encoding."""
        print(f"  Loading retriever: {ARGS.retriever_path}")
        retriever = SentenceTransformer(ARGS.retriever_path, device=self.device)
        self._retriever_obj = retriever
        embeddings = self._db_embeddings(retriever, rag_database).copy()
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)
        print(f"  FAISS index built: {faiss_index.ntotal} docs, dim={dimension}")
        return retriever, faiss_index

    def _db_embeddings(self, retriever, rag_database):
        full = _FULL_DF["full_sample"].tolist()
        path = pathlib.Path(ARGS.emb_cache) / f"emb_{ARGS.dataset}.npy"
        if path.exists():
            allemb = np.load(path)
        else:
            allemb = retriever.encode(full, batch_size=self.config['batch_size'], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            path.parent.mkdir(parents=True, exist_ok=True); np.save(path, allemb)
        pos = {}
        for i, t in enumerate(full):
            pos.setdefault(t, []).append(i)
        return np.stack([allemb[pos[t][0]] for t in rag_database])

    def _load_llm(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """H10 EDIT: load the LLM in float16 without quantization, offline from the local cache at a pinned revision."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, revision=ARGS.llm_revision,
                                                  cache_dir=ARGS.llm_cache, local_files_only=True)
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
            if hasattr(tokenizer, 'eos_token'):
                tokenizer.pad_token = tokenizer.eos_token

        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            revision=ARGS.llm_revision, cache_dir=ARGS.llm_cache, local_files_only=True,
            trust_remote_code=False,
        ).to(self.device)
        llm.eval()
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
            "label": 1 if sample['label'] == 'member' else 0,
            "gen_chars": len(generated),  # H10 EDIT: length only, no text
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
        # H10 EDIT: reference and evaluation sizes are set separately (ref 50+50, eval N+N from the command line)
        ref_limit = 2 * ARGS.ref_per_class
        eval_limit = 2 * ARGS.eval_per_class

        def sample_balanced(ds, n):
            members = ds[ds['label'] == 'member']
            non_members = ds[ds['label'] == 'non_member']
            n_per = min(n // 2, len(members), len(non_members))
            return pd.concat([
                members.sample(n=n_per, random_state=SEED),
                non_members.sample(n=n_per, random_state=SEED)
            ]).reset_index(drop=True)

        ref_samples = sample_balanced(ref_dataset, ref_limit)
        eval_samples = sample_balanced(eval_dataset, eval_limit)
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
        """Run a single S2MIA experiment and save results locally (H10 EDIT)."""
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

            # Save (H10 EDIT: local writer)
            exp_id = self.sb.save_experiment(result)
            self.sb.save_sample_scores(exp_id, ref_scores, "reference")
            self.sb.save_sample_scores(exp_id, eval_scores, "evaluation")
            print(f"  Saved locally: {exp_id}")

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

    # H10 EDIT: experiment-group runners (run_vary_*) removed; main() below runs one experiment


# ===================== H10 EDIT: entry point =====================
def main():
    t0 = time.time()
    out = pathlib.Path(ARGS.out_dir)
    logger = SupabaseLogger(ARGS.out_dir)
    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    logger.save_config({"args": vars(ARGS), "experiment_config": EXPERIMENT_CONFIG, "code_sha256": code_sha,
                        "host": socket.gethostname(), "torch": torch.__version__,
                        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")})
    tester = SequentialTester(EXPERIMENT_CONFIG, logger)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res = tester.run_single_experiment("s2mia_rep", f"{ARGS.llm}|{ARGS.dataset}|s{ARGS.seed}", ARGS.llm,
                                       "facebook/contriever-msmarco", ARGS.dataset, 0.8, 100.0, prompt_style="Regular")
    extra = {"run_id": ARGS.run_id, "seed": ARGS.seed, "llm_revision": ARGS.llm_revision, "code_sha256": code_sha,
             "eval_per_class_requested": ARGS.eval_per_class, "ref_per_class_requested": ARGS.ref_per_class,
             "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
             "wall_seconds": time.time() - t0}
    if res.get("status") == "success":
        ev = pd.read_csv(out / "sample_scores_evaluation.csv")
        try:
            extra["roc_auc_bleu_only"] = float(roc_auc_score(ev["label"], ev["s_sem"]))
            extra["roc_auc_ppl_only"] = float(roc_auc_score(ev["label"], -ev["ppl_gen"]))
        except ValueError:
            pass
        extra["eval_n_member"] = int((ev["label"] == 1).sum()); extra["eval_n_nonmember"] = int((ev["label"] == 0).sum())
        extra["majority_rate"] = float(max(ev["label"].mean(), 1 - ev["label"].mean()))
    res.update(extra)
    logger.save_experiment(res)
    return 0 if res.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
