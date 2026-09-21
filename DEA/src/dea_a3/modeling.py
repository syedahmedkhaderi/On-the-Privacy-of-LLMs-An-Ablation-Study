"""Model loading and batched, seeded generation.

Batching is not an optimisation here, it is what makes the study affordable.
CRELT generated one prompt per call and spent 3.3 to 5.7 seconds per 64-token
generation; batched, GPT-J produces a 256-token generation in about a second.

Every batch reseeds torch's global RNG from the cell identity and the batch
index immediately before generate(), so a batch reproduces regardless of how
many draws happened earlier in the process. transformers samples on the global
RNG and its generate() rejects a `generator` keyword, so this is the only route
that is both reproducible and valid.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import DecoderSpec, ModelSpec
from .seeds import batch_seed, seed_torch_for_batch


@dataclass
class LoadedModel:
    key: str
    model: object
    tokenizer: object
    device: str


def vram_free_total_gb() -> list[tuple[float, float]]:
    """(free, total) in GB for every visible CUDA device.

    Visible means after CUDA_VISIBLE_DEVICES, so pinning the study to one card
    makes this report that card only. Returns [] when there is no GPU.
    """
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        out.append((free / 1e9, total / 1e9))
    return out


def estimate_weights_gb(spec: ModelSpec) -> float:
    """fp16 weight footprint. Two bytes per parameter, params_m in millions."""
    return spec.params_m * 2 / 1000.0


def estimate_peak_gb(spec: ModelSpec, batch_size: int,
                     max_new_tokens: int = 256, max_prompt_tokens: int = 128) -> float:
    """Rough ceiling for weights plus KV cache plus context, in GB.

    Deliberately generous. The cost of overestimating is a skipped model and a
    line in a report; the cost of underestimating is an OOM that can also take
    down whatever else is sharing the card.
    """
    weights = estimate_weights_gb(spec)
    # KV cache scales with layers x hidden, both of which track parameter count
    # closely enough for a guard. Two tensors, fp16, per token per layer.
    tokens = batch_size * (max_prompt_tokens + max_new_tokens)
    kv = weights * tokens / 60000.0
    context_and_allocator = 1.2
    return weights + kv + context_and_allocator


def check_vram_headroom(spec: ModelSpec, batch_size: int, reserve_gb: float = 1.0,
                        max_new_tokens: int = 256) -> tuple[bool, str]:
    """Whether this model should be loaded onto the currently visible GPUs.

    `reserve_gb` is headroom left for anything else sharing the card. A sharded
    model (device_map set) is checked against the sum across visible devices;
    an unsharded one against the largest single device, because it cannot be
    split.
    """
    devices = vram_free_total_gb()
    if not devices:
        return True, "no CUDA visible, nothing to check"

    need = estimate_peak_gb(spec, batch_size, max_new_tokens) + reserve_gb
    if spec.device_map:
        have = sum(free for free, _ in devices)
        where = f"{len(devices)} visible devices"
    else:
        have = max(free for free, _ in devices)
        where = "the largest visible device"

    detail = (f"{spec.key}: needs about {need:.1f} GB "
              f"(weights {estimate_weights_gb(spec):.1f}, batch {batch_size}), "
              f"{have:.1f} GB free on {where}")
    return have >= need, detail


def load_model(spec: ModelSpec, cache_dir: str | None = None) -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    if spec.revision:
        kwargs["revision"] = spec.revision
    if spec.device_map:
        kwargs["device_map"] = spec.device_map
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    tok_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    model.eval()

    if spec.device_map:
        device = str(next(model.parameters()).device)
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
    return LoadedModel(key=spec.key, model=model, tokenizer=tokenizer, device=device)


def free_model(loaded: LoadedModel) -> None:
    import gc

    import torch

    del loaded.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Keys transformers' GenerationConfig recognises. Anything outside this set
# survives into model_kwargs and is rejected by _validate_model_kwargs with
# "The following `model_kwargs` are not used by the model", which kills the
# first batch of a run. `generator` is the specific trap: torch.Generator is
# accepted by many torch APIs but not by generate().
GENERATE_ALLOWED_KEYS = frozenset({
    "max_new_tokens", "do_sample", "temperature", "top_k", "top_p",
    "repetition_penalty", "pad_token_id", "eos_token_id",
})


def build_generation_kwargs(decoder: DecoderSpec, pad_token_id) -> dict:
    """The keyword arguments handed to model.generate().

    Separated from generate_batch so it can be checked without torch. See
    tests/test_modeling.py.
    """
    kwargs = {
        "max_new_tokens": decoder.max_new_tokens,
        "do_sample": decoder.do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": None,
    }
    if decoder.do_sample:
        kwargs.update({
            "temperature": decoder.temperature,
            "top_k": decoder.top_k,
            "top_p": decoder.top_p,
            "repetition_penalty": decoder.repetition_penalty,
        })
    unknown = set(kwargs) - GENERATE_ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"generate() would reject these keyword arguments: {sorted(unknown)}"
        )
    return kwargs


def generate_batch(loaded: LoadedModel, prompts: list[str], decoder: DecoderSpec,
                   seed: int, max_prompt_tokens: int = 128) -> list[str]:
    """Generate one batch. Returns decoded full sequences, prompt included."""
    import torch

    tokenizer, model = loaded.tokenizer, loaded.model
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_device = loaded.device
    enc = {k: v.to(input_device) for k, v in enc.items()}

    gen_kwargs = build_generation_kwargs(decoder, tokenizer.pad_token_id)
    if decoder.do_sample:
        # Not gen_kwargs["generator"]: generate() rejects unknown keywords.
        seed_torch_for_batch(seed)

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def iter_batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield start // size, items[start:start + size]


def cell_batch_seeds(cell: int, n_items: int, batch_size: int) -> list[int]:
    """The seed each batch of a cell will use. Exposed for testing."""
    n_batches = (n_items + batch_size - 1) // batch_size
    return [batch_seed(cell, i) for i in range(n_batches)]


def reset_peak_vram() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float | None:
    """Peak allocated since the last reset, in GB. None without a GPU.

    This is what P10 asked for and could never report: no run has ever
    measured its own memory.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e9
