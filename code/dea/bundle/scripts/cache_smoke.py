"""Exercise a pinned cached checkpoint using harmless synthetic text."""

from __future__ import annotations

import hashlib
import math
import re
import time

MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
REVISION = re.compile(r"[0-9a-f]{40}")
FIELDS = {"model_id", "revision", "tokenizer_revision", "dtype",
          "chat_template_sha256", "template_options"}


def validate_smoke_settings(spec):
    if (not isinstance(spec, dict) or spec.keys() != FIELDS
            or not MODEL_ID.fullmatch(spec["model_id"])
            or any(not REVISION.fullmatch(spec[key]) for key in ("revision", "tokenizer_revision"))
            or spec["dtype"] not in ("fp16", "bf16", "fp32")
            or not isinstance(spec["template_options"], dict)
            or spec["template_options"].keys() - {"enable_thinking"}
            or any(type(value) is not bool for value in spec["template_options"].values())):
        raise ValueError("invalid cache-smoke settings")
    chat_hash = spec["chat_template_sha256"]
    if chat_hash is None and spec["template_options"]:
        raise ValueError("chat options require a pinned template")
    if chat_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", chat_hash):
        raise ValueError("invalid chat template hash")


def validate_smoke_result(result):
    evidence = {"synthetic_only", "input_tokens", "generated_tokens", "peak_allocated_mib", "seconds"}
    if (result.keys() != FIELDS | evidence or result["synthetic_only"] is not True
            or type(result["input_tokens"]) is not int or not 2 <= result["input_tokens"] <= 1024
            or type(result["generated_tokens"]) is not int or not 1 <= result["generated_tokens"] <= 4
            or any(type(result[key]) not in (int, float) or not math.isfinite(result[key]) or result[key] <= 0
                   for key in ("peak_allocated_mib", "seconds"))):
        raise ValueError("cache smoke summary lacks valid execution evidence")
    validate_smoke_settings({key: result[key] for key in FIELDS})


def run_smoke(spec):
    validate_smoke_settings(spec)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    options = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["tokenizer_revision"], **options)
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is None:
        raise ValueError("cached tokenizer requires an EOS or padding token")
    if spec["chat_template_sha256"] is not None:
        template = tokenizer.get_chat_template()
        if hashlib.sha256(template.encode()).hexdigest() != spec["chat_template_sha256"]:
            raise ValueError("cached chat template differs from the pinned template")
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Return a short greeting."}], chat_template=template,
            tokenize=True, add_generation_prompt=True, return_tensors="pt", **spec["template_options"],
        )
    else:
        ids = tokenizer("A small synthetic sentence for a software check.",
                        return_tensors="pt", return_token_type_ids=False)["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not 2 <= ids.shape[1] <= 1024:
        raise ValueError("smoke input has an unexpected token shape")
    ids = ids.to("cuda:0")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[spec["dtype"]]
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(spec["model_id"], revision=spec["revision"],
                                                torch_dtype=dtype, **options).to("cuda:0").eval()
    mask = torch.ones_like(ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=4,
                                   do_sample=False, pad_token_id=pad_id)
    width = ids.shape[1]
    if (not bool(torch.isfinite(logits).all()) or not torch.equal(generated[:, :width], ids)
            or not width < generated.shape[1] <= width + 4):
        raise ValueError("checkpoint failed the finite-logit or continuation check")
    result = dict(spec, synthetic_only=True, input_tokens=width,
                generated_tokens=generated.shape[1] - width,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                seconds=time.monotonic() - started)
    validate_smoke_result(result)
    return result
