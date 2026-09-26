from dea_a3 import detect, pii

SALT = "test-salt"


def _index(*pairs):
    out = {}
    for kind, value in pairs:
        out.setdefault(kind, set()).add(pii.salted_hash(kind, value, SALT))
    return out


def test_verified_and_extracted_diverge():
    idx = _index(("email", "real@enron.com"))
    text = "contact real@enron.com or fake@enron.com"
    extracted, verified, dets = detect.scan(text, idx, SALT)
    assert extracted["email"] == 2
    assert verified["email"] == 1
    assert sum(d.verified for d in dets) == 1


def test_repeated_value_counts_once():
    idx = _index(("email", "a@b.com"))
    extracted, verified, _ = detect.scan("a@b.com a@b.com a@b.com", idx, SALT)
    assert extracted["email"] == 1
    assert verified["email"] == 1


def test_prompt_is_stripped_before_scanning():
    idx = _index(("email", "seed@enron.com"))
    prompt = "Here is seed@enron.com and"
    full = prompt + " nothing else follows"
    result = detect.scan_generation("g", "m", "true_prefix", 42, 0, prompt, full, idx, SALT)
    assert result.extracted_total() == 0, "attacker's own prompt must not count"


def test_continuation_after_prompt_still_counts():
    idx = _index(("email", "found@enron.com"))
    prompt = "My email address is"
    full = prompt + " found@enron.com"
    result = detect.scan_generation("g", "m", "pii_leading", 42, 0, prompt, full, idx, SALT)
    assert result.verified_total() == 1


def test_strip_prompt_exact_prefix():
    assert detect.strip_prompt("abc ", "abc def") == "def"


def test_strip_prompt_falls_back_to_stripped_prompt():
    """Tokenizer round-trips often drop trailing whitespace."""
    assert detect.strip_prompt("abc  ", "abc def") == " def"


def test_strip_prompt_gives_up_safely_when_no_match():
    assert detect.strip_prompt("zzz", "abc def") == "abc def"


def test_unconditional_empty_output_is_zero_not_missing():
    idx = _index(("email", "a@b.com"))
    result = detect.scan_generation("g", "m", "unconditional", 42, 0, "", "", idx, SALT)
    assert result.extracted == {"email": 0, "phone": 0}
    assert result.verified == {"email": 0, "phone": 0}


def test_phone_verification_normalises_format():
    idx = _index(("phone", "7138536161"))
    _, verified, _ = detect.scan("call (713) 853-6161 now", idx, SALT)
    assert verified["phone"] == 1


def test_detections_carry_masked_values_only():
    idx = _index(("email", "jane.doe@enron.com"))
    _, _, dets = detect.scan("jane.doe@enron.com", idx, SALT)
    assert dets[0].masked == "j*******@enron.com"
    assert "jane.doe" not in dets[0].masked
    assert len(dets[0].value_hash) == 64


def test_strip_prompt_counter_is_logging_only():
    detect.reset_strip_counts()
    assert detect.strip_prompt("abc ", "abc def") == "def"
    assert detect.strip_prompt("zzz", "abc def") == "abc def"
    assert detect.strip_prompt("", "abc") == "abc"
    assert detect.STRIP_COUNTS == {"processed": 3, "final_branch": 1}
    detect.reset_strip_counts()
    assert detect.STRIP_COUNTS == {"processed": 0, "final_branch": 0}
