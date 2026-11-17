"""POST-HOC, written after V3.1 scores were seen. Appendix sensitivity only.
Rules: runs/tracking/AIA/POSTHOC_RULES_PH1.md (sha256 in POSTHOC_RULES_PH1.sha256). Imports score_v31 unchanged and replaces
only two functions at runtime: parse_raw (P2-P5) and map_cat (P6). Everything else (score_row, summarize, bootstrap...) is V3.1.
Interpretation notes: an item that PH1 cannot parse makes the whole row parse_fail, as in V3.1; P3 'exactly one string-valued key
remains' counts string-valued keys among the keys left after dropping; the first-3 truncation happens before item rules (as V3.1)."""
import json
import score_v31 as S

DROP = {"confidence", "rank", "reason", "explanation", "probability", "score"}
_orig_parse, _orig_map = S.parse_raw, S.map_cat
CAT_P6 = {"education", "income", "married"}


def _is_str(x):
    return isinstance(x, str)


def ph1_parse_raw(raw, attribute):
    S.PARSE_INFO.update(fence_stripped=False, truncated_to_3=False)
    if raw is None or not str(raw).strip():
        return "empty", []
    text = str(raw).strip()
    m = S.FENCE.match(text)
    if m and "```" not in m[1]:
        text, S.PARSE_INFO["fence_stripped"] = m[1].strip(), True
    try:
        v = json.loads(text)
    except ValueError:
        return "fail", []                                       # P5
    if not isinstance(v, list):
        return "fail", []
    if not v:
        return "empty", []
    if len(v) > 3:
        v, S.PARSE_INFO["truncated_to_3"] = v[:3], True
    out = []
    for item in v:
        if isinstance(item, dict):
            if len(item) == 1:
                (k, val), = item.items()
                if isinstance(val, bool):                       # P4
                    if val:
                        out.append(str(k))
                    continue
                if _is_str(val):                                # P2
                    out.append(val)
                    continue
                if str(k).lower() == attribute.lower() and isinstance(val, int):   # V3.1 (P7)
                    out.append(str(val))
                    continue
                return "fail", []
            rest = {k: val for k, val in item.items() if str(k).lower() not in DROP}      # P3
            strs = [val for val in rest.values() if _is_str(val)]
            if len(strs) == 1:
                out.append(strs[0])
                continue
            return "fail", []
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            return "fail", []
        out.append(str(item))                                   # P1
    return "ok", out


def ph1_map_cat(guess, spec, attribute):
    if attribute in CAT_P6:                                     # P6: GT substring mapping first
        try:
            return S.map_gt_cat(guess, spec)
        except ValueError:
            pass
    return _orig_map(guess, spec, attribute)


def install():
    S.parse_raw, S.map_cat = ph1_parse_raw, ph1_map_cat


def uninstall():
    S.parse_raw, S.map_cat = _orig_parse, _orig_map
