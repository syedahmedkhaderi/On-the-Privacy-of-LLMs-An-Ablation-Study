"""POST-HOC, written after V3.1 and PH1 scores were seen. Final appendix sensitivity; supersedes PH1 for reporting, PH1 kept for audit.
Rules: runs/tracking/AIA/POSTHOC_RULES_PH2.md (sha256 in POSTHOC_RULES_PH2.sha256). Imports the PH1 module unchanged and adds only P8."""
import json
import score_v31 as S
import score_v31_posthoc as PH1

_DEC = json.JSONDecoder()


def ph2_parse_raw(raw, attribute):
    if raw is None or not str(raw).strip():
        return PH1.ph1_parse_raw(raw, attribute)
    text = str(raw).strip()
    fenced = False
    m = S.FENCE.match(text)
    if m and "```" not in m[1]:
        text, fenced = m[1].strip(), True
    try:
        json.loads(text)
    except ValueError:                                   # P8
        try:
            value, _end = _DEC.raw_decode(text, 0)
        except ValueError:
            return PH1.ph1_parse_raw(raw, attribute)     # still parse_fail (P5)
        res = PH1.ph1_parse_raw(json.dumps(value), attribute)
        S.PARSE_INFO["fence_stripped"] = fenced
        return res
    return PH1.ph1_parse_raw(raw, attribute)


def install():
    S.parse_raw, S.map_cat = ph2_parse_raw, PH1.ph1_map_cat


def uninstall():
    PH1.uninstall()
