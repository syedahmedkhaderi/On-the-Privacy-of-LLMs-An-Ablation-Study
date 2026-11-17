# Repository maintenance

- Keep main as exactly one parentless commit. Amend it; do not add commits, branches or tags.
- Save a recovery bundle outside the repository before replacing history.
- Do not push unless the owner explicitly requests it. Use an explicit force-with-lease against the inspected remote hash.
- Never commit protected data, real PII, raw outputs, rater sheets, private keys, model weights or adapters.
- Keep result values unchanged during audits. Record mismatches and leave verification incomplete until resolved.
- Preserve retained source file names and bodies recorded in code/SOURCES.csv.
