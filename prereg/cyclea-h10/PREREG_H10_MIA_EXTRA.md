# PREREG_H10_MIA_EXTRA

Written (UTC): 2026-11-10T17:42:56Z
Code hash (sidecar tools/mia_extra.py sha256): cd1be3e94ca53bdfb5f84e379fedc19163963ae04f6c9336683ec83098192fe3
Bundle scoring code read (not modified): mia-v3-splitlabel attacks/MIA/scoring.py

Same splits, documents and windows as the 359+359 PRIMARY and SECONDARY runs, models pythia-1.4b @9cc5c8c8 and pythia-6.9b @21bfa02e. Additional scores: Min-K 10 and Min-K 30; zlib ratio (window total negative log-likelihood / zlib bytes of the window text); reference ratio (mean loss / mean loss of pythia-70m @a39f36b1 on the same token windows). Same bootstrap as the primary analysis (10,000, seed 0). Implemented as a sidecar script that reuses the bundle's scoring function and must first reproduce the stored mean-loss and Min-K-20 AUCs to 4 decimals.
Labelled "added after the primary results, secondary".
