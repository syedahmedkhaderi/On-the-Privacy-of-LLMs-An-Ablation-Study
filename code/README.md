# Experiment code

Implementations are grouped by attack. Analysis scripts are in analysis/; fixed launch arguments and seeds are in queues/. See the root README.md for the paper-to-artifact map and REPRODUCE.md for settings.

SOURCES.csv records the source and released hashes of retained scientific files. File names and bodies remain unchanged. Some launchers and analyses contain operational guards or remote collection alongside scientific work; they need local input paths and should be read before execution. The standalone CPU checks use only released aggregates.

ba/ba_run.py retains an unused Qwen2.5-1.5B fallback branch. It is not part of the paper's experiments. MBMIA's compatibility parser supports the included reporting modules; it does not contain earlier submission results.
