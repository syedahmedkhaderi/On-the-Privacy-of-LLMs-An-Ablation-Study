"""dea_a3: single-turn data-extraction study on public pretrained models.

Stages:
  00  build the ground-truth index (V-B intersect V-P) and the true-prefix pool
  01  generate and detect, one cell per (model, condition, seed)
  02  aggregate across seeds and produce tables and the duplication figure
"""

__version__ = "1.0.0"

from .config import CONFIG, Config, DecoderSpec, ModelSpec  # noqa: F401
from .paths import PATHS, Paths, find_root  # noqa: F401
