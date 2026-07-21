import warnings

# Silence noisy third-party DeprecationWarnings (e.g. pysbd's unescaped regex
# strings pulled in via RadEval) so they don't drown out real output. These come
# from dependencies, not CCS, and are harmless. Scoped by module so genuine CCS
# deprecations still surface.
for _mod in (r"pysbd.*", r".*allennlp.*", r"transformers.*"):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=_mod)
    warnings.filterwarnings("ignore", category=SyntaxWarning, module=_mod)

# Let an ImportError propagate: swallowing it would leave `import ccs` looking
# successful while `ccs.CCS` does not exist, surfacing later as a confusing
# AttributeError far from the missing dependency.
from .run_ccs import CCS
from .ccs_utils import ALL_UTILITIES, PAPER_UTILITIES

__all__ = ["CCS", "ALL_UTILITIES", "PAPER_UTILITIES"]
