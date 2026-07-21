#    Copyright (c) 2026 Xi Zhang
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

"""Core Clinical Consensus Selection (CCS) utilities.

This module holds the *pure* selection kernel and the utility configuration
(no heavy imports at module load time), plus lazily-constructed scoring backends:

* ``mbr_from_matrix``  -- the consensus (MBR-centroid) kernel shared by every utility.
* ``ccs_select``       -- RadEval-metric consensus selection over a candidate pool.
* ``get_evaluator``    -- lazily build / cache a RadEval evaluator per utility.
* ``qwen_embed`` / ``qwen_mbr_select`` -- the image-informed RRG-Qwen-VL-Embedding utility.

Heavy dependencies (``RadEval``, ``torch``, ``transformers``) are imported only
inside the functions that need them, so ``import ccs`` works with just numpy.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

# Default placeholder for blank candidates (avoids tokenizer crashes / degenerate scores).
EMPTY_PLACEHOLDER = "no findings."

# Qwen3-VL-Embedding-2B, LoRA fine-tuned on MIMIC-CXR image-report pairs and merged.
QWEN_MODEL_ID = "Yingshu/mimic_zx_lora8_32_0.01_mrl_fn_lr1e4"

# The image-informed utility: trained on image-report pairs, but reads reports only.
QWEN_UTILITY = "qwen_vl_emb"


# ============================================================
# Metric extractors -- map a RadEval result dict to a list of per-hypothesis
# scores (one float per candidate). Matches the API of RadEval 0.0.6rc2 (the
# version used in the CCS paper / run_mbr_select.py): build with do_*=True,
# do_details=True, then read per-sample lists from result["<metric>"]["sample_scores"].
# ============================================================
METRICS = {
    "radgraph_partial":  lambda r: r["radgraph"]["sample_scores"][1],
    "bleu_4":            lambda r: r["bleu"]["bleu_4"]["sample_scores"],
    "bertscore":         lambda r: r["bertscore"]["sample_scores"],
    "rougeL":            lambda r: r["rouge"]["rougeL"]["sample_scores"],
    "chexbert_all":      lambda r: list(r["chexbert"]["sample_scores"]["all_labels"]),
    "chexbert_5":        lambda r: list(r["chexbert"]["sample_scores"]["5_labels"]),
    "ratescore":         lambda r: r["ratescore"]["sample_scores"],
    "temporal_f1":       lambda r: r["temporal_f1"]["sample_scores"],
    "radeval_bertscore": lambda r: r["radeval_bertscore"]["sample_scores"],
    "srr_bert":          lambda r: r["srr_bert"]["srr_bert_weighted_f1"]["sample_scores"],
    "green":             lambda r: r["green"]["sample_scores"],
}

# Which RadEval component each utility needs (kwargs for the RadEval constructor).
METRIC_TO_RADEVAL = {
    "radgraph_partial":  dict(do_radgraph=True),
    "bleu_4":            dict(do_bleu=True),
    "bertscore":         dict(do_bertscore=True),
    "rougeL":            dict(do_rouge=True),
    "chexbert_all":      dict(do_chexbert=True),
    "chexbert_5":        dict(do_chexbert=True),
    "ratescore":         dict(do_ratescore=True),
    "temporal_f1":       dict(do_temporal=True),
    "radeval_bertscore": dict(do_radeval_bertscore=True),
    "srr_bert":          dict(do_srr_bert=True),
    "green":             dict(do_green=True),
}

# All RadEval-backed text utilities, and the full utility set.
RADEVAL_UTILITIES: List[str] = list(METRICS.keys())              # text utilities
ALL_UTILITIES: List[str] = RADEVAL_UTILITIES + [QWEN_UTILITY]    # + image-informed utility

# The utilities reported in the paper; pass as CCS(utilities=PAPER_UTILITIES).
PAPER_UTILITIES: List[str] = [
    "bleu_4", "rougeL", "bertscore", "radeval_bertscore",
    "radgraph_partial", "chexbert_5", "chexbert_all", "ratescore",
    QWEN_UTILITY,
]

# Unambiguous shortenings only -- no "chexbert" alias (could mean _5 or _all).
UTILITY_ALIASES: Dict[str, str] = {
    "bleu": "bleu_4",
    "rouge": "rougeL",
    "rougel": "rougeL",
    "radgraph": "radgraph_partial",
    "qwen": QWEN_UTILITY,
}


def canonical_utility(utility: str) -> str:
    """Map an alias to its canonical utility name; raise if unknown."""
    if utility in ALL_UTILITIES:
        return utility
    resolved = UTILITY_ALIASES.get(utility.lower() if isinstance(utility, str) else utility)
    if resolved is not None:
        return resolved
    raise ValueError(
        f"Unknown utility {utility!r}. Valid utilities: {ALL_UTILITIES}. "
        f"Aliases: {sorted(UTILITY_ALIASES)}."
    )


def is_valid_utility(utility: str) -> bool:
    """True if ``utility`` is a known utility name or alias."""
    try:
        canonical_utility(utility)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def sanitize_candidates(candidates: List[str],
                        placeholder: str = EMPTY_PLACEHOLDER) -> List[str]:
    """Replace blank/whitespace candidates with ``placeholder``."""
    return [c if c and c.strip() else placeholder for c in candidates]


# ============================================================
# Consensus kernel (shared by every utility)
# ============================================================
def mbr_from_matrix(sim_matrix: np.ndarray) -> Tuple[int, List[float]]:
    """The CCS consensus centroid: pick the candidate with the highest mean
    pairwise similarity to the rest of the pool.

    Args:
        sim_matrix: ``(N, N)`` array; entry ``[i, j]`` is the similarity of
            candidate ``j`` scored against candidate ``i`` as the reference.

    Returns:
        ``(best_idx, consensus_scores)`` -- the argmax index and the per-candidate
        consensus scores (mean over reference rows).
    """
    consensus = np.mean(sim_matrix, axis=0).tolist()
    best_idx = int(np.argmax(consensus))
    return best_idx, consensus


# ============================================================
# RadEval text-utility selection
# ============================================================
def ccs_select(candidates: List[str],
               metric_name: str,
               evaluator) -> Tuple[int, List[float], np.ndarray]:
    """Consensus selection driven by a RadEval text-similarity metric.

    For each candidate ``i`` taken as the single reference, all ``N`` candidates
    are scored as hypotheses; the resulting ``(N, N)`` matrix is reduced to a
    consensus score per candidate via :func:`mbr_from_matrix`.

    Args:
        candidates: list of ``N`` candidate report strings.
        metric_name: a key of :data:`METRICS`.
        evaluator: a RadEval evaluator with the matching component enabled.

    Returns:
        ``(best_idx, consensus_scores, sim_matrix)`` -- the matrix is returned so
        callers can surface it under ``details=True``.
    """
    N = len(candidates)
    if N == 1:
        return 0, [1.0], np.ones((1, 1))

    candidates = sanitize_candidates(candidates)
    extractor = METRICS[metric_name]
    sim_matrix = np.zeros((N, N))

    failures = []
    for i in range(N):
        refs = [candidates[i]] * N
        hyps = candidates
        try:
            results = evaluator(refs=refs, hyps=hyps)
            sim_matrix[i, :] = extractor(results)
        except Exception as e:  # keep one bad reference row from sinking the pool
            print(f"  [warn] {metric_name} failed for ref={i}: {e}")
            failures.append(e)

    # All rows failed -> all-zero matrix -> argmax silently returns cand_0 as if it
    # had won. Fail loudly instead. (Typical cause: scikit-learn != 1.8.0.)
    if failures and len(failures) == N:
        raise RuntimeError(
            f"Utility {metric_name!r} failed on all {N} reference rows; "
            f"no consensus could be computed. First error: {failures[0]!r}"
        )

    best_idx, consensus = mbr_from_matrix(sim_matrix)
    return best_idx, consensus, sim_matrix


# Cache of RadEval evaluators, keyed by utility name (heavy to build).
_evaluator_cache: Dict[str, object] = {}


@lru_cache(maxsize=None)
def get_evaluator(utility: str):
    """Lazily build (and memoise) a RadEval evaluator with only ``utility``'s
    component enabled. Requires the ``RadEval`` dependency (pinned to 0.0.6rc2)."""
    from RadEval import RadEval

    if utility not in METRIC_TO_RADEVAL:
        raise ValueError(
            f"Unknown RadEval utility {utility!r}. "
            f"Valid: {sorted(METRIC_TO_RADEVAL)}"
        )
    return RadEval(**METRIC_TO_RADEVAL[utility], do_details=True)


# ============================================================
# Image-informed multimodal utility (RRG-Qwen-VL-Embedding)
# ============================================================
# The embedder classes inherit from transformers' Qwen3VL base classes, so they
# can only be defined once transformers is importable. We build them lazily and
# cache both the class and a per-model embedder instance.
_QWEN_EMBEDDER_CLS = None
_qwen_embedder_cache: Dict[str, object] = {}


def _build_qwen_embedder_cls():
    """Define and return the ``Qwen3VLEmbedder`` class (lazy, needs torch/transformers)."""
    global _QWEN_EMBEDDER_CLS
    if _QWEN_EMBEDDER_CLS is not None:
        return _QWEN_EMBEDDER_CLS

    import unicodedata
    from dataclasses import dataclass
    from typing import Optional as _Optional

    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.modeling_outputs import ModelOutput
    from qwen_vl_utils.vision_process import process_vision_info

    @dataclass
    class Qwen3VLEmbeddingOutput(ModelOutput):
        last_hidden_state: _Optional[torch.FloatTensor] = None
        attention_mask: _Optional[torch.Tensor] = None

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        config: Qwen3VLConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                    inputs_embeds=None, pixel_values=None, image_grid_thw=None, **kw):
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, inputs_embeds=inputs_embeds,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kw,
            )
            return Qwen3VLEmbeddingOutput(
                last_hidden_state=out.last_hidden_state,
                attention_mask=attention_mask,
            )

    class Qwen3VLEmbedder:
        IMAGE_FACTOR = 32
        MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
        MAX_PIXELS = 1024 * 28 * 28

        def __init__(self, model_name_or_path, max_length=8192,
                     default_instruction="Represent the user's input.", **kw):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.max_length = max_length
            self.default_instruction = default_instruction
            self.model = Qwen3VLForEmbedding.from_pretrained(
                model_name_or_path, trust_remote_code=True, **kw,
            ).to(device).eval()
            self.processor = Qwen3VLProcessor.from_pretrained(
                model_name_or_path, padding_side="right",
            )

        def _to_conversation(self, item):
            instr = item.get("instruction", self.default_instruction).strip()
            if instr and not unicodedata.category(instr[-1]).startswith("P"):
                instr += "."
            content = []
            if "image" in item:
                img = item["image"]
                if isinstance(img, str) and not img.startswith(("http://", "https://")):
                    img = "file://" + img
                content.append({"type": "image", "image": img,
                                "min_pixels": self.MIN_PIXELS, "max_pixels": self.MAX_PIXELS})
            if "text" in item and item["text"]:
                content.append({"type": "text", "text": item["text"]})
            if not content:
                content.append({"type": "text", "text": "NULL"})
            return [
                {"role": "system", "content": [{"type": "text", "text": instr}]},
                {"role": "user", "content": content},
            ]

        def _preprocess(self, convs):
            texts = self.processor.apply_chat_template(
                convs, add_generation_prompt=True, tokenize=False,
            )
            try:
                images, _, _ = process_vision_info(
                    convs, image_patch_size=16,
                    return_video_metadata=True, return_video_kwargs=True,
                )
            except Exception:
                images = None
            return self.processor(
                text=texts, images=images,
                truncation=True, max_length=self.max_length,
                padding=True, do_resize=False, return_tensors="pt",
            )

        @staticmethod
        def _pool_last(hidden, mask):
            last_pos = mask.flip(dims=[1]).argmax(dim=1)
            col = mask.shape[1] - last_pos - 1
            row = torch.arange(hidden.shape[0], device=hidden.device)
            return hidden[row, col]

        @torch.no_grad()
        def process(self, inputs, normalize=True):
            convs = [self._to_conversation(it) for it in inputs]
            enc = self._preprocess(convs)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            out = self.model(**enc)
            emb = self._pool_last(out.last_hidden_state, out.attention_mask)
            return F.normalize(emb, p=2, dim=-1) if normalize else emb

    _QWEN_EMBEDDER_CLS = Qwen3VLEmbedder
    return _QWEN_EMBEDDER_CLS


def get_qwen_embedder(model_id: str = QWEN_MODEL_ID):
    """Lazily build (and cache) a :class:`Qwen3VLEmbedder` for ``model_id``."""
    if model_id not in _qwen_embedder_cache:
        cls = _build_qwen_embedder_cls()
        print(f"Loading RRG-Qwen-VL-Embedding model: {model_id}")
        _qwen_embedder_cache[model_id] = cls(model_id)
    return _qwen_embedder_cache[model_id]


def qwen_embed(candidates: List[str],
               instruction: Optional[str] = None,
               embedder=None,
               model_id: str = QWEN_MODEL_ID) -> np.ndarray:
    """Embed candidate reports into L2-normalized vectors.

    Reports only -- no image is read at inference time. The encoder is *image-
    informed* (fine-tuned on image-report pairs), which is what lets its text
    representation carry clinical information a text-only encoder would miss.

    Returns an ``(N, D)`` numpy array.
    """
    embedder = embedder or get_qwen_embedder(model_id)
    candidates = sanitize_candidates(candidates)

    inputs = []
    for text in candidates:
        item = {"text": text}
        if instruction is not None:
            item["instruction"] = instruction
        inputs.append(item)

    emb = embedder.process(inputs, normalize=True)
    # .float() upcasts bf16/fp16 -> fp32, which numpy supports (it has no bf16 dtype).
    return emb.detach().float().cpu().numpy()


def qwen_mbr_select(embeddings: np.ndarray) -> Tuple[int, List[float], np.ndarray]:
    """Consensus selection over L2-normalized embeddings (cosine similarity).

    Returns ``(best_idx, consensus_scores, sim_matrix)``, matching :func:`ccs_select`.
    """
    if len(embeddings) == 1:
        return 0, [1.0], np.ones((1, 1))
    sim_matrix = embeddings @ embeddings.T
    best_idx, consensus = mbr_from_matrix(sim_matrix)
    return best_idx, consensus, sim_matrix
