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

"""Clinical Consensus Selection (CCS) -- public entry point and CLI.

Usage (script):

    from ccs import CCS

    selector = CCS(utilities=["radgraph_partial", "bleu_4"])
    results  = selector(candidates=reports)
    print(json.dumps(results, indent=4))

Usage (CLI):

    python -m ccs.run_ccs --input-file rollout.jsonl --utilities qwen_vl_emb \
        -N 8 --output-file selected.jsonl
"""

import json
import os
from typing import List, Optional, Sequence, Union

from .ccs_utils import (
    ALL_UTILITIES,
    PAPER_UTILITIES,
    QWEN_MODEL_ID,
    QWEN_UTILITY,
    RADEVAL_UTILITIES,
    canonical_utility,
    ccs_select,
    get_evaluator,
    get_qwen_embedder,
    qwen_embed,
    qwen_mbr_select,
    sanitize_candidates,
)

DEFAULT_UTILITY = QWEN_UTILITY


class CCS:
    """Clinical Consensus Selection over a pool of candidate reports.

    Each candidate is scored by its average pairwise similarity to the rest of the
    pool; the highest-consensus one wins. Utilities differ only in how that
    similarity is computed. Backends load on first use and are reused.

    Args:
        utilities: a utility name or list of names; aliases accepted. Pass
            :data:`PAPER_UTILITIES` for the set reported in the paper.
        details: also return ``utility_matrix``, the ``(N, N)`` similarity matrix.
        instruction: prompt for the embedding utility. **Leave this alone.** The
            encoder was trained with an asymmetric query/doc paradigm and the
            doc-side default is part of that setup; changing it shifts the
            embeddings enough to select a different candidate.
        model_id: HuggingFace id of the embedding utility's encoder.

    Example::

        results = CCS(utilities=["radgraph", "bleu"])(candidates=reports)
    """

    def __init__(self,
                 utilities: Union[str, Sequence[str]] = DEFAULT_UTILITY,
                 details: bool = False,
                 instruction: Optional[str] = None,
                 model_id: str = QWEN_MODEL_ID):
        if isinstance(utilities, str):
            utilities = [utilities]
        if not utilities:
            raise ValueError("`utilities` must name at least one utility.")
        # Canonicalize up front so a bad name fails before any model loads.
        self.utilities: List[str] = [canonical_utility(u) for u in utilities]
        self.details = details
        self.instruction = instruction
        self.model_id = model_id

    def __repr__(self) -> str:
        # Show only what is in play; defaults are noise.
        parts = [f"utilities={self.utilities}"]
        if self.details:
            parts.append("details=True")
        if self.instruction is not None:
            parts.append(f"instruction={self.instruction!r}")
        if QWEN_UTILITY in self.utilities and self.model_id != QWEN_MODEL_ID:
            parts.append(f"model_id={self.model_id!r}")
        return f"CCS({', '.join(parts)})"

    # -- backends: built on first use, then reused for the instance's lifetime --

    def _evaluator(self, utility: str):
        return get_evaluator(utility)

    def _embedder(self):
        return get_qwen_embedder(self.model_id)

    def _run_one(self, utility: str, candidates: List[str]):
        """Run a single utility over a sanitized pool -> (idx, scores, matrix)."""
        if utility == QWEN_UTILITY:
            embeddings = qwen_embed(candidates,
                                    instruction=self.instruction,
                                    embedder=self._embedder())
            return qwen_mbr_select(embeddings)
        return ccs_select(candidates, utility, self._evaluator(utility))

    def __call__(self, candidates: List[str]) -> List[dict]:
        """Select the highest-consensus candidate under each configured utility.

        Reads the reports only -- no image at inference time.

        Returns one dict per utility, in configured order::

            [{"utility": <name>, "text": <selected report>,
              "utility_score": [<float>, ...], "selected_idx": <int>,
              "utility_matrix": [[...]]  # only if details=True
             }, ...]

        Always a list, even for a single utility.
        """
        if not isinstance(candidates, (list, tuple)) or len(candidates) == 0:
            raise ValueError("`candidates` must be a non-empty list of report strings.")
        candidates = list(candidates)
        if not all(isinstance(c, str) for c in candidates):
            raise TypeError("Every element of `candidates` must be a str.")

        sanitized = sanitize_candidates(candidates)

        results = []
        for utility in self.utilities:
            idx, scores, matrix = self._run_one(utility, sanitized)
            entry = {
                "utility": utility,
                "text": candidates[idx],
                "utility_score": [float(s) for s in scores],
                "selected_idx": int(idx),
            }
            if self.details:
                entry["utility_matrix"] = [[float(x) for x in row] for row in matrix]
            results.append(entry)
        return results

    # -- batch helper -------------------------------------------------------

    def run_jsonl(self,
                  input_file: str,
                  output_file: Optional[str] = None,
                  num_candidates: Optional[int] = None) -> List[dict]:
        """Run CCS over a rollout JSONL, one selection per sample.

        Each input line has ``"text"`` = ``[{"text": <report>}, ...]``. Output keeps
        that schema (selected report wrapped in a 1-element list) so downstream
        evaluation treats it like a single-path output. The first utility drives the
        written report; all utilities are kept under ``metadata["per_utility"]``.

        Args:
            input_file: path to the rollout JSONL.
            output_file: optional path to write the selected JSONL.
            num_candidates: if given, truncate each pool to the first ``N``.
        """
        print(f"Loading: {input_file}")
        samples = []
        with open(input_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"Loaded {len(samples)} samples")
        if not samples:
            return []

        first_n = len(samples[0]["text"])
        effective_n = min(first_n, num_candidates) if num_candidates else first_n
        print(f"Candidates per sample: {first_n} (using first {effective_n}) | "
              f"utilities: {self.utilities}")

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kw):
                return x

        records = []
        for sample in tqdm(samples, desc="CCS select"):
            candidates = [c["text"] for c in sample["text"]]
            if num_candidates:
                candidates = candidates[:num_candidates]

            results = self(candidates)
            headline = results[0]

            records.append({
                "question_id": sample.get("question_id"),
                "prompt": sample.get("prompt", ""),
                "text": [{"text": headline["text"]}],
                "answer_id": sample.get("answer_id", ""),
                "model_id": sample.get("model_id", ""),
                "metadata": {
                    "approach": "ccs",
                    "utility": headline["utility"],
                    "num_candidates": len(candidates),
                    "selected_idx": headline["selected_idx"],
                    "utility_score": headline["utility_score"],
                    "per_utility": results,
                },
            })

        if output_file:
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"Wrote {len(records)} records -> {output_file}")

        return records


def _stream_print(report: str, header: str = "[🩺 CCS Selected Report]") -> None:
    """Print the selected report word-by-word for a light streaming effect."""
    import time
    print(header)
    for word in report.split(" "):
        print(word, end=" ", flush=True)
        time.sleep(0.02)
    print()


def _parse_reports_arg(reports_arg: str) -> List[str]:
    """Parse ``--reports``: a JSON list, a path to a .json/.txt file of reports,
    or a single string."""
    if os.path.isfile(reports_arg):
        with open(reports_arg, "r") as f:
            content = f.read()
        stripped = content.strip()
        if stripped.startswith("["):
            return json.loads(stripped)
        lines = [ln for ln in content.splitlines() if ln.strip()]
        # A rollout JSONL here would parse as one giant "report" per line -- N=1,
        # cand_0 "selected", looking like success. Reject it explicitly.
        if lines and lines[0].lstrip().startswith("{"):
            raise SystemExit(
                f"{reports_arg!r} looks like a rollout JSONL (lines are JSON objects), "
                f"not a list of report strings.\n"
                f"Use --input-file for rollout JSONL (batch mode), or pass --reports "
                f"a JSON list / a plain-text file with one report per line."
            )
        return lines
    stripped = reports_arg.strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    return [reports_arg]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CCS: select the most clinically-consistent report from a candidate pool."
    )
    parser.add_argument("--input-file", default=None,
                        help="Rollout JSONL (batch mode). Mutually exclusive with --reports.")
    parser.add_argument("--output-file", default=None,
                        help="Output JSONL path (batch mode).")
    parser.add_argument("--reports", default=None,
                        help="Inline candidates: a JSON list, a file path, or a single string.")
    parser.add_argument("--utilities", nargs="+", default=[DEFAULT_UTILITY],
                        help=(f"One or more utilities. Canonical names: {ALL_UTILITIES}. "
                              f"Use 'paper' to expand to the {len(PAPER_UTILITIES)} "
                              f"utilities reported in the paper."))
    parser.add_argument("-N", "--num-candidates", type=int, default=None,
                        help="Truncate each pool to the first N candidates.")
    parser.add_argument("--details", action="store_true",
                        help="Also emit the full (N, N) pairwise similarity matrix.")
    args = parser.parse_args()

    utilities = list(args.utilities)
    if utilities == ["paper"]:
        utilities = list(PAPER_UTILITIES)

    selector = CCS(utilities=utilities, details=args.details)

    if args.input_file:
        selector.run_jsonl(args.input_file, output_file=args.output_file,
                           num_candidates=args.num_candidates)
        return

    if not args.reports:
        parser.error("Provide either --input-file (batch) or --reports (single pool).")

    reports = _parse_reports_arg(args.reports)
    if args.num_candidates:
        reports = reports[:args.num_candidates]

    results = selector(candidates=reports)

    if len(results) > 1:
        print(f"\n{'utility':22s} {'selected':>10s} {'consensus(top)':>16s}")
        for r in results:
            top = max(r["utility_score"]) if r["utility_score"] else float("nan")
            print(f"{r['utility']:22s} {'cand_' + str(r['selected_idx']):>10s} {top:>16.3f}")
        print()
        _stream_print(results[0]["text"],
                      header=f"[🩺 CCS Selected Report ({results[0]['utility']})]")
    else:
        r = results[0]
        print(f"utility={r['utility']}  selected=cand_{r['selected_idx']}  "
              f"consensus={[round(s, 3) for s in r['utility_score']]}")
        _stream_print(r["text"])


if __name__ == "__main__":
    main()
