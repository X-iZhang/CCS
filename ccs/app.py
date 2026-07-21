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

"""Gradio demo for Clinical Consensus Selection.

Paste a pool of candidate reports, pick one or more utilities, and see which
candidate each utility selects and why.

Run with::

    python -m ccs.app                 # local only
    python -m ccs.app --share         # public gradio.live link
"""

import json
from typing import List

import gradio as gr

from .ccs_utils import ALL_UTILITIES, PAPER_UTILITIES, QWEN_UTILITY
from .run_ccs import CCS

# Utilities that run on CPU -- safe defaults for a demo with no GPU.
CPU_UTILITIES = ["bleu_4", "rougeL", "temporal_f1"]
DEFAULT_UTILITIES = [QWEN_UTILITY]

# A rollout pool sampled from a radiology MLLM, lightly cleaned for spelling, used
# to pre-fill the demo. Model output, not a clinical record -- the candidates
# disagree with each other, which is exactly what CCS is for.
EXAMPLE_CANDIDATES = [
    "The chest X-ray image shows multiple cavitary nodules in the right upper lobe of the lung. These nodules are surrounded by an ill-defined ground glass opacity, which is indicative of an abnormal growth or lesion in the lung tissue.",
    "The chest X-ray image shows bilateral hilar lymphadenopathy, which means there is enlargement of the lymph nodes in the region of the hilum on both sides of the lungs. Additionally, there is a right-sided pleural effusion, which is the accumulation of fluid in the pleural space surrounding the right lung.",
    "The chest radiograph shows a large, well-defined mass in the right hemithorax, which is the right side of the chest cavity. This mass has a homogeneous density and is displacing the mediastinum, which contains the heart, great vessels, trachea, and esophagus.",
    "The chest X-ray image shows multiple nodules in the right lower lung field. These nodules are small, round masses that can be seen on the X-ray.",
    "The chest X-ray image shows several findings, including bilateral basilar infiltrates, right-sided pleural effusion, and a right-upper-lobe lesion. Bilateral basilar infiltrates refer to areas of increased opacity in the lower regions of both lungs.",
]

# One selector per (utilities, details); backends are cached inside ccs_utils.
_selector_cache = {}


def _get_selector(utilities: List[str], details: bool) -> CCS:
    key = (tuple(utilities), details)
    if key not in _selector_cache:
        _selector_cache[key] = CCS(utilities=list(utilities), details=details)
    return _selector_cache[key]


class CandidateFormatError(ValueError):
    """Raised when the pasted text is structured data rather than reports."""


def parse_candidates(raw: str) -> List[str]:
    """Parse the candidate textbox: a JSON list of strings, or one report per line."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise CandidateFormatError(
                "That looks like JSON but does not parse. Paste a JSON list of report "
                "strings, or one report per line."
            )
        if not (isinstance(parsed, list) and all(isinstance(x, str) for x in parsed)):
            raise CandidateFormatError(
                "That JSON is not a list of report strings. Paste the report text "
                "itself — one per line, or as a JSON list of strings."
            )
        return [x for x in parsed if x.strip()]

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # Same guard as the CLI: a rollout JSONL would otherwise become N raw JSON blobs
    # that CCS "selects" among, output indistinguishable from a real selection.
    if lines and lines[0].lstrip().startswith("{"):
        raise CandidateFormatError(
            "That looks like a rollout JSONL (lines are JSON objects), not report text. "
            "Extract the reports first — one per line."
        )
    return lines


def _truncate(text: str, n: int = 90) -> str:
    return text if len(text) <= n else text[:n].rstrip() + "..."


def run_ccs(raw_candidates: str,
            utilities: List[str],
            show_details: bool):
    """Run the configured utilities over the pasted pool.

    Returns (selected_report_md, comparison_table, consensus_table, matrix_md).
    """
    try:
        candidates = parse_candidates(raw_candidates)
    except CandidateFormatError as e:
        return f"⚠️ {e}", [], [], ""

    if len(candidates) == 0:
        return "⚠️ Paste at least one candidate report (one per line).", [], [], ""
    if len(candidates) == 1:
        return ("⚠️ Only one candidate found — CCS selects *among* candidates, so a pool "
                "of one is a trivial pick. Paste several reports, one per line."), [], [], ""
    if not utilities:
        return "⚠️ Select at least one utility.", [], [], ""

    try:
        selector = _get_selector(utilities, show_details)
        results = selector(candidates=candidates)
    except Exception as e:  # surfaced in the UI rather than only in the console
        return f"❌ Selection failed: `{type(e).__name__}: {e}`", [], [], ""

    # -- headline: what the first configured utility picked --
    head = results[0]
    md = [
        f"### 🩺 Selected by `{head['utility']}` — **cand_{head['selected_idx']}**",
        "",
        f"> {head['text']}",
    ]
    picks = {r["selected_idx"] for r in results}
    if len(results) > 1:
        if len(picks) == 1:
            md.append(f"\n*All {len(results)} utilities agree on cand_{head['selected_idx']}.*")
        else:
            md.append(f"\n*The {len(results)} utilities disagree — they picked "
                      f"{', '.join(f'cand_{i}' for i in sorted(picks))}. "
                      f"See the comparison below.*")

    # -- which candidate each utility chose --
    comparison = [
        [r["utility"],
         f"cand_{r['selected_idx']}",
         round(max(r["utility_score"]), 4),
         _truncate(r["text"])]
        for r in results
    ]

    # -- consensus score of every candidate, per utility --
    consensus = []
    for i, cand in enumerate(candidates):
        row = [f"cand_{i}"]
        for r in results:
            mark = " ★" if r["selected_idx"] == i else ""
            row.append(f"{r['utility_score'][i]:.4f}{mark}")
        row.append(_truncate(cand, 60))
        consensus.append(row)

    # -- optional (N, N) matrix for the first utility --
    matrix_md = ""
    if show_details and "utility_matrix" in head:
        M = head["utility_matrix"]
        header = "| |" + "|".join(f" cand_{j} " for j in range(len(M))) + "|"
        sep = "|---|" + "|".join("---" for _ in M) + "|"
        rows = [
            f"| **cand_{i}** |" + "|".join(f" {v:.3f} " for v in row) + "|"
            for i, row in enumerate(M)
        ]
        matrix_md = "\n".join([
            f"#### `{head['utility']}` pairwise similarity  (row = reference, column = hypothesis)",
            "", header, sep, *rows, "",
            "*Column-wise mean of this matrix is exactly the consensus score above. "
            "It is asymmetric for lexical utilities — BLEU/ROUGE score relative to the reference.*",
        ])

    return "\n".join(md), comparison, consensus, matrix_md


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="🩺 Clinical Consensus Selection", theme="soft") as demo:
        gr.Markdown("""
        # 🩺 CCS: Clinical Consensus Selection

        A radiology MLLM usually commits to **one** report token by token — a single
        unlucky step can drop a finding or assert one the image does not support.
        CCS instead samples several reports and keeps the one with the highest
        **clinical consensus** across the pool.

        Every utility follows the same rule: each candidate is scored by its **average
        pairwise similarity** to the rest of the pool, and the highest-consensus
        candidate wins. Utilities differ only in how that similarity is measured.
        CCS reads the **reports only** — no image at inference time.
        """)

        with gr.Tab("✨ CCS Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Candidate Pool")
                    candidates_box = gr.Textbox(
                        label="Candidate reports (one per line, or a JSON list)",
                        lines=12,
                        value="\n".join(EXAMPLE_CANDIDATES),
                        placeholder=(
                            "Paste the candidate reports your model sampled for one "
                            "study — one report per line."
                        ),
                    )
                    gr.Markdown(
                        "*Pre-filled with a real rollout from a radiology MLLM. "
                        "Replace it with your own pool — one report per line.*"
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### Utilities")
                    utilities_in = gr.CheckboxGroup(
                        choices=ALL_UTILITIES,
                        value=DEFAULT_UTILITIES,
                        label="Which utilities to run",
                        info="Each selects independently — results are never merged.",
                    )
                    with gr.Row():
                        paper_btn = gr.Button("📄 Paper set (9)", size="sm")
                        cpu_btn = gr.Button("💻 CPU-only (fast)", size="sm")
                        clear_btn = gr.Button("✖ Clear", size="sm")

                    gr.Markdown(
                        "> ⏳ **First call downloads and loads models.** `qwen_vl_emb` pulls a "
                        "~4 GB checkpoint and needs a CUDA GPU; `green` is a 7B LLM and is very "
                        "slow. `bleu_4` / `rougeL` run on CPU and are the quickest way to try "
                        "this out. Models stay loaded afterwards."
                    )

                    details_in = gr.Checkbox(
                        label="Show pairwise similarity matrix (details=True)",
                        value=False,
                    )
                    run_btn = gr.Button("🚀 Select", variant="primary")

            output_md = gr.Markdown(value="### 👆 Paste candidates and hit **Select**.")

            gr.Markdown("### Which candidate did each utility choose?")
            comparison_out = gr.Dataframe(
                headers=["utility", "selected", "top consensus", "selected report"],
                datatype=["str", "str", "number", "str"],
                wrap=True,
            )

            gr.Markdown("### Consensus score of every candidate  (★ = that utility's pick)")
            consensus_out = gr.Dataframe(wrap=True)

            matrix_out = gr.Markdown(value="")

            # -- wiring --
            run_btn.click(
                fn=run_ccs,
                inputs=[candidates_box, utilities_in, details_in],
                outputs=[output_md, comparison_out, consensus_out, matrix_out],
            )
            paper_btn.click(lambda: gr.update(value=list(PAPER_UTILITIES)), outputs=utilities_in)
            cpu_btn.click(lambda: gr.update(value=list(CPU_UTILITIES)), outputs=utilities_in)
            clear_btn.click(lambda: gr.update(value=[]), outputs=utilities_in)

        gr.Markdown("""
        ### Terms of Use

        This demo is a **research preview** for non-commercial use only.

        - **Not medical advice** — all output is experimental and must not replace
          professional medical judgment. Every report must be reviewed by a qualified
          radiologist before informing any clinical decision.
        - **Pool-bounded** — CCS can only return a report the base model actually
          sampled. It cannot recover a finding that appears in no candidate.
        - **Patient privacy** — only paste de-identified report text, compliant with
          HIPAA, GDPR, or your local equivalent.
        """)

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch the CCS Gradio demo.")
    parser.add_argument("--share", action="store_true",
                        help="Create a public gradio.live link (off by default — the "
                             "demo accepts clinical report text).")
    parser.add_argument("--server-name", default="127.0.0.1",
                        help="Bind address (use 0.0.0.0 to expose on your network).")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(share=args.share,
                server_name=args.server_name,
                server_port=args.server_port)


if __name__ == "__main__":
    main()
