from __future__ import annotations

from pathlib import Path

from src.training.train_weighted_bce import VANILLA_BCE_TEST


def _row(name: str, m: dict) -> str:
    acc = m.get("accuracy")
    acc_s = f"{acc * 100:.2f}" if acc is not None else "NA"
    auroc = m.get("macro_auroc")
    auroc_s = f"{auroc * 100:.2f}" if auroc is not None else "NA"
    aupr = m.get("macro_auprc")
    aupr_s = f"{aupr * 100:.2f}" if aupr is not None else "NA"
    f1 = m.get("micro_f1")
    f1_s = f"{f1 * 100:.2f}" if f1 is not None else "NA"
    return f"| {name} | {acc_s} | {auroc_s} | {aupr_s} | {f1_s} |"


def write_paper_table1(path: Path, full: dict, no_concept: dict, no_note: dict) -> None:
    """Professor-facing Table 1 analogue on Coherent. Metrics in percent like the paper."""
    ft = full["test_metrics"]
    ct = no_concept["test_metrics"]
    nt = no_note["test_metrics"]
    lines = [
        "# Table 1 analogue: MINGLE on Coherent",
        "",
        "Same layout as Cui et al. (arXiv:2403.08818) Table 1 last three MINGLE rows. "
        "**Not** MIMIC-III or CRADLE. Do not paste these numbers next to the paper’s 80.17 ACC as a beat/loss.",
        "",
        "## Protocol",
        "",
        "- Parent model: frozen MINGLE, **vanilla BCE** (paper Eq. 4). No class weights, no focal loss.",
        "- d=48, h=4, L=2, 25 next-visit SNOMED labels, patient split 70/15/15, seed **42** (paper averages **5** seeds).",
        "- **MINGLE**: DeepWalk `s_v` + concept semantics `C_v` + note semantics `N_e`.",
        "- **w/o Medical Concept Semantics**: `C_v` zeroed in `X_v=[s_v;C_v]` and in `N_aug` self-loop rows. Notes and DeepWalk kept.",
        "- **w/o Clinical Note Semantics**: `N_e` zeroed. `C_v` and DeepWalk kept.",
        "- Best checkpoint = lowest unweighted val BCE. 20 epochs.",
        "",
        "## Column definitions (aligned to the paper’s names)",
        "",
        "- **ACC**: Hamming accuracy at 0.5 (fraction of all label cells correct). Multi-label; always-negative is already high because labels are rare.",
        "- **AUROC**: macro-AUROC over 25 classes.",
        "- **AUPR**: macro-AUPRC over 25 classes (ranking; use this, not ACC, to judge ablations on Coherent).",
        "- **F1**: micro-F1 at threshold 0.5.",
        "",
        "## Results (test, %)",
        "",
        "| Model | ACC | AUROC | AUPR | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        _row("MINGLE", ft),
        _row("MINGLE w/o Medical Concept Semantics", ct),
        _row("MINGLE w/o Clinical Note Semantics", nt),
        "",
        f"- full MINGLE best epoch: `{full.get('best_epoch')}`",
        f"- no `C_v` best epoch: `{no_concept.get('best_epoch')}`",
        f"- no `N_e` best epoch: `{no_note.get('best_epoch')}`",
        "",
        "## Extra (not in the paper table)",
        "",
        "| Model | unweighted BCE | micro-AUPRC | macro-F1 @0.5 |",
        "| --- | ---: | ---: | ---: |",
        f"| MINGLE | {ft['bce']:.6f} | {ft['micro_auprc']} | {ft['macro_f1']:.4f} |",
        f"| w/o concept semantics | {ct['bce']:.6f} | {ct['micro_auprc']} | {ct['macro_f1']:.4f} |",
        f"| w/o note semantics | {nt['bce']:.6f} | {nt['micro_auprc']} | {nt['macro_f1']:.4f} |",
        "",
        "## What to tell a reader",
        "",
        "On MIMIC-III the paper’s full MINGLE beats both ablations on every column. "
        "On Coherent, vanilla BCE often predicts almost no positives at 0.5, so **F1 can sit at 0** for all three rows. "
        "That does not mean the ablations are identical — compare **AUPR** and **AUROC**.",
        "",
        "A later Weighted BCE run is a Coherent imbalance experiment, not a paper Table 1 row.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def table1_from_stored_vanilla() -> dict:
    """If full vanilla was not retrained this session, use stored test metrics (no ACC)."""
    stored = {
        "best_epoch": VANILLA_BCE_TEST.get("best_epoch", 16),
        "test_metrics": {
            "accuracy": None,
            "macro_auroc": VANILLA_BCE_TEST["macro_auroc"],
            "macro_auprc": VANILLA_BCE_TEST["macro_auprc"],
            "micro_f1": VANILLA_BCE_TEST["micro_f1"],
            "macro_f1": VANILLA_BCE_TEST["macro_f1"],
            "micro_auprc": VANILLA_BCE_TEST["micro_auprc"],
            "bce": VANILLA_BCE_TEST["bce"],
        },
    }
    return stored
