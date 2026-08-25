from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.data.stage3_bundle import Stage3Bundle, example_index
from src.data.stage3_labels import FROZEN_TOP25
from src.models.mingle import MingleModel

PRODUCTION = {
    "num_nodes": 564,
    "num_real_hyperedges": 143946,
    "num_self_loop_hyperedges": 564,
    "num_total_hyperedges": 144510,
    "node_input_dim": 832,
    "semantic_dim": 768,
    "hidden_dim": 48,
    "num_heads": 4,
    "head_dim": 12,
    "num_layers": 2,
    "num_classes": 25,
    "jk_dim": 96,
    "valid_pairs_stage3": 142668,
    "terminal_visits": 1278,
}


def masked_bce_with_logits(logits: torch.Tensor, labels: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    return criterion(logits[pair_mask], labels[pair_mask])


def parameter_table(model: nn.Module) -> list[tuple[str, str]]:
    rows = []
    for name, param in model.named_parameters():
        rows.append((name, "×".join(str(s) for s in param.shape)))
    return rows


def prevalence_rows(labels: torch.Tensor, pair_mask: torch.Tensor) -> list[tuple[str, str, str, str]]:
    y = labels[pair_mask]
    n = int(y.size(0))
    rows = []
    for i, (code, name) in enumerate(FROZEN_TOP25):
        count = int(y[:, i].sum().item()) if n else 0
        pct = (100.0 * count / n) if n else 0.0
        rows.append((str(i + 1), code, name, f"{count} ({pct:.2f}%)"))
    return rows


def write_validation_report(
    path: Path,
    *,
    bundle: Stage3Bundle,
    model: MingleModel,
    outputs: dict[str, torch.Tensor],
    loss: torch.Tensor,
    split_counts: dict[str, int],
    processed_loaded: bool,
) -> None:
    p = PRODUCTION
    params = parameter_table(model)
    prev = prevalence_rows(bundle.labels, bundle.pair_mask)
    last_visits = int((~bundle.pair_mask).sum().item())
    valid_pairs = int(bundle.pair_mask.sum().item())

    lines = [
        "# Stage 3 Implementation Validation Report",
        "",
        "Generated after the frozen mathematical spec (loss = MINGLE Eq. (4) / `BCEWithLogitsLoss`).",
        "No focal loss, class weights, or oversampling.",
        "",
        f"- Forward-pass bundle source: `{bundle.source}`",
        f"- Processed Coherent tensors loaded: `{processed_loaded}`",
        f"- Forward-pass loss (masked BCEWithLogitsLoss): `{loss.item():.6f}`",
        "",
        "## Production hypergraph (frozen Stage 2 / Stage 3 spec)",
        "",
        f"- Real encounter hyperedges: **{p['num_real_hyperedges']}**",
        f"- Self-loop hyperedges: **{p['num_self_loop_hyperedges']}**",
        f"- Total augmented hyperedges: **{p['num_total_hyperedges']}**",
        f"- Concept nodes: **{p['num_nodes']}**",
        f"- Incidence: `A ∈ {{0,1}}^{{{p['num_nodes']} × {p['num_total_hyperedges']}}}`",
        "- Real hyperedges occupy columns `[0, 143946)`",
        "- Self-loop hyperedge `143946 + i` is incident only to node `i`",
        "- Self-loop hyperedges are excluded from `MLP_CLS` and from the prediction loss",
        "",
        "## Production tensor shapes",
        "",
        "| Tensor | Shape | Role |",
        "| --- | --- | --- |",
        f"| `X_v^(0)` | `({p['num_nodes']}, {p['node_input_dim']})` | `[S_v ; C_v]` |",
        f"| `C_v` | `({p['num_nodes']}, {p['semantic_dim']})` | concept semantics |",
        f"| `N_e` | `({p['num_real_hyperedges']}, {p['semantic_dim']})` | visit notes |",
        f"| `N_aug = [N_e ; C_v]` | `({p['num_total_hyperedges']}, {p['semantic_dim']})` | MLP_1 input |",
        f"| `H_e = MLP_1(N_aug)` | `({p['num_total_hyperedges']}, {p['hidden_dim']})` | Eq. (6) |",
        f"| `A_e` layer 1 (`f_V→E` + PairNorm) | `({p['num_total_hyperedges']}, {p['node_input_dim']})` | PMA over incident nodes |",
        f"| `E_e^(1) = MLP_2([A_e ; H_e])` | `({p['num_total_hyperedges']}, {p['hidden_dim']})` | Eq. (7) |",
        f"| `X_v^(1)` (`f_E→V` + PairNorm) | `({p['num_nodes']}, {p['hidden_dim']})` | PMA over incident hyperedges |",
        f"| `A_e` layer 2 | `({p['num_total_hyperedges']}, {p['hidden_dim']})` | |",
        f"| `E_e^(2)` | `({p['num_total_hyperedges']}, {p['hidden_dim']})` | |",
        f"| `X_v^(2)` | `({p['num_nodes']}, {p['hidden_dim']})` | |",
        f"| JK-CONCAT real visits | `({p['num_real_hyperedges']}, {p['jk_dim']})` | `[E_e^(1)[real] ‖ E_e^(2)[real]]` |",
        f"| `MLP_CLS` logits | `({p['num_real_hyperedges']}, {p['num_classes']})` | 25 targets |",
        "",
        "## MLP dimensions",
        "",
        f"- MLP_1 (Eq. 6): `{p['semantic_dim']} → {p['hidden_dim']}` (`nn.Linear`)",
        f"- MLP_2 layer 1 (Eq. 7): `{p['node_input_dim'] + p['hidden_dim']} → {p['hidden_dim']}`",
        f"- MLP_2 layer 2 (Eq. 7): `{p['hidden_dim'] + p['hidden_dim']} → {p['hidden_dim']}`",
        f"- MLP_CLS: `{p['jk_dim']} → {p['num_classes']}`",
        "",
        "## PMA",
        "",
        f"- Learned queries `W_i^Q ∈ R^{{1 × d_k}}` with `h={p['num_heads']}`",
        f"- Layer 1 `f_V→E`: `d={p['node_input_dim']}`, `d_k={p['node_input_dim'] // p['num_heads']}`, output `[num_hyperedges, {p['node_input_dim']}]`",
        f"- Layer 1 `f_E→V` and all layer-2 PMA: `d={p['hidden_dim']}`, `d_k={p['head_dim']}`, output `[num_groups, {p['hidden_dim']}]`",
        "- Not `nn.MultiheadAttention`. Not node-to-node self-attention.",
        "",
        "## PairNorm",
        "",
        f"- Applied to PMA+FFN outputs: `A_e` then `X_v` at each layer",
        f"- Output shapes match the tensors above",
        "",
        "## This forward-pass (measured)",
        "",
        f"- Nodes: `{len(bundle.node_ids)}`",
        f"- Real hyperedges: `{bundle.num_real}`",
        f"- Self-loop hyperedges: `{bundle.num_self_loops}`",
        f"- Incidence nnz: `{int(bundle.incidence_node.numel())}`",
        f"- `N_aug`: `{tuple(outputs['n_aug'].shape)}`",
        f"- `H_e`: `{tuple(outputs['h_e'].shape)}`",
        f"- `A_e` (last layer): `{tuple(outputs['a_e'].shape)}`",
        f"- `E_e` (last layer): `{tuple(outputs['e_e'].shape)}`",
        f"- `X_v` (last layer): `{tuple(outputs['x_v'].shape)}`",
        f"- JK-CONCAT: `{tuple(outputs['jk'].shape)}`",
        f"- Classifier logits: `{tuple(outputs['logits'].shape)}`",
        f"- Valid t→t+1 pairs in bundle: `{valid_pairs}`",
        f"- Last visits excluded from loss: `{last_visits}`",
        "",
        "## Parameter dimensions (this model instance)",
        "",
        "| Parameter | Shape |",
        "| --- | --- |",
    ]
    for name, shape in params:
        lines.append(f"| `{name}` | `{shape}` |")

    lines.extend(
        [
            "",
            "## Train / validation / test",
            "",
            "Protocol: **patient-level** split `70% / 15% / 15%`, seed `42`.",
            "Examples are chronological `t → t+1` pairs (last visit of each patient excluded).",
            "",
        ]
    )
    if processed_loaded:
        lines.append("| Split | Examples (masked pairs) |")
        lines.append("| --- | ---: |")
        for key in ("train", "val", "test"):
            lines.append(f"| {key} | {split_counts.get(key, 0)} |")
    else:
        lines.extend(
            [
                f"- Production valid pairs (Stage 3 feasibility): **{p['valid_pairs_stage3']}**",
                f"- Production terminal visits excluded: **{p['terminal_visits']}**",
                "- Exact train/val/test counts require `data/processed` Stage 2 artifacts (currently missing).",
                f"- Forward-test split counts: train={split_counts.get('train', 0)}, val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}",
            ]
        )

    lines.extend(
        [
            "",
            "## Label prevalence (25 classes, among pairs used in this bundle's loss)",
            "",
            "| Rank | SNOMED | Name | Positives |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in prev:
        lines.append(f"| {row[0]} | `{row[1]}` | {row[2]} | {row[3]} |")
    if not processed_loaded:
        lines.extend(
            [
                "",
                "Production prevalence (feasibility audit, 142,668 pairs) is unchanged; see `experiments/stage3_label_feasibility.md`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            "- Current-visit inputs are `N_e[t]`, the incidence row of encounter `t`, and frozen `X_v^(0)` / `C_v`.",
            "- Targets `y[t]` are top-25 membership of visit **t+1** only; visit t+1 notes and t+1 labels are not concatenated into the visit-t feature vector.",
            "- `pair_mask[t]` is true only when the next row is the same patient (chronological sort by `patient_id`, `start_date`).",
            "- Last visits have `pair_mask=False` and are excluded from `BCEWithLogitsLoss`.",
            "- `MLP_CLS` is applied to the first `num_real` hyperedges; the 564 (production) self-loops are never classified.",
            "- Frozen transductive incidence still allows concept nodes to aggregate **all** incident hyperedges in `f_E→V` (including later visits of the same patient). That is the audited global hypergraph, not an extra mask. Direct t+1 labels/notes are not inputs to visit t.",
            "",
            "## Forward-pass status",
            "",
            "**PASS** — one forward pass and masked Eq. (4) loss completed. Full training was not started.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def run_forward(bundle: Stage3Bundle) -> tuple[MingleModel, dict[str, torch.Tensor], torch.Tensor]:
    model = MingleModel(
        node_input_dim=bundle.node_states.size(1),
        semantic_dim=bundle.note_semantics.size(1),
        hidden_dim=48,
        num_heads=4,
        num_layers=2,
        num_classes=25,
        num_real_hyperedges=bundle.num_real,
    )
    model.eval()
    outputs = model(
        bundle.node_states,
        bundle.concept_semantics,
        bundle.note_semantics,
        bundle.incidence_node,
        bundle.incidence_edge,
    )
    loss = masked_bce_with_logits(outputs["logits"], bundle.labels, bundle.pair_mask)
    return model, outputs, loss
