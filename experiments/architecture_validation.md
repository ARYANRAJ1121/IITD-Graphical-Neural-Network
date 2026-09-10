# Architecture validation report

Generated after the frozen mathematical spec (loss = MINGLE Eq. (4) / `BCEWithLogitsLoss`).
No focal loss, class weights, or oversampling.

- Forward-pass bundle source: `processed`
- Processed Coherent tensors loaded: `True`
- Forward-pass loss (masked BCEWithLogitsLoss): `0.696014`

## Production hypergraph (frozen Hypergraph construction / Vanilla BCE spec)

- Real encounter hyperedges: **143946**
- Self-loop hyperedges: **564**
- Total augmented hyperedges: **144510**
- Concept nodes: **564**
- Incidence: `A ∈ {0,1}^{564 × 144510}`
- Real hyperedges occupy columns `[0, 143946)`
- Self-loop hyperedge `143946 + i` is incident only to node `i`
- Self-loop hyperedges are excluded from `MLP_CLS` and from the prediction loss

## Production tensor shapes

| Tensor | Shape | Role |
| --- | --- | --- |
| `X_v^(0)` | `(564, 832)` | `[S_v ; C_v]` |
| `C_v` | `(564, 768)` | concept semantics |
| `N_e` | `(143946, 768)` | visit notes |
| `N_aug = [N_e ; C_v]` | `(144510, 768)` | MLP_1 input |
| `H_e = MLP_1(N_aug)` | `(144510, 48)` | Eq. (6) |
| `A_e` layer 1 (`f_V→E` + PairNorm) | `(144510, 832)` | PMA over incident nodes |
| `E_e^(1) = MLP_2([A_e ; H_e])` | `(144510, 48)` | Eq. (7) |
| `X_v^(1)` (`f_E→V` + PairNorm) | `(564, 48)` | PMA over incident hyperedges |
| `A_e` layer 2 | `(144510, 48)` | |
| `E_e^(2)` | `(144510, 48)` | |
| `X_v^(2)` | `(564, 48)` | |
| JK-CONCAT real visits | `(143946, 96)` | `[E_e^(1)[real] ‖ E_e^(2)[real]]` |
| `MLP_CLS` logits | `(143946, 25)` | 25 targets |

## MLP dimensions

- MLP_1 (Eq. 6): `768 → 48` (`nn.Linear`)
- MLP_2 layer 1 (Eq. 7): `880 → 48`
- MLP_2 layer 2 (Eq. 7): `96 → 48`
- MLP_CLS: `96 → 25`

## PMA

- Learned queries `W_i^Q ∈ R^{1 × d_k}` with `h=4`
- Layer 1 `f_V→E`: `d=832`, `d_k=208`, output `[num_hyperedges, 832]`
- Layer 1 `f_E→V` and all layer-2 PMA: `d=48`, `d_k=12`, output `[num_groups, 48]`
- Not `nn.MultiheadAttention`. Not node-to-node self-attention.

## PairNorm

- Applied to PMA+FFN outputs: `A_e` then `X_v` at each layer
- Output shapes match the tensors above

## This forward-pass (measured)

- Nodes: `564`
- Real hyperedges: `143946`
- Self-loop hyperedges: `564`
- Incidence nnz: `924214`
- `N_aug`: `(144510, 768)`
- `H_e`: `(144510, 48)`
- `A_e` (last layer): `(144510, 48)`
- `E_e` (last layer): `(144510, 48)`
- `X_v` (last layer): `(564, 48)`
- JK-CONCAT: `(143946, 96)`
- Classifier logits: `(143946, 25)`
- Valid t→t+1 pairs in bundle: `142668`
- Last visits excluded from loss: `1278`

## Parameter dimensions (this model instance)

| Parameter | Shape |
| --- | --- |
| `mlp1.weight` | `48×768` |
| `mlp1.bias` | `48` |
| `layers.0.node_to_edge.pma.query` | `4×208` |
| `layers.0.node_to_edge.pma.key_weight` | `4×832×208` |
| `layers.0.node_to_edge.pma.value_weight` | `4×832×208` |
| `layers.0.node_to_edge.pma.output_weight` | `832×832` |
| `layers.0.node_to_edge.ffn.0.weight` | `3328×832` |
| `layers.0.node_to_edge.ffn.0.bias` | `3328` |
| `layers.0.node_to_edge.ffn.2.weight` | `832×3328` |
| `layers.0.node_to_edge.ffn.2.bias` | `832` |
| `layers.0.mlp2.weight` | `48×880` |
| `layers.0.mlp2.bias` | `48` |
| `layers.0.edge_to_node.pma.query` | `4×12` |
| `layers.0.edge_to_node.pma.key_weight` | `4×48×12` |
| `layers.0.edge_to_node.pma.value_weight` | `4×48×12` |
| `layers.0.edge_to_node.pma.output_weight` | `48×48` |
| `layers.0.edge_to_node.ffn.0.weight` | `192×48` |
| `layers.0.edge_to_node.ffn.0.bias` | `192` |
| `layers.0.edge_to_node.ffn.2.weight` | `48×192` |
| `layers.0.edge_to_node.ffn.2.bias` | `48` |
| `layers.1.node_to_edge.pma.query` | `4×12` |
| `layers.1.node_to_edge.pma.key_weight` | `4×48×12` |
| `layers.1.node_to_edge.pma.value_weight` | `4×48×12` |
| `layers.1.node_to_edge.pma.output_weight` | `48×48` |
| `layers.1.node_to_edge.ffn.0.weight` | `192×48` |
| `layers.1.node_to_edge.ffn.0.bias` | `192` |
| `layers.1.node_to_edge.ffn.2.weight` | `48×192` |
| `layers.1.node_to_edge.ffn.2.bias` | `48` |
| `layers.1.mlp2.weight` | `48×96` |
| `layers.1.mlp2.bias` | `48` |
| `layers.1.edge_to_node.pma.query` | `4×12` |
| `layers.1.edge_to_node.pma.key_weight` | `4×48×12` |
| `layers.1.edge_to_node.pma.value_weight` | `4×48×12` |
| `layers.1.edge_to_node.pma.output_weight` | `48×48` |
| `layers.1.edge_to_node.ffn.0.weight` | `192×48` |
| `layers.1.edge_to_node.ffn.0.bias` | `192` |
| `layers.1.edge_to_node.ffn.2.weight` | `48×192` |
| `layers.1.edge_to_node.ffn.2.bias` | `48` |
| `classifier.weight` | `25×96` |
| `classifier.bias` | `25` |

## Train / validation / test

Protocol: **patient-level** split `70% / 15% / 15%`, seed `42`.
Examples are chronological `t → t+1` pairs (last visit of each patient excluded).

| Split | Examples (masked pairs) |
| --- | ---: |
| train | 99695 |
| val | 21124 |
| test | 21849 |

## Label prevalence (25 classes, among pairs used in this bundle's loss)

| Rank | SNOMED | Name | Positives |
| --- | --- | --- | --- |
| 1 | `265764009` | Renal dialysis (procedure) | 13508 (9.47%) |
| 2 | `430193006` | Medication Reconciliation (procedure) | 11176 (7.83%) |
| 3 | `180325003` | Electrical cardioversion | 6616 (4.64%) |
| 4 | `40701008` | Echocardiography (procedure) | 2388 (1.67%) |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 1814 (1.27%) |
| 6 | `18286008` | Catheter ablation of tissue of heart | 1734 (1.22%) |
| 7 | `444814009` | Viral sinusitis (disorder) | 1415 (0.99%) |
| 8 | `431182000` | Placing subject in prone position (procedure) | 1343 (0.94%) |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 1343 (0.94%) |
| 10 | `29303009` | Electrocardiographic procedure | 1081 (0.76%) |
| 11 | `73761001` | Colonoscopy | 990 (0.69%) |
| 12 | `399208008` | Plain chest X-ray (procedure) | 942 (0.66%) |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 773 (0.54%) |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 473 (0.33%) |
| 15 | `230690007` | Stroke | 692 (0.49%) |
| 16 | `127783003` | Spirometry (procedure) | 649 (0.45%) |
| 17 | `180256009` | Subcutaneous immunotherapy | 648 (0.45%) |
| 18 | `15777000` | Prediabetes | 505 (0.35%) |
| 19 | `423475008` | Heart failure education (procedure) | 600 (0.42%) |
| 20 | `10509002` | Acute bronchitis (disorder) | 595 (0.42%) |
| 21 | `271737000` | Anemia (disorder) | 525 (0.37%) |
| 22 | `410006001` | Digital examination of rectum | 589 (0.41%) |
| 23 | `71651007` | Mammography (procedure) | 568 (0.40%) |
| 24 | `312681000` | Bone density scan (procedure) | 547 (0.38%) |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 521 (0.37%) |

## Leakage checks

- Current-visit inputs are `N_e[t]`, the incidence row of encounter `t`, and frozen `X_v^(0)` / `C_v`.
- Targets `y[t]` are top-25 membership of visit **t+1** only; visit t+1 notes and t+1 labels are not concatenated into the visit-t feature vector.
- `pair_mask[t]` is true only when the next row is the same patient (chronological sort by `patient_id`, `start_date`).
- Last visits have `pair_mask=False` and are excluded from `BCEWithLogitsLoss`.
- `MLP_CLS` is applied to the first `num_real` hyperedges; the 564 (production) self-loops are never classified.
- Frozen transductive incidence still allows concept nodes to aggregate **all** incident hyperedges in `f_E→V` (including later visits of the same patient). That is the audited global hypergraph, not an extra mask. Direct t+1 labels/notes are not inputs to visit t.

## Forward-pass status

**PASS** — one forward pass and masked Eq. (4) loss completed. Full training was not started.
