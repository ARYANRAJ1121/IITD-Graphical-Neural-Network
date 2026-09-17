# MINGLE on Coherent

Reproduction of [MINGLE](https://arxiv.org/abs/2403.08818) (Cui et al., *Multimodal Fusion of EHR with Hypergraph Neural Networks*) on the **Coherent Synthea** dump (`11-07-2022`) instead of MIMIC-III.

The model is a hypergraph GNN: medical **concepts** are nodes, **visits** are hyperedges, **DeepWalk** encodes co-occurrence structure, and **BioMedBERT** encodes concept names and clinical notes. The task is **next-visit prediction** of the 25 most frequent SNOMED concepts.

## Dataset vs the paper

| | Paper (MIMIC-III) | This repo (Coherent) |
|---|---|---|
| Setting | Real ICU | Synthetic general EHR (Synthea FHIR) |
| Patients / visits | ~37k visits | **1,278** patients, **143,946** visits |
| Concept nodes | ~7.4k codes | **564** unique coded concepts |
| Labels | 25 ICU phenotypes | Top-25 SNOMED concepts in this dump |
| Split | 7 : 1 : 2 | **70 / 15 / 15**, patient-level, seed **42** |
| Text encoder | `text-embedding-ada-002` (1536-d) | `NeuML/biomedbert-base-embeddings` (768-d) |
| Notes | Discharge / clinical notes | FHIR `DocumentReference` (every visit) |

Same MINGLE *recipe* (patients → visits → codes + notes → next visit). Different hospital world and much sparser labels. **Do not compare F1@0.5 to the paper’s ~46 as a leaderboard score.**

Raw FHIR lives next to this repo (`../fhir/`). DNA/DICOM in the dump are unused.

## Frozen model

Used for every training run unless a later experiment says otherwise:

- Hidden dim **d = 48**, **h = 4** heads, **L = 2** layers, **25** classes  
- Custom PMA (learned query), PairNorm, MLP₁ (768→48), MLP₂, self-loops, JK-CONCAT  
- Adam, lr **0.001**, no weight decay  
- Full-graph training (CPU unless CUDA is available)  
- Best checkpoint = lowest **unweighted** validation BCE  
- F1 reported at threshold **0.5**

## Pipeline

1. **Dataset profile** — inventory FHIR bundles (`src/data/profile_dataset.py`). Report: `experiments/dataset_profile.md`.
2. **Hypergraph + embeddings** — nodes, visit hyperedges, DeepWalk, BioMedBERT. Script: `src/data/construct_hypergraph.py`. Report: `experiments/hypergraph_embeddings.md`.
3. **Vanilla BCE** — paper Eq. (4), no class weights. `python train.py`. Report: `experiments/vanilla_bce_training_report.md`.
4. **Weighted BCE** — same model; train-only `pos_weight = n_neg/n_pos`. `python train_weighted_bce.py` (20 epochs) or `python train_weighted_bce_converge.py` (50 epochs). Reports: `experiments/weighted_bce_report.md`, `experiments/weighted_bce_convergence_report.md`.
5. **Focal loss** — same model; γ=2, α=1 (no class weights). `python train_focal_loss.py`.
6. **Note ablation (weighted)** — extra Coherent run: `N_e` zeros + weighted BCE. `python train_note_ablation.py`.
7. **Paper Table 1 ablations** — vanilla BCE parent: w/o `N_e` and w/o `C_v`. `train_vanilla_note_ablation` / `python train_concept_ablation.py`. Combined report: `experiments/paper_table1_coherent.md`.

## Results so far (test)

| | Vanilla BCE | Weighted BCE (50-epoch) |
|---|---:|---:|
| Unweighted BCE | 0.062 | 0.347 |
| macro-AUPRC | 0.064 | **0.271** |
| micro-AUPRC | 0.158 | **0.339** |
| macro-AUROC | 0.655 | **0.910** |
| micro-F1 @ 0.5 | 0.00 | **0.12** |
| macro-F1 @ 0.5 | 0.00 | **0.14** |
| Classes above vanilla AUPRC | — | **25 / 25** |

Vanilla BCE looks good on loss but predicts almost no positives (F1@0.5 = 0). Weighted BCE is the last **completed** experiment.

## Setup

```bash
cd mingle-coherent
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Point `configs/base.yaml` at your FHIR directory. Processed tensors (`data/processed/`) are **gitignored** (~note embeddings, graphs, checkpoints). Copy that folder if you clone onto another machine.

```bash
python -m src.data.profile_dataset
python -m src.data.construct_hypergraph   # long on CPU (notes ~hours)
python train.py                           # vanilla BCE
python train_weighted_bce_converge.py     # weighted BCE, 50 epochs
pytest
```

## Layout

```
src/data/          FHIR profile, hypergraph build, processed bundle
src/models/        PMA, PairNorm, MINGLE layers
src/graph/         incidence + train/val/test graphs
src/training/      vanilla / weighted / focal / note-ablation loops
src/evaluation/    AUPRC, AUROC, F1, threshold / ranking diagnostics
experiments/       reports for each experiment
train*.py          entry points
```

Paper math: `experiments/equation_mapping.md`. Label sparsity: `experiments/label_feasibility.md`.
