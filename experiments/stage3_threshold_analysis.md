# Stage 3 Threshold Sensitivity Analysis

Checkpoint frozen. No retraining. Architecture, loss, and optimizer unchanged.
Threshold chosen on **validation only**. Test is scored once at that threshold.
A higher F1 at a lower threshold is not treated as model success by itself.

- checkpoint: `data\processed\checkpoints\stage3_best.pt`
- checkpoint epoch: `16`
- checkpoint val BCE: `0.06223045289516449`
- selection rule: maximize validation **micro-F1**; ties broken by validation macro-F1, then by the larger threshold
- selected threshold: **0.2**

## Threshold-independent ranking (unchanged)

| Split | macro-AUROC | micro-AUPRC | macro-AUPRC |
| --- | ---: | ---: | ---: |
| validation | 0.6907142665016985 | 0.09424123624304347 | 0.04060229568453863 |
| test | 0.6545617075251373 | 0.15783638442963063 | 0.06427900818886663 |

## Predicted probability calibration (sigmoid logits)

| Split | mean | std | min | max | p50 | p90 | p99 | share ≥ 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.010599 | 0.019144 | 0.000136 | 0.620399 | 0.006852 | 0.023828 | 0.052411 | 0.0002% |
| test | 0.011205 | 0.019839 | 0.000121 | 0.419333 | 0.007219 | 0.024567 | 0.054609 | 0.0000% |

## Validation sweep (used for selection)

| τ | micro-F1 | macro-F1 | micro-P | micro-R | macro-P | macro-R | # pos | % pos |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.010 | 0.0448 | 0.0369 | 0.0232 | 0.6051 | 0.0220 | 0.4287 | 170523 | 32.2899% |
| 0.020 | 0.0465 | 0.0317 | 0.0256 | 0.2582 | 0.0236 | 0.1783 | 66140 | 12.5241% |
| 0.030 | 0.0639 | 0.0368 | 0.0384 | 0.1884 | 0.0361 | 0.1163 | 32096 | 6.0776% |
| 0.050 | 0.1372 | 0.0303 | 0.1422 | 0.1325 | 0.0433 | 0.0469 | 6102 | 1.1555% |
| 0.075 | 0.1752 | 0.0248 | 0.3918 | 0.1128 | 0.0311 | 0.0273 | 1886 | 0.3571% |
| 0.100 | 0.1828 | 0.0233 | 0.5112 | 0.1113 | 0.0415 | 0.0249 | 1426 | 0.2700% |
| 0.150 | 0.1845 | 0.0233 | 0.5458 | 0.1110 | 0.0219 | 0.0248 | 1332 | 0.2522% |
| 0.200 | 0.1850 | 0.0234 | 0.5545 | 0.1110 | 0.0222 | 0.0248 | 1311 | 0.2482% ← selected |
| 0.250 | 0.1705 | 0.0226 | 0.5729 | 0.1002 | 0.0229 | 0.0224 | 1145 | 0.2168% |
| 0.300 | 0.1701 | 0.0227 | 0.5784 | 0.0997 | 0.0231 | 0.0222 | 1129 | 0.2138% |
| 0.400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 0.0002% |
| 0.500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 0.0002% |

## Test sweep (reference only; not used for selection)

| τ | micro-F1 | macro-F1 | micro-P | micro-R | macro-P | macro-R | # pos | % pos |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.010 | 0.0445 | 0.0395 | 0.0230 | 0.6380 | 0.0233 | 0.4368 | 193737 | 35.4684% |
| 0.020 | 0.0548 | 0.0341 | 0.0300 | 0.3131 | 0.0245 | 0.1854 | 72971 | 13.3591% |
| 0.030 | 0.0809 | 0.0451 | 0.0481 | 0.2557 | 0.0410 | 0.1325 | 37208 | 6.8118% |
| 0.050 | 0.1767 | 0.0435 | 0.1707 | 0.1832 | 0.0532 | 0.0652 | 7511 | 1.3751% |
| 0.075 | 0.2209 | 0.0355 | 0.4928 | 0.1423 | 0.0705 | 0.0397 | 2021 | 0.3700% |
| 0.100 | 0.2293 | 0.0332 | 0.7049 | 0.1369 | 0.0301 | 0.0370 | 1359 | 0.2488% |
| 0.150 | 0.2324 | 0.0336 | 0.7670 | 0.1369 | 0.0307 | 0.0370 | 1249 | 0.2287% |
| 0.200 | 0.2327 | 0.0337 | 0.7751 | 0.1369 | 0.0310 | 0.0370 | 1236 | 0.2263% |
| 0.250 | 0.2161 | 0.0322 | 0.7659 | 0.1258 | 0.0306 | 0.0340 | 1149 | 0.2104% |
| 0.300 | 0.2129 | 0.0324 | 0.7892 | 0.1231 | 0.0316 | 0.0333 | 1091 | 0.1997% |
| 0.400 | 0.0046 | 0.0012 | 0.8421 | 0.0023 | 0.0337 | 0.0006 | 19 | 0.0035% |
| 0.500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000% |

## Locked test evaluation at validation-selected τ = 0.2

- micro-F1: `0.2327`
- macro-F1: `0.0337`
- micro-precision: `0.7751`
- micro-recall: `0.1369`
- macro-precision: `0.0310`
- macro-recall: `0.0370`
- positive predictions: `1236` (0.2263% of label cells)

| Rank | Code | Name | Precision | Recall | F1 | Support |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `265764009` | Renal dialysis (procedure) | 0.7751 | 0.9256 | 0.8437 | 1035 |
| 2 | `430193006` | Medication Reconciliation (procedure) | 0.0000 | 0.0000 | 0.0000 | 1885 |
| 3 | `180325003` | Electrical cardioversion | 0.0000 | 0.0000 | 0.0000 | 1030 |
| 4 | `40701008` | Echocardiography (procedure) | 0.0000 | 0.0000 | 0.0000 | 356 |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 0.0000 | 0.0000 | 0.0000 | 355 |
| 6 | `18286008` | Catheter ablation of tissue of heart | 0.0000 | 0.0000 | 0.0000 | 240 |
| 7 | `444814009` | Viral sinusitis (disorder) | 0.0000 | 0.0000 | 0.0000 | 197 |
| 8 | `431182000` | Placing subject in prone position (procedure) | 0.0000 | 0.0000 | 0.0000 | 174 |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 0.0000 | 0.0000 | 0.0000 | 174 |
| 10 | `29303009` | Electrocardiographic procedure | 0.0000 | 0.0000 | 0.0000 | 141 |
| 11 | `73761001` | Colonoscopy | 0.0000 | 0.0000 | 0.0000 | 156 |
| 12 | `399208008` | Plain chest X-ray (procedure) | 0.0000 | 0.0000 | 0.0000 | 124 |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 0.0000 | 0.0000 | 0.0000 | 113 |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 0.0000 | 0.0000 | 0.0000 | 69 |
| 15 | `230690007` | Stroke | 0.0000 | 0.0000 | 0.0000 | 110 |
| 16 | `127783003` | Spirometry (procedure) | 0.0000 | 0.0000 | 0.0000 | 131 |
| 17 | `180256009` | Subcutaneous immunotherapy | 0.0000 | 0.0000 | 0.0000 | 60 |
| 18 | `15777000` | Prediabetes | 0.0000 | 0.0000 | 0.0000 | 72 |
| 19 | `423475008` | Heart failure education (procedure) | 0.0000 | 0.0000 | 0.0000 | 78 |
| 20 | `10509002` | Acute bronchitis (disorder) | 0.0000 | 0.0000 | 0.0000 | 89 |
| 21 | `271737000` | Anemia (disorder) | 0.0000 | 0.0000 | 0.0000 | 74 |
| 22 | `410006001` | Digital examination of rectum | 0.0000 | 0.0000 | 0.0000 | 90 |
| 23 | `71651007` | Mammography (procedure) | 0.0000 | 0.0000 | 0.0000 | 74 |
| 24 | `312681000` | Bone density scan (procedure) | 0.0000 | 0.0000 | 0.0000 | 83 |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.0000 | 0.0000 | 0.0000 | 87 |

## Interpretation

At τ=0.50, validation micro-F1 is `0.0000` with `1` positive predictions (0.0002% of cells). Mean predicted probability is `0.010599` (p99 `0.052411`), so a 0.5 cut sits far above the model's score mass.

If F1 rises at lower thresholds while AUROC/AUPRC stay fixed, the 0.5-threshold zero-F1 is consistent with **miscalibration / imbalance operating-point mismatch**, not with a complete absence of ranking signal.

That does **not** make the run successful: a lower threshold also increases false positives, and F1 can be inflated by over-predicting. Compare AUPRC and per-class F1 before claiming utility.
