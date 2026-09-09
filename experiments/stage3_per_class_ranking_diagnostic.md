# Stage 3 Per-Class Ranking Diagnostic

Frozen checkpoint. No retraining, no architecture change, no class weights, no focal loss, no oversampling.
No decision threshold is applied. Metrics are ranking-only.
Aggregate micro-F1 is not used as a success criterion.

- checkpoint: `data\processed\checkpoints\stage3_best.pt`
- checkpoint epoch: `16`
- examples: train `99695`, val `21124`, test `21849`

## Scientific question

Does the frozen MINGLE model contain meaningful ranking signal across multiple target classes, or is almost all useful signal concentrated in renal dialysis?

## Per-class results (test ranking)

| # | Code | Name | Prev tr/va/te | Pos tr/va/te | AUROC | AUPRC | AUPRC/prev | mean p+ | mean p− | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `265764009` | Renal dialysis (procedure) | 0.1133/0.0556/0.0474 | 11299/1174/1035 | 0.9566 | 0.8130 | 17.16× | 0.3365 | 0.0236 | 0.0000 | 0.0000 | 0.6667 | 0.0019 | 0.6000 | 0.0029 |
| 2 | `430193006` | Medication Reconciliation (procedure) | 0.0773/0.0749/0.0863 | 7709/1582/1885 | 0.5762 | 0.1151 | 1.33× | 0.0144 | 0.0125 | 0.0000 | 0.0000 | 0.3333 | 0.0005 | 0.2000 | 0.0005 |
| 3 | `180325003` | Electrical cardioversion | 0.0504/0.0267/0.0471 | 5021/565/1030 | 0.7756 | 0.3146 | 6.67× | 0.0311 | 0.0129 | 0.0000 | 0.0000 | 0.6667 | 0.0019 | 0.8000 | 0.0039 |
| 4 | `40701008` | Echocardiography (procedure) | 0.0172/0.0148/0.0163 | 1719/313/356 | 0.6318 | 0.0305 | 1.87× | 0.0063 | 0.0047 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 0.0105/0.0196/0.0162 | 1046/413/355 | 0.8624 | 0.1394 | 8.58× | 0.0121 | 0.0056 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 6 | `18286008` | Catheter ablation of tissue of heart | 0.0134/0.0077/0.0110 | 1331/163/240 | 0.7045 | 0.0248 | 2.26× | 0.0095 | 0.0069 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 7 | `444814009` | Viral sinusitis (disorder) | 0.0101/0.0101/0.0090 | 1005/213/197 | 0.5575 | 0.0101 | 1.12× | 0.0106 | 0.0102 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | `431182000` | Placing subject in prone position (procedure) | 0.0098/0.0090/0.0080 | 979/190/174 | 0.6979 | 0.0165 | 2.08× | 0.0011 | 0.0007 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 0.0098/0.0090/0.0080 | 979/190/174 | 0.6894 | 0.0148 | 1.86× | 0.0428 | 0.0362 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 10 | `29303009` | Electrocardiographic procedure | 0.0080/0.0070/0.0065 | 793/147/141 | 0.7223 | 0.0126 | 1.95× | 0.0035 | 0.0027 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 11 | `73761001` | Colonoscopy | 0.0068/0.0075/0.0071 | 675/159/156 | 0.5769 | 0.0089 | 1.25× | 0.0097 | 0.0089 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | `399208008` | Plain chest X-ray (procedure) | 0.0068/0.0066/0.0057 | 678/140/124 | 0.5476 | 0.0067 | 1.18× | 0.0197 | 0.0187 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 0.0053/0.0063/0.0052 | 527/133/113 | 0.6232 | 0.0082 | 1.58× | 0.0041 | 0.0033 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 0.0032/0.0039/0.0032 | 322/82/69 | 0.6005 | 0.0049 | 1.56× | 0.0065 | 0.0056 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 15 | `230690007` | Stroke | 0.0048/0.0049/0.0050 | 479/103/110 | 0.5922 | 0.0068 | 1.34× | 0.0390 | 0.0360 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 16 | `127783003` | Spirometry (procedure) | 0.0042/0.0045/0.0060 | 422/96/131 | 0.4127 | 0.0051 | 0.86× | 0.0103 | 0.0113 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 17 | `180256009` | Subcutaneous immunotherapy | 0.0047/0.0057/0.0027 | 468/120/60 | 0.8200 | 0.0138 | 5.02× | 0.0068 | 0.0042 | 0.0000 | 0.0000 | 0.3333 | 0.0167 | 0.2000 | 0.0167 |
| 18 | `15777000` | Prediabetes | 0.0035/0.0038/0.0033 | 353/80/72 | 0.5381 | 0.0039 | 1.18× | 0.0022 | 0.0021 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 19 | `423475008` | Heart failure education (procedure) | 0.0044/0.0040/0.0036 | 437/85/78 | 0.8326 | 0.0110 | 3.08× | 0.0058 | 0.0036 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 20 | `10509002` | Acute bronchitis (disorder) | 0.0040/0.0049/0.0041 | 403/103/89 | 0.5840 | 0.0061 | 1.50× | 0.0153 | 0.0134 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 21 | `271737000` | Anemia (disorder) | 0.0036/0.0043/0.0034 | 361/90/74 | 0.6153 | 0.0151 | 4.45× | 0.0125 | 0.0083 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 22 | `410006001` | Digital examination of rectum | 0.0039/0.0054/0.0041 | 384/115/90 | 0.6203 | 0.0060 | 1.46× | 0.0136 | 0.0118 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 23 | `71651007` | Mammography (procedure) | 0.0036/0.0066/0.0034 | 354/140/74 | 0.5716 | 0.0039 | 1.15× | 0.0118 | 0.0116 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | `312681000` | Bone density scan (procedure) | 0.0039/0.0037/0.0038 | 386/78/83 | 0.6616 | 0.0100 | 2.63× | 0.0026 | 0.0016 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.0036/0.0036/0.0040 | 359/75/87 | 0.5932 | 0.0052 | 1.31× | 0.0084 | 0.0075 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Averages

| Metric | Macro (25 classes) | Micro (pooled test cells) |
| --- | ---: | ---: |
| AUROC | 0.6546 | 0.7029 |
| AUPRC | 0.0643 | 0.1578 |
| AUPRC/prevalence | 2.98× | NA |
| mean p+ | 0.0254 | NA |
| mean p− | 0.0106 | NA |
| P@1 | 0.0000 | NA |
| R@1 | 0.0000 | NA |
| P@3 | 0.0800 | NA |
| R@3 | 0.0008 | NA |
| P@5 | 0.0720 | NA |
| R@5 | 0.0010 | NA |

Micro P@k / R@k are omitted: those metrics are defined on a per-class ranked list.

## Rank by test AUPRC

| AUPRC rank | # | Name | AUPRC | AUPRC/prev | AUROC | test prev |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | Renal dialysis (procedure) | 0.8130 | 17.16× | 0.9566 | 0.0474 |
| 2 | 3 | Electrical cardioversion | 0.3146 | 6.67× | 0.7756 | 0.0471 |
| 3 | 5 | Combined chemotherapy and radiation therapy (procedure) | 0.1394 | 8.58× | 0.8624 | 0.0162 |
| 4 | 2 | Medication Reconciliation (procedure) | 0.1151 | 1.33× | 0.5762 | 0.0863 |
| 5 | 4 | Echocardiography (procedure) | 0.0305 | 1.87× | 0.6318 | 0.0163 |
| 6 | 6 | Catheter ablation of tissue of heart | 0.0248 | 2.26× | 0.7045 | 0.0110 |
| 7 | 8 | Placing subject in prone position (procedure) | 0.0165 | 2.08× | 0.6979 | 0.0080 |
| 8 | 21 | Anemia (disorder) | 0.0151 | 4.45× | 0.6153 | 0.0034 |
| 9 | 9 | Oxygen administration by mask (procedure) | 0.0148 | 1.86× | 0.6894 | 0.0080 |
| 10 | 17 | Subcutaneous immunotherapy | 0.0138 | 5.02× | 0.8200 | 0.0027 |
| 11 | 10 | Electrocardiographic procedure | 0.0126 | 1.95× | 0.7223 | 0.0065 |
| 12 | 19 | Heart failure education (procedure) | 0.0110 | 3.08× | 0.8326 | 0.0036 |
| 13 | 7 | Viral sinusitis (disorder) | 0.0101 | 1.12× | 0.5575 | 0.0090 |
| 14 | 24 | Bone density scan (procedure) | 0.0100 | 2.63× | 0.6616 | 0.0038 |
| 15 | 11 | Colonoscopy | 0.0089 | 1.25× | 0.5769 | 0.0071 |
| 16 | 13 | Acute viral pharyngitis (disorder) | 0.0082 | 1.58× | 0.6232 | 0.0052 |
| 17 | 15 | Stroke | 0.0068 | 1.34× | 0.5922 | 0.0050 |
| 18 | 12 | Plain chest X-ray (procedure) | 0.0067 | 1.18× | 0.5476 | 0.0057 |
| 19 | 20 | Acute bronchitis (disorder) | 0.0061 | 1.50× | 0.5840 | 0.0041 |
| 20 | 22 | Digital examination of rectum | 0.0060 | 1.46× | 0.6203 | 0.0041 |
| 21 | 25 | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.0052 | 1.31× | 0.5932 | 0.0040 |
| 22 | 16 | Spirometry (procedure) | 0.0051 | 0.86× | 0.4127 | 0.0060 |
| 23 | 14 | Body mass index 30+ - obesity (finding) | 0.0049 | 1.56× | 0.6005 | 0.0032 |
| 24 | 23 | Mammography (procedure) | 0.0039 | 1.15× | 0.5716 | 0.0034 |
| 25 | 18 | Prediabetes | 0.0039 | 1.18× | 0.5381 | 0.0033 |

## Rank by test AUROC

| AUROC rank | # | Name | AUROC | AUPRC | AUPRC/prev | test prev |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | Renal dialysis (procedure) | 0.9566 | 0.8130 | 17.16× | 0.0474 |
| 2 | 5 | Combined chemotherapy and radiation therapy (procedure) | 0.8624 | 0.1394 | 8.58× | 0.0162 |
| 3 | 19 | Heart failure education (procedure) | 0.8326 | 0.0110 | 3.08× | 0.0036 |
| 4 | 17 | Subcutaneous immunotherapy | 0.8200 | 0.0138 | 5.02× | 0.0027 |
| 5 | 3 | Electrical cardioversion | 0.7756 | 0.3146 | 6.67× | 0.0471 |
| 6 | 10 | Electrocardiographic procedure | 0.7223 | 0.0126 | 1.95× | 0.0065 |
| 7 | 6 | Catheter ablation of tissue of heart | 0.7045 | 0.0248 | 2.26× | 0.0110 |
| 8 | 8 | Placing subject in prone position (procedure) | 0.6979 | 0.0165 | 2.08× | 0.0080 |
| 9 | 9 | Oxygen administration by mask (procedure) | 0.6894 | 0.0148 | 1.86× | 0.0080 |
| 10 | 24 | Bone density scan (procedure) | 0.6616 | 0.0100 | 2.63× | 0.0038 |
| 11 | 4 | Echocardiography (procedure) | 0.6318 | 0.0305 | 1.87× | 0.0163 |
| 12 | 13 | Acute viral pharyngitis (disorder) | 0.6232 | 0.0082 | 1.58× | 0.0052 |
| 13 | 22 | Digital examination of rectum | 0.6203 | 0.0060 | 1.46× | 0.0041 |
| 14 | 21 | Anemia (disorder) | 0.6153 | 0.0151 | 4.45× | 0.0034 |
| 15 | 14 | Body mass index 30+ - obesity (finding) | 0.6005 | 0.0049 | 1.56× | 0.0032 |
| 16 | 25 | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.5932 | 0.0052 | 1.31× | 0.0040 |
| 17 | 15 | Stroke | 0.5922 | 0.0068 | 1.34× | 0.0050 |
| 18 | 20 | Acute bronchitis (disorder) | 0.5840 | 0.0061 | 1.50× | 0.0041 |
| 19 | 11 | Colonoscopy | 0.5769 | 0.0089 | 1.25× | 0.0071 |
| 20 | 2 | Medication Reconciliation (procedure) | 0.5762 | 0.1151 | 1.33× | 0.0863 |
| 21 | 23 | Mammography (procedure) | 0.5716 | 0.0039 | 1.15× | 0.0034 |
| 22 | 7 | Viral sinusitis (disorder) | 0.5575 | 0.0101 | 1.12× | 0.0090 |
| 23 | 12 | Plain chest X-ray (procedure) | 0.5476 | 0.0067 | 1.18× | 0.0057 |
| 24 | 18 | Prediabetes | 0.5381 | 0.0039 | 1.18× | 0.0033 |
| 25 | 16 | Spirometry (procedure) | 0.4127 | 0.0051 | 0.86× | 0.0060 |

## Comparison to prevalence baseline

Random ranking AUPRC ≈ class prevalence. Lift = AUPRC / prevalence. Lift ≈ 1 means no ranking value.

- renal dialysis AUPRC `0.8130` vs prevalence `0.0474` → lift `17.16×`
- mean AUPRC lift over the other 24 classes: `2.39×`
- mean AUROC over the other 24 classes: `0.6420`
- classes with AUPRC lift ≥ 2: `9/25`
- classes with AUROC ≥ 0.60: `15/25`

## Answer

**Almost all usable ranking mass is in renal dialysis**, with a short tail of weaker but real signal—not a flat 24-class null.

- Dialysis: AUROC 0.96, AUPRC 0.81, lift 17× vs 4.7% prevalence. This is the only class with high absolute AUPRC.
- Next by AUPRC: electrical cardioversion (0.31, lift 6.7×) and chemo+radiation (0.14, lift 8.6×). Several rare classes have AUROC 0.70–0.86 but AUPRC still near 0.01, so they rank better than chance without being useful at the top of the list.
- 16/25 classes have AUPRC lift &lt; 2 (near the prevalence baseline). Spirometry is below chance (AUROC 0.41, lift 0.86×).
- Medication reconciliation is frequent (8.6% test) but weak (AUROC 0.58, lift 1.33×).
- Top-k is fragile: P@1 is 0 for every class, including dialysis (the single highest score is a false positive).

So the frozen model is **not** a 25-class ranker. It is a dialysis-dominated ranker plus a few secondary procedure signals. That is not model success, and it does not follow from micro-F1.
