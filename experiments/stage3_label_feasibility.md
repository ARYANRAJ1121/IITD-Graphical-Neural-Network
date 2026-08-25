# Stage 3 Label Feasibility Analysis

## Dataset Statistics
- Total Encounters (Hyperedges): 143946
- Valid $t ightarrow t+1$ training pairs: 142668
- Terminal visits (no next visit, excluded from loss): 1278

## Top 25 Candidate Conditions (Target Labels)
| Rank | Node ID | Condition Name | Positive Examples in $t+1$ | Prevalence (%) |
|------|---------|----------------|----------------------------|----------------|
| 1 | `265764009` | Renal dialysis (procedure) | 13508 | 9.47% |
| 2 | `430193006` | Medication Reconciliation (procedure) | 11176 | 7.83% |
| 3 | `180325003` | Electrical cardioversion | 6616 | 4.64% |
| 4 | `40701008` | Echocardiography (procedure) | 2388 | 1.67% |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 1814 | 1.27% |
| 6 | `18286008` | Catheter ablation of tissue of heart | 1734 | 1.22% |
| 7 | `444814009` | Viral sinusitis (disorder) | 1415 | 0.99% |
| 8 | `431182000` | Placing subject in prone position (procedure) | 1343 | 0.94% |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 1343 | 0.94% |
| 10 | `29303009` | Electrocardiographic procedure | 1081 | 0.76% |
| 11 | `73761001` | Colonoscopy | 990 | 0.69% |
| 12 | `399208008` | Plain chest X-ray (procedure) | 942 | 0.66% |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 773 | 0.54% |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 473 | 0.33% |
| 15 | `230690007` | Stroke | 692 | 0.49% |
| 16 | `127783003` | Spirometry (procedure) | 649 | 0.45% |
| 17 | `180256009` | Subcutaneous immunotherapy | 648 | 0.45% |
| 18 | `15777000` | Prediabetes | 505 | 0.35% |
| 19 | `423475008` | Heart failure education (procedure) | 600 | 0.42% |
| 20 | `10509002` | Acute bronchitis (disorder) | 595 | 0.42% |
| 21 | `271737000` | Anemia (disorder) | 525 | 0.37% |
| 22 | `410006001` | Digital examination of rectum | 589 | 0.41% |
| 23 | `71651007` | Mammography (procedure) | 568 | 0.40% |
| 24 | `312681000` | Bone density scan (procedure) | 547 | 0.38% |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 521 | 0.37% |

## Conclusion
**WARNING: Label Sparsity Detected.** There are 0 classes with zero positive examples, and 19 classes with <1% prevalence in the next-visit targets. This task may lead to degenerate model training or all-zero predictions.