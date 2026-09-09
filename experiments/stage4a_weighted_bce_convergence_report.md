# Stage 4A: Weighted BCE (stage4a_weighted_bce_convergence)

Isolated loss-function experiment. Architecture, graph, embeddings, labels, split, and seed are frozen.
Positive-class weights from **training pairs only**. No focal loss, oversampling, or threshold tuning during training.
F1 uses the paper's 0.5 threshold. Checkpoint selected by **unweighted** validation BCE (same rule as Stage 3).

## Setup

- seed: `42`
- split: `[0.7, 0.15, 0.15]`
- epochs: `50`
- optimizer: Adam, lr `0.001`
- d=48, h=4, L=2, 25 classes, PMA + PairNorm + MLP_1/2 + self-loops + JK-CONCAT
- parameter count: `7782633`
- device: `cpu`
- hardware: `AMD64 Family 25 Model 124 Stepping 0, AuthenticAMD | cuda=False cpu`
- runtime_sec: `4075.6`
- best epoch: `49` (lowest unweighted val BCE)
- checkpoint: `data\processed\checkpoints\stage4a_converged_best.pt`
- early stopping: `patience=10 on unweighted val BCE; stopped_early=False`

## Train-only class weights (`n_neg / n_pos`)

| Rank | Code | Name | pos_weight | train pos |
| ---: | --- | --- | ---: | ---: |
| 1 | `265764009` | Renal dialysis (procedure) | 7.8233 | 11299 |
| 2 | `430193006` | Medication Reconciliation (procedure) | 11.9323 | 7709 |
| 3 | `180325003` | Electrical cardioversion | 18.8556 | 5021 |
| 4 | `40701008` | Echocardiography (procedure) | 56.9959 | 1719 |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 94.3107 | 1046 |
| 6 | `18286008` | Catheter ablation of tissue of heart | 73.9023 | 1331 |
| 7 | `444814009` | Viral sinusitis (disorder) | 98.1990 | 1005 |
| 8 | `431182000` | Placing subject in prone position (procedure) | 100.8335 | 979 |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 100.8335 | 979 |
| 10 | `29303009` | Electrocardiographic procedure | 124.7188 | 793 |
| 11 | `73761001` | Colonoscopy | 146.6963 | 675 |
| 12 | `399208008` | Plain chest X-ray (procedure) | 146.0428 | 678 |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 188.1746 | 527 |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 308.6118 | 322 |
| 15 | `230690007` | Stroke | 207.1315 | 479 |
| 16 | `127783003` | Spirometry (procedure) | 235.2441 | 422 |
| 17 | `180256009` | Subcutaneous immunotherapy | 212.0235 | 468 |
| 18 | `15777000` | Prediabetes | 281.4221 | 353 |
| 19 | `423475008` | Heart failure education (procedure) | 227.1350 | 437 |
| 20 | `10509002` | Acute bronchitis (disorder) | 246.3821 | 403 |
| 21 | `271737000` | Anemia (disorder) | 275.1634 | 361 |
| 22 | `410006001` | Digital examination of rectum | 258.6224 | 384 |
| 23 | `71651007` | Mammography (procedure) | 280.6243 | 354 |
| 24 | `312681000` | Bone density scan (procedure) | 257.2772 | 386 |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 276.7019 | 359 |

## Epoch losses

| Epoch | Train weighted BCE | Train unweighted BCE | Val weighted BCE | Val unweighted BCE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.365859 | 0.699685 | 1.375689 | 0.697756 |
| 2 | 1.332677 | 0.696945 | 1.348097 | 0.693708 |
| 3 | 1.304386 | 0.692027 | 1.323355 | 0.687682 |
| 4 | 1.278957 | 0.685864 | 1.296760 | 0.679518 |
| 5 | 1.252225 | 0.677288 | 1.272713 | 0.668985 |
| 6 | 1.227620 | 0.665914 | 1.247397 | 0.655177 |
| 7 | 1.202173 | 0.651402 | 1.220893 | 0.637698 |
| 8 | 1.176150 | 0.633437 | 1.193359 | 0.616743 |
| 9 | 1.149439 | 0.612138 | 1.165591 | 0.593197 |
| 10 | 1.122399 | 0.588381 | 1.138065 | 0.568833 |
| 11 | 1.095192 | 0.563896 | 1.110923 | 0.545972 |
| 12 | 1.067870 | 0.540836 | 1.084236 | 0.526772 |
| 13 | 1.040839 | 0.521178 | 1.058159 | 0.512618 |
| 14 | 1.014672 | 0.506220 | 1.032958 | 0.503756 |
| 15 | 0.989912 | 0.496322 | 1.008880 | 0.498567 |
| 16 | 0.966948 | 0.490206 | 0.986163 | 0.493380 |
| 17 | 0.945996 | 0.484468 | 0.965226 | 0.485006 |
| 18 | 0.927268 | 0.475863 | 0.946313 | 0.474456 |
| 19 | 0.910674 | 0.464986 | 0.929137 | 0.465512 |
| 20 | 0.895806 | 0.455378 | 0.913442 | 0.459504 |
| 21 | 0.882468 | 0.448585 | 0.898707 | 0.455728 |
| 22 | 0.870107 | 0.444205 | 0.884441 | 0.454368 |
| 23 | 0.858157 | 0.442460 | 0.870993 | 0.453496 |
| 24 | 0.846464 | 0.440936 | 0.858891 | 0.452186 |
| 25 | 0.835193 | 0.438909 | 0.847198 | 0.449570 |
| 26 | 0.824026 | 0.435912 | 0.836094 | 0.442588 |
| 27 | 0.813307 | 0.427475 | 0.826090 | 0.442414 |
| 28 | 0.803279 | 0.426175 | 0.819101 | 0.428798 |
| 29 | 0.794317 | 0.411864 | 0.810063 | 0.441595 |
| 30 | 0.787960 | 0.427378 | 0.807300 | 0.422164 |
| 31 | 0.779679 | 0.401923 | 0.799290 | 0.428363 |
| 32 | 0.772555 | 0.411930 | 0.794894 | 0.427708 |
| 33 | 0.765209 | 0.409138 | 0.793895 | 0.411592 |
| 34 | 0.760867 | 0.390050 | 0.788401 | 0.421123 |
| 35 | 0.755004 | 0.402248 | 0.786399 | 0.421101 |
| 36 | 0.749757 | 0.401157 | 0.784737 | 0.402610 |
| 37 | 0.745568 | 0.380019 | 0.779642 | 0.411376 |
| 38 | 0.739836 | 0.390247 | 0.777825 | 0.413973 |
| 39 | 0.735872 | 0.392695 | 0.775843 | 0.399486 |
| 40 | 0.731233 | 0.376107 | 0.773042 | 0.401100 |
| 41 | 0.726751 | 0.377544 | 0.769136 | 0.408875 |
| 42 | 0.722741 | 0.386614 | 0.767228 | 0.400982 |
| 43 | 0.718437 | 0.377570 | 0.766353 | 0.395891 |
| 44 | 0.714689 | 0.371164 | 0.761968 | 0.401631 |
| 45 | 0.710684 | 0.377502 | 0.760514 | 0.399640 |
| 46 | 0.706815 | 0.375244 | 0.761210 | 0.395579 |
| 47 | 0.703340 | 0.369827 | 0.756945 | 0.396474 |
| 48 | 0.699400 | 0.371132 | 0.754049 | 0.394859 |
| 49 | 0.696249 | 0.370428 | 0.755078 | 0.390743 |
| 50 | 0.692807 | 0.364805 | 0.752267 | 0.393071 |

## Train (best checkpoint)

- n examples: `99695`
- unweighted BCE: `0.364805`
- micro-AUPRC: `0.3975967181147149`
- macro-AUPRC: `0.2586194497227014`
- macro-AUROC: `0.9112969771967263`
- micro-F1 @0.5: `0.1289`
- macro-F1 @0.5: `0.1429`
- positive predictions @0.5: `475969` (19.0970% of cells)

| Rank | Code | Name | AUROC | AUPRC | F1@0.5 | P | R | Support |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `265764009` | Renal dialysis (procedure) | 0.9751356877701349 | 0.7565795350139258 | 0.7805 | 0.6870 | 0.9035 | 11299 |
| 2 | `430193006` | Medication Reconciliation (procedure) | 0.8203568151985497 | 0.21614995881394022 | 0.2956 | 0.1813 | 0.7991 | 7709 |
| 3 | `180325003` | Electrical cardioversion | 0.944664804256203 | 0.5105104295749635 | 0.4477 | 0.3013 | 0.8709 | 5021 |
| 4 | `40701008` | Echocardiography (procedure) | 0.8998743883948168 | 0.13591300850190421 | 0.1308 | 0.0708 | 0.8546 | 1719 |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 0.9829535553046321 | 0.655470787244615 | 0.3574 | 0.2240 | 0.8843 | 1046 |
| 6 | `18286008` | Catheter ablation of tissue of heart | 0.9167340156790792 | 0.1202638851562754 | 0.1289 | 0.0697 | 0.8527 | 1331 |
| 7 | `444814009` | Viral sinusitis (disorder) | 0.8610021530809827 | 0.0506881041855136 | 0.0642 | 0.0334 | 0.8299 | 1005 |
| 8 | `431182000` | Placing subject in prone position (procedure) | 0.9464751619165984 | 0.5855366806428258 | 0.1146 | 0.0614 | 0.8560 | 979 |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 0.9451560798569878 | 0.5824440733698028 | 0.1142 | 0.0612 | 0.8437 | 979 |
| 10 | `29303009` | Electrocardiographic procedure | 0.9577447319869774 | 0.669025226288457 | 0.0969 | 0.0512 | 0.8890 | 793 |
| 11 | `73761001` | Colonoscopy | 0.875691457767604 | 0.04302166189149417 | 0.0452 | 0.0232 | 0.8533 | 675 |
| 12 | `399208008` | Plain chest X-ray (procedure) | 0.8768707679677067 | 0.03513045056526012 | 0.0489 | 0.0252 | 0.8392 | 678 |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 0.8648780912218117 | 0.027034741974744602 | 0.0360 | 0.0184 | 0.8672 | 527 |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 0.9389153845543232 | 0.0801315708297317 | 0.0319 | 0.0162 | 0.9348 | 322 |
| 15 | `230690007` | Stroke | 0.7709097466096618 | 0.01466891748516246 | 0.0204 | 0.0103 | 0.7745 | 479 |
| 16 | `127783003` | Spirometry (procedure) | 0.91278028470774 | 0.1006993588400294 | 0.0355 | 0.0182 | 0.7986 | 422 |
| 17 | `180256009` | Subcutaneous immunotherapy | 0.9586664747558457 | 0.514106922636438 | 0.0699 | 0.0365 | 0.8248 | 468 |
| 18 | `15777000` | Prediabetes | 0.9149229550841136 | 0.06236401842646962 | 0.0311 | 0.0159 | 0.8782 | 353 |
| 19 | `423475008` | Heart failure education (procedure) | 0.9952425602086474 | 0.8646996286237331 | 0.5343 | 0.3674 | 0.9794 | 437 |
| 20 | `10509002` | Acute bronchitis (disorder) | 0.8593624874033716 | 0.018967678047007983 | 0.0277 | 0.0141 | 0.8536 | 403 |
| 21 | `271737000` | Anemia (disorder) | 0.9226326419828634 | 0.13572571374405554 | 0.0334 | 0.0170 | 0.8670 | 361 |
| 22 | `410006001` | Digital examination of rectum | 0.8758521080033095 | 0.07552969623816981 | 0.0236 | 0.0120 | 0.7917 | 384 |
| 23 | `71651007` | Mammography (procedure) | 0.9721634071355089 | 0.09627946741687501 | 0.0474 | 0.0243 | 0.9633 | 354 |
| 24 | `312681000` | Bone density scan (procedure) | 0.9106836661016745 | 0.08642927778238872 | 0.0320 | 0.0163 | 0.8782 | 386 |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.8827550029690179 | 0.02811544977374993 | 0.0242 | 0.0123 | 0.8886 | 359 |

## Validation (best checkpoint)

- n examples: `21124`
- unweighted BCE: `0.390743`
- micro-AUPRC: `0.24194972883374016`
- macro-AUPRC: `0.237178408924108`
- macro-AUROC: `0.9016374932203688`
- micro-F1 @0.5: `0.0966`
- macro-F1 @0.5: `0.1178`
- positive predictions @0.5: `104480` (19.7841% of cells)

| Rank | Code | Name | AUROC | AUPRC | F1@0.5 | P | R | Support |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `265764009` | Renal dialysis (procedure) | 0.9687605085968755 | 0.5299912203993944 | 0.5342 | 0.4111 | 0.7624 | 1174 |
| 2 | `430193006` | Medication Reconciliation (procedure) | 0.8040803489673317 | 0.20656604704621037 | 0.2650 | 0.1583 | 0.8135 | 1582 |
| 3 | `180325003` | Electrical cardioversion | 0.9127578861097804 | 0.18993698521793403 | 0.2647 | 0.1609 | 0.7451 | 565 |
| 4 | `40701008` | Echocardiography (procedure) | 0.9006046046857438 | 0.14599769565099785 | 0.1148 | 0.0616 | 0.8466 | 313 |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 0.9698076012758542 | 0.38807632510860945 | 0.3118 | 0.1891 | 0.8886 | 413 |
| 6 | `18286008` | Catheter ablation of tissue of heart | 0.9162088342270468 | 0.057523187655596886 | 0.0811 | 0.0427 | 0.8160 | 163 |
| 7 | `444814009` | Viral sinusitis (disorder) | 0.8726485128230688 | 0.05367654547990611 | 0.0662 | 0.0344 | 0.8779 | 213 |
| 8 | `431182000` | Placing subject in prone position (procedure) | 0.9481337335887728 | 0.5802103675267348 | 0.0965 | 0.0512 | 0.8526 | 190 |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 0.9464238483856532 | 0.5754807184585142 | 0.0988 | 0.0525 | 0.8316 | 190 |
| 10 | `29303009` | Electrocardiographic procedure | 0.9688654791658763 | 0.699546312726374 | 0.0861 | 0.0452 | 0.9252 | 147 |
| 11 | `73761001` | Colonoscopy | 0.867823731376193 | 0.04826143304307061 | 0.0511 | 0.0263 | 0.8679 | 159 |
| 12 | `399208008` | Plain chest X-ray (procedure) | 0.8704965688143347 | 0.038644207386878515 | 0.0469 | 0.0242 | 0.7929 | 140 |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 0.8372259790536796 | 0.027866264855261752 | 0.0388 | 0.0199 | 0.7895 | 133 |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 0.9266994466351849 | 0.07546032665800899 | 0.0360 | 0.0183 | 0.9268 | 82 |
| 15 | `230690007` | Stroke | 0.7447843418717205 | 0.014739163428354358 | 0.0189 | 0.0096 | 0.7476 | 103 |
| 16 | `127783003` | Spirometry (procedure) | 0.9231272985226049 | 0.16198208380380358 | 0.0427 | 0.0219 | 0.8229 | 96 |
| 17 | `180256009` | Subcutaneous immunotherapy | 0.8195071970418333 | 0.5019793823014865 | 0.0594 | 0.0311 | 0.6583 | 120 |
| 18 | `15777000` | Prediabetes | 0.9143033643793955 | 0.06533411321567492 | 0.0327 | 0.0166 | 0.8875 | 80 |
| 19 | `423475008` | Heart failure education (procedure) | 0.9944573523120926 | 0.8330301144544182 | 0.4756 | 0.3144 | 0.9765 | 85 |
| 20 | `10509002` | Acute bronchitis (disorder) | 0.8555184990691231 | 0.02802477293376931 | 0.0311 | 0.0159 | 0.7961 | 103 |
| 21 | `271737000` | Anemia (disorder) | 0.9225021922178908 | 0.1870724596270458 | 0.0374 | 0.0191 | 0.8667 | 90 |
| 22 | `410006001` | Digital examination of rectum | 0.8564515828620033 | 0.08571803260909812 | 0.0291 | 0.0148 | 0.8261 | 115 |
| 23 | `71651007` | Mammography (procedure) | 0.9794925385327596 | 0.25763241900481343 | 0.0719 | 0.0373 | 1.0000 | 140 |
| 24 | `312681000` | Bone density scan (procedure) | 0.9273051459927827 | 0.13994804738736832 | 0.0316 | 0.0161 | 0.9359 | 78 |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.8929507340016153 | 0.03676199712337404 | 0.0230 | 0.0117 | 0.8800 | 75 |

## Test (best checkpoint)

- n examples: `21849`
- unweighted BCE: `0.346839`
- micro-AUPRC: `0.3390088719738635`
- macro-AUPRC: `0.27067037987480197`
- macro-AUROC: `0.9103792169007596`
- micro-F1 @0.5: `0.1172`
- macro-F1 @0.5: `0.1374`
- positive predictions @0.5: `94679` (17.3333% of cells)

| Rank | Code | Name | AUROC | AUPRC | F1@0.5 | P | R | Support |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `265764009` | Renal dialysis (procedure) | 0.9924800011512132 | 0.8437717856964287 | 0.7035 | 0.5557 | 0.9585 | 1035 |
| 2 | `430193006` | Medication Reconciliation (procedure) | 0.7982392178600527 | 0.21082238851478857 | 0.2912 | 0.1795 | 0.7708 | 1885 |
| 3 | `180325003` | Electrical cardioversion | 0.9608717205204171 | 0.5264041395103363 | 0.4774 | 0.3240 | 0.9068 | 1030 |
| 4 | `40701008` | Echocardiography (procedure) | 0.8934710647887972 | 0.11564343595096302 | 0.1209 | 0.0654 | 0.8090 | 356 |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 0.981437138697075 | 0.6609869533233595 | 0.3332 | 0.2043 | 0.9014 | 355 |
| 6 | `18286008` | Catheter ablation of tissue of heart | 0.939970035633301 | 0.12211729943370254 | 0.1230 | 0.0659 | 0.9292 | 240 |
| 7 | `444814009` | Viral sinusitis (disorder) | 0.8527989114380591 | 0.04037341758971558 | 0.0618 | 0.0322 | 0.7665 | 197 |
| 8 | `431182000` | Placing subject in prone position (procedure) | 0.9578098609288206 | 0.5818365385762887 | 0.1056 | 0.0561 | 0.8966 | 174 |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 0.9558215010141988 | 0.5737559488119758 | 0.1057 | 0.0562 | 0.8851 | 174 |
| 10 | `29303009` | Electrocardiographic procedure | 0.9616077741055687 | 0.6637754183547397 | 0.0843 | 0.0442 | 0.8865 | 141 |
| 11 | `73761001` | Colonoscopy | 0.8497875954313516 | 0.031760757565367434 | 0.0500 | 0.0258 | 0.8205 | 156 |
| 12 | `399208008` | Plain chest X-ray (procedure) | 0.8887527376665801 | 0.03362465868610937 | 0.0435 | 0.0223 | 0.8387 | 124 |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 0.8367758231521623 | 0.022926853376457182 | 0.0354 | 0.0181 | 0.7611 | 113 |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 0.9578239576263292 | 0.07294234962440331 | 0.0365 | 0.0186 | 0.9565 | 69 |
| 15 | `230690007` | Stroke | 0.7651585545876911 | 0.017564023881180436 | 0.0255 | 0.0129 | 0.7636 | 110 |
| 16 | `127783003` | Spirometry (procedure) | 0.8885551718102056 | 0.11608629025208657 | 0.0530 | 0.0275 | 0.6870 | 131 |
| 17 | `180256009` | Subcutaneous immunotherapy | 0.9717801031101321 | 0.6579245636294139 | 0.0542 | 0.0280 | 0.8833 | 60 |
| 18 | `15777000` | Prediabetes | 0.8947921609445235 | 0.04239032920129909 | 0.0306 | 0.0156 | 0.7778 | 72 |
| 19 | `423475008` | Heart failure education (procedure) | 0.9974177599229274 | 0.8408974006846077 | 0.5034 | 0.3409 | 0.9615 | 78 |
| 20 | `10509002` | Acute bronchitis (disorder) | 0.847056758096497 | 0.02224671697125173 | 0.0301 | 0.0153 | 0.7978 | 89 |
| 21 | `271737000` | Anemia (disorder) | 0.9226220250100847 | 0.1484866111683669 | 0.0356 | 0.0182 | 0.8243 | 74 |
| 22 | `410006001` | Digital examination of rectum | 0.8932931966848966 | 0.07864233260141743 | 0.0270 | 0.0137 | 0.8667 | 90 |
| 23 | `71651007` | Mammography (procedure) | 0.9864911409687529 | 0.17173650484500017 | 0.0449 | 0.0230 | 1.0000 | 74 |
| 24 | `312681000` | Bone density scan (procedure) | 0.8983918767969055 | 0.14439891169509148 | 0.0342 | 0.0175 | 0.8434 | 83 |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.8662743345724436 | 0.025643866925697804 | 0.0255 | 0.0130 | 0.7816 | 87 |

## Comparison vs Stage 3 vanilla BCE (test)

| Metric | Stage 3 vanilla | Stage 4A weighted | Δ |
| --- | ---: | ---: | ---: |
| BCE | 0.062200 | 0.346839 | 0.284639 |
| micro-AUPRC | 0.157836 | 0.339009 | 0.181172 |
| macro-AUPRC | 0.064279 | 0.270670 | 0.206391 |
| macro-AUROC | 0.654562 | 0.910379 | 0.255818 |
| micro-F1 @0.5 | 0.0000 | 0.1172 | 0.1172 |
| macro-F1 @0.5 | 0.0000 | 0.1374 | 0.1374 |
| # pos preds @0.5 | 0 | 94679 | |

| Rank | Name | S3 AUROC | S4 AUROC | S3 AUPRC | S4 AUPRC | S3 F1 | S4 F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Renal dialysis (procedure) | 0.9566 | 0.9925 | 0.8130 | 0.8438 | 0.0000 | 0.7035 |
| 2 | Medication Reconciliation (procedure) | 0.5762 | 0.7982 | 0.1151 | 0.2108 | 0.0000 | 0.2912 |
| 3 | Electrical cardioversion | 0.7756 | 0.9609 | 0.3146 | 0.5264 | 0.0000 | 0.4774 |
| 4 | Echocardiography (procedure) | 0.6318 | 0.8935 | 0.0305 | 0.1156 | 0.0000 | 0.1209 |
| 5 | Combined chemotherapy and radiation therapy (procedure) | 0.8624 | 0.9814 | 0.1394 | 0.6610 | 0.0000 | 0.3332 |
| 6 | Catheter ablation of tissue of heart | 0.7045 | 0.9400 | 0.0248 | 0.1221 | 0.0000 | 0.1230 |
| 7 | Viral sinusitis (disorder) | 0.5575 | 0.8528 | 0.0101 | 0.0404 | 0.0000 | 0.0618 |
| 8 | Placing subject in prone position (procedure) | 0.6979 | 0.9578 | 0.0165 | 0.5818 | 0.0000 | 0.1056 |
| 9 | Oxygen administration by mask (procedure) | 0.6894 | 0.9558 | 0.0148 | 0.5738 | 0.0000 | 0.1057 |
| 10 | Electrocardiographic procedure | 0.7223 | 0.9616 | 0.0126 | 0.6638 | 0.0000 | 0.0843 |
| 11 | Colonoscopy | 0.5769 | 0.8498 | 0.0089 | 0.0318 | 0.0000 | 0.0500 |
| 12 | Plain chest X-ray (procedure) | 0.5476 | 0.8888 | 0.0067 | 0.0336 | 0.0000 | 0.0435 |
| 13 | Acute viral pharyngitis (disorder) | 0.6232 | 0.8368 | 0.0082 | 0.0229 | 0.0000 | 0.0354 |
| 14 | Body mass index 30+ - obesity (finding) | 0.6005 | 0.9578 | 0.0049 | 0.0729 | 0.0000 | 0.0365 |
| 15 | Stroke | 0.5922 | 0.7652 | 0.0068 | 0.0176 | 0.0000 | 0.0255 |
| 16 | Spirometry (procedure) | 0.4127 | 0.8886 | 0.0051 | 0.1161 | 0.0000 | 0.0530 |
| 17 | Subcutaneous immunotherapy | 0.8200 | 0.9718 | 0.0138 | 0.6579 | 0.0000 | 0.0542 |
| 18 | Prediabetes | 0.5381 | 0.8948 | 0.0039 | 0.0424 | 0.0000 | 0.0306 |
| 19 | Heart failure education (procedure) | 0.8326 | 0.9974 | 0.0110 | 0.8409 | 0.0000 | 0.5034 |
| 20 | Acute bronchitis (disorder) | 0.5840 | 0.8471 | 0.0061 | 0.0222 | 0.0000 | 0.0301 |
| 21 | Anemia (disorder) | 0.6153 | 0.9226 | 0.0151 | 0.1485 | 0.0000 | 0.0356 |
| 22 | Digital examination of rectum | 0.6203 | 0.8933 | 0.0060 | 0.0786 | 0.0000 | 0.0270 |
| 23 | Mammography (procedure) | 0.5716 | 0.9865 | 0.0039 | 0.1717 | 0.0000 | 0.0449 |
| 24 | Bone density scan (procedure) | 0.6616 | 0.8984 | 0.0100 | 0.1444 | 0.0000 | 0.0342 |
| 25 | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.5932 | 0.8663 | 0.0052 | 0.0256 | 0.0000 | 0.0255 |

## Is improvement distributed or dialysis-dominated?

- dialysis test AUPRC: `0.8437717856964287`
- mean AUPRC of other 24 classes: `0.2467911546322342`
- classes with AUPRC > Stage 3: `25/25`
- dialysis AUPRC / macro-AUPRC: `3.12`

AUPRC is **more distributed** than Stage 3: dialysis no longer accounts for nearly all ranking mass.

## Numerical issues

None recorded.

## Comparison vs Stage 3 and Stage 4A (20-epoch)

| Metric | Stage 3 vanilla | 4A 20-epoch | 4A converged | vs S3 | vs 4A-20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| unweighted test BCE | 0.062200 | 0.415957 | 0.346839 | 0.284639 | -0.069118 |
| micro-AUPRC | 0.157836 | 0.150431 | 0.339009 | 0.181172 | 0.188578 |
| macro-AUPRC | 0.064279 | 0.236658 | 0.270670 | 0.206391 | 0.034012 |
| macro-AUROC | 0.654562 | 0.880121 | 0.910379 | 0.255818 | 0.030258 |
| micro-F1 @0.5 | 0.0000 | 0.1029 | 0.1172 | 0.1172 | 0.0143 |
| macro-F1 @0.5 | 0.0000 | 0.1249 | 0.1374 | 0.1374 | 0.0125 |
| mean AUPRC other 24 | — | 0.2112 | 0.2468 | — | 0.0356 |

- classes with test AUPRC > Stage 3: `25/25`
- dialysis AUPRC: `0.8437717856964287`

## Convergence questions

- A. macro-AUPRC still ≥ 0.20? **yes** (`0.27067037987480197`)
- B. most classes still above Stage 3 AUPRC? **yes** (`25/25`)
- C. distributed vs dialysis-only? mean other-24 AUPRC `0.2468` vs dialysis `0.8437717856964287`
