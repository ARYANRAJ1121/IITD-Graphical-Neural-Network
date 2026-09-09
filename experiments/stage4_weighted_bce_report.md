# Stage 4A: Weighted BCE

Isolated loss-function experiment. Architecture, graph, embeddings, labels, split, and seed are frozen.
Positive-class weights from **training pairs only**. No focal loss, oversampling, or threshold tuning during training.
F1 uses the paper's 0.5 threshold. Checkpoint selected by **unweighted** validation BCE (same rule as Stage 3).

## Setup

- seed: `42`
- split: `[0.7, 0.15, 0.15]`
- epochs: `20`
- optimizer: Adam, lr `0.001`
- d=48, h=4, L=2, 25 classes, PMA + PairNorm + MLP_1/2 + self-loops + JK-CONCAT
- parameter count: `7782633`
- device: `cpu`
- hardware: `AMD64 Family 25 Model 124 Stepping 0, AuthenticAMD | cuda=False cpu`
- runtime_sec: `1581.9`
- best epoch: `20` (lowest unweighted val BCE)
- checkpoint: `data\processed\checkpoints\stage4a_weighted_bce_best.pt`

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

| Epoch | Train weighted BCE | Train unweighted BCE | Val unweighted BCE |
| ---: | ---: | ---: | ---: |
| 1 | 1.365859 | 0.699685 | 0.697756 |
| 2 | 1.332677 | 0.696945 | 0.693708 |
| 3 | 1.304386 | 0.692027 | 0.687682 |
| 4 | 1.278957 | 0.685864 | 0.679518 |
| 5 | 1.252225 | 0.677288 | 0.668983 |
| 6 | 1.227619 | 0.665912 | 0.655175 |
| 7 | 1.202172 | 0.651399 | 0.637695 |
| 8 | 1.176150 | 0.633434 | 0.616740 |
| 9 | 1.149440 | 0.612135 | 0.593197 |
| 10 | 1.122400 | 0.588378 | 0.568833 |
| 11 | 1.095192 | 0.563895 | 0.545976 |
| 12 | 1.067871 | 0.540840 | 0.526783 |
| 13 | 1.040840 | 0.521189 | 0.512636 |
| 14 | 1.014673 | 0.506236 | 0.503773 |
| 15 | 0.989914 | 0.496336 | 0.498576 |
| 16 | 0.966950 | 0.490211 | 0.493373 |
| 17 | 0.945997 | 0.484462 | 0.484994 |
| 18 | 0.927271 | 0.475856 | 0.474454 |
| 19 | 0.910677 | 0.464986 | 0.465501 |
| 20 | 0.895808 | 0.455363 | 0.459502 |

## Train (best checkpoint)

- n examples: `99695`
- unweighted BCE: `0.448579`
- micro-AUPRC: `0.17892991208985357`
- macro-AUPRC: `0.21865082328591626`
- macro-AUROC: `0.8670484029916122`
- micro-F1 @0.5: `0.1034`
- macro-F1 @0.5: `0.1170`
- positive predictions @0.5: `537510` (21.5662% of cells)

| Rank | Code | Name | AUROC | AUPRC | F1@0.5 | P | R | Support |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `265764009` | Renal dialysis (procedure) | 0.9562049810401705 | 0.7569910575316707 | 0.7275 | 0.6762 | 0.7873 | 11299 |
| 2 | `430193006` | Medication Reconciliation (procedure) | 0.7698551289354698 | 0.2414975322575241 | 0.2606 | 0.1623 | 0.6596 | 7709 |
| 3 | `180325003` | Electrical cardioversion | 0.9073147717583907 | 0.3819056836952417 | 0.3565 | 0.2270 | 0.8303 | 5021 |
| 4 | `40701008` | Echocardiography (procedure) | 0.8662197395351726 | 0.12880100834127878 | 0.0981 | 0.0523 | 0.8028 | 1719 |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 0.9401817260559179 | 0.6321871039501633 | 0.3882 | 0.2600 | 0.7658 | 1046 |
| 6 | `18286008` | Catheter ablation of tissue of heart | 0.8586276173922883 | 0.05773508881715621 | 0.0830 | 0.0436 | 0.8760 | 1331 |
| 7 | `444814009` | Viral sinusitis (disorder) | 0.8310410960699592 | 0.03774891072099188 | 0.0596 | 0.0310 | 0.7910 | 1005 |
| 8 | `431182000` | Placing subject in prone position (procedure) | 0.9254975923544729 | 0.432373007461958 | 0.0972 | 0.0517 | 0.8172 | 979 |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 0.9257797339493851 | 0.46539800152363164 | 0.1024 | 0.0547 | 0.8080 | 979 |
| 10 | `29303009` | Electrocardiographic procedure | 0.9424138006305451 | 0.5053535922165996 | 0.0858 | 0.0451 | 0.8663 | 793 |
| 11 | `73761001` | Colonoscopy | 0.8182063032533644 | 0.023958676897363636 | 0.0375 | 0.0192 | 0.7644 | 675 |
| 12 | `399208008` | Plain chest X-ray (procedure) | 0.8450319591436326 | 0.02914660798676506 | 0.0392 | 0.0201 | 0.7876 | 678 |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 0.826937300503376 | 0.022144243225341385 | 0.0299 | 0.0152 | 0.8159 | 527 |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 0.9014391976825129 | 0.07678120651246592 | 0.0234 | 0.0119 | 0.8882 | 322 |
| 15 | `230690007` | Stroke | 0.6984496342767801 | 0.009685421161683938 | 0.0211 | 0.0108 | 0.5219 | 479 |
| 16 | `127783003` | Spirometry (procedure) | 0.8082923636830277 | 0.018376707212129445 | 0.0216 | 0.0109 | 0.8128 | 422 |
| 17 | `180256009` | Subcutaneous immunotherapy | 0.914993325758541 | 0.5389584925049408 | 0.1876 | 0.1087 | 0.6859 | 468 |
| 18 | `15777000` | Prediabetes | 0.8682132397179104 | 0.06961729215928829 | 0.0222 | 0.0112 | 0.8584 | 353 |
| 19 | `423475008` | Heart failure education (procedure) | 0.9931643365857039 | 0.7793582299239609 | 0.1384 | 0.0744 | 0.9863 | 437 |
| 20 | `10509002` | Acute bronchitis (disorder) | 0.8136968046423767 | 0.013714709107673827 | 0.0211 | 0.0107 | 0.7667 | 403 |
| 21 | `271737000` | Anemia (disorder) | 0.8866771256122562 | 0.1512147266550764 | 0.0345 | 0.0176 | 0.8144 | 361 |
| 22 | `410006001` | Digital examination of rectum | 0.8024202903840796 | 0.01286140356286252 | 0.0215 | 0.0109 | 0.6745 | 384 |
| 23 | `71651007` | Mammography (procedure) | 0.889533338258445 | 0.02569650985561627 | 0.0219 | 0.0111 | 0.9011 | 354 |
| 24 | `312681000` | Bone density scan (procedure) | 0.8605813711607309 | 0.04239965214457074 | 0.0255 | 0.0130 | 0.8057 | 386 |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.8254372964057946 | 0.012365716721951562 | 0.0197 | 0.0100 | 0.8217 | 359 |

## Validation (best checkpoint)

- n examples: `21124`
- unweighted BCE: `0.459502`
- micro-AUPRC: `0.1176661130205204`
- macro-AUPRC: `0.1900764241169938`
- macro-AUROC: `0.8722678121682194`
- micro-F1 @0.5: `0.0774`
- macro-F1 @0.5: `0.1007`
- positive predictions @0.5: `111979` (21.2041% of cells)

| Rank | Code | Name | AUROC | AUPRC | F1@0.5 | P | R | Support |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `265764009` | Renal dialysis (procedure) | 0.9094135679915292 | 0.4328498591010384 | 0.4871 | 0.4211 | 0.5775 | 1174 |
| 2 | `430193006` | Medication Reconciliation (procedure) | 0.755496492303329 | 0.24456545110984787 | 0.2498 | 0.1575 | 0.6037 | 1582 |
| 3 | `180325003` | Electrical cardioversion | 0.8503939665120933 | 0.11895457993272769 | 0.1988 | 0.1169 | 0.6637 | 565 |
| 4 | `40701008` | Echocardiography (procedure) | 0.884886310585011 | 0.1450670543905584 | 0.0947 | 0.0502 | 0.8403 | 313 |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 0.9235412326654268 | 0.409963691110292 | 0.4054 | 0.2874 | 0.6877 | 413 |
| 6 | `18286008` | Catheter ablation of tissue of heart | 0.8276198303422395 | 0.02756077540381534 | 0.0510 | 0.0263 | 0.7853 | 163 |
| 7 | `444814009` | Viral sinusitis (disorder) | 0.8487670639910752 | 0.04255500303646018 | 0.0609 | 0.0316 | 0.8357 | 213 |
| 8 | `431182000` | Placing subject in prone position (procedure) | 0.9270667712560278 | 0.41225533948520826 | 0.0893 | 0.0473 | 0.8000 | 190 |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 0.926068646824858 | 0.44785943674336903 | 0.0914 | 0.0485 | 0.8000 | 190 |
| 10 | `29303009` | Electrocardiographic procedure | 0.9586719370972874 | 0.508350138971879 | 0.0784 | 0.0409 | 0.9184 | 147 |
| 11 | `73761001` | Colonoscopy | 0.8134959883723546 | 0.026214188397682348 | 0.0422 | 0.0217 | 0.7736 | 159 |
| 12 | `399208008` | Plain chest X-ray (procedure) | 0.8480190689504928 | 0.029988873920471545 | 0.0393 | 0.0201 | 0.7857 | 140 |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 0.8270372228986071 | 0.02381991557964326 | 0.0356 | 0.0182 | 0.7519 | 133 |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 0.8872524984873459 | 0.05437758840791265 | 0.0250 | 0.0127 | 0.8537 | 82 |
| 15 | `230690007` | Stroke | 0.6667516487211355 | 0.01016653661649567 | 0.0211 | 0.0108 | 0.4854 | 103 |
| 16 | `127783003` | Spirometry (procedure) | 0.8931736850865513 | 0.04679969075578638 | 0.0272 | 0.0138 | 0.9375 | 96 |
| 17 | `180256009` | Subcutaneous immunotherapy | 0.9666019964451216 | 0.5213275644167034 | 0.2054 | 0.1226 | 0.6333 | 120 |
| 18 | `15777000` | Prediabetes | 0.8816482726667934 | 0.08266557454542177 | 0.0226 | 0.0114 | 0.8875 | 80 |
| 19 | `423475008` | Heart failure education (procedure) | 0.9890203907029802 | 0.7668361565334286 | 0.1115 | 0.0592 | 0.9647 | 85 |
| 20 | `10509002` | Acute bronchitis (disorder) | 0.8409232930730851 | 0.02279648625080079 | 0.0275 | 0.0140 | 0.7767 | 103 |
| 21 | `271737000` | Anemia (disorder) | 0.9099368218651285 | 0.23176035070718928 | 0.0362 | 0.0185 | 0.8222 | 90 |
| 22 | `410006001` | Digital examination of rectum | 0.8185916180850029 | 0.019470289582089902 | 0.0325 | 0.0166 | 0.7826 | 115 |
| 23 | `71651007` | Mammography (procedure) | 0.9014255759490223 | 0.04045227995985902 | 0.0380 | 0.0194 | 0.9786 | 140 |
| 24 | `312681000` | Bone density scan (procedure) | 0.8973743716450169 | 0.07060082587128648 | 0.0246 | 0.0125 | 0.8590 | 78 |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.8535170316879662 | 0.014652952094877316 | 0.0226 | 0.0115 | 0.8667 | 75 |

## Test (best checkpoint)

- n examples: `21849`
- unweighted BCE: `0.415957`
- micro-AUPRC: `0.15043074436793055`
- macro-AUPRC: `0.23665811809956`
- macro-AUROC: `0.8801206200018405`
- micro-F1 @0.5: `0.1029`
- macro-F1 @0.5: `0.1249`
- positive predictions @0.5: `99180` (18.1574% of cells)

| Rank | Code | Name | AUROC | AUPRC | F1@0.5 | P | R | Support |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `265764009` | Renal dialysis (procedure) | 0.9932882874728037 | 0.8488636168144681 | 0.7491 | 0.6082 | 0.9749 | 1035 |
| 2 | `430193006` | Medication Reconciliation (procedure) | 0.7786661747112974 | 0.23960270305584289 | 0.2915 | 0.1940 | 0.5857 | 1885 |
| 3 | `180325003` | Electrical cardioversion | 0.9320982933345522 | 0.38522424454526516 | 0.3701 | 0.2342 | 0.8816 | 1030 |
| 4 | `40701008` | Echocardiography (procedure) | 0.872238191478072 | 0.12376523950692374 | 0.1156 | 0.0624 | 0.7837 | 356 |
| 5 | `703423002` | Combined chemotherapy and radiation therapy (procedure) | 0.9494339068747649 | 0.6701764086011379 | 0.4992 | 0.3567 | 0.8310 | 355 |
| 6 | `18286008` | Catheter ablation of tissue of heart | 0.8864657087324727 | 0.055343028378505923 | 0.0792 | 0.0414 | 0.9125 | 240 |
| 7 | `444814009` | Viral sinusitis (disorder) | 0.8302399468847792 | 0.033643660249736765 | 0.0592 | 0.0309 | 0.6904 | 197 |
| 8 | `431182000` | Placing subject in prone position (procedure) | 0.9421901390711794 | 0.42393423865615176 | 0.0949 | 0.0503 | 0.8563 | 174 |
| 9 | `371908008` | Oxygen administration by mask (procedure) | 0.9409728353816171 | 0.4587206083617672 | 0.0963 | 0.0511 | 0.8448 | 174 |
| 10 | `29303009` | Electrocardiographic procedure | 0.9513298362403899 | 0.4943191636938793 | 0.0803 | 0.0420 | 0.9007 | 141 |
| 11 | `73761001` | Colonoscopy | 0.8199754558660658 | 0.02481342809118893 | 0.0461 | 0.0238 | 0.7308 | 156 |
| 12 | `399208008` | Plain chest X-ray (procedure) | 0.8642009725676528 | 0.025333843104854047 | 0.0410 | 0.0211 | 0.7742 | 124 |
| 13 | `195662009` | Acute viral pharyngitis (disorder) | 0.8217263639946454 | 0.020974754589654823 | 0.0331 | 0.0169 | 0.7522 | 113 |
| 14 | `162864005` | Body mass index 30+ - obesity (finding) | 0.9198673161123754 | 0.06391325915241958 | 0.0246 | 0.0125 | 0.8696 | 69 |
| 15 | `230690007` | Stroke | 0.7144754504890666 | 0.014264351989434605 | 0.0255 | 0.0131 | 0.4818 | 110 |
| 16 | `127783003` | Spirometry (procedure) | 0.828736356165674 | 0.02195601011761004 | 0.0356 | 0.0182 | 0.7786 | 131 |
| 17 | `180256009` | Subcutaneous immunotherapy | 0.9865329600562975 | 0.6927524484171526 | 0.1840 | 0.1025 | 0.9000 | 60 |
| 18 | `15777000` | Prediabetes | 0.8483396090676708 | 0.07046187260713992 | 0.0217 | 0.0110 | 0.7917 | 72 |
| 19 | `423475008` | Heart failure education (procedure) | 0.995637574802519 | 0.7672094222945781 | 0.1107 | 0.0586 | 0.9872 | 78 |
| 20 | `10509002` | Acute bronchitis (disorder) | 0.8093331749834766 | 0.01749068733414007 | 0.0243 | 0.0123 | 0.7191 | 89 |
| 21 | `271737000` | Anemia (disorder) | 0.8613138051944021 | 0.2015078196954591 | 0.0340 | 0.0174 | 0.8108 | 74 |
| 22 | `410006001` | Digital examination of rectum | 0.8272924102925482 | 0.013548113122850472 | 0.0302 | 0.0154 | 0.7778 | 90 |
| 23 | `71651007` | Mammography (procedure) | 0.9416120023582709 | 0.05666627722443904 | 0.0269 | 0.0136 | 1.0000 | 74 |
| 24 | `312681000` | Bone density scan (procedure) | 0.8526019911678322 | 0.1764875866822848 | 0.0258 | 0.0131 | 0.6988 | 83 |
| 25 | `433112001` | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.8344467367455873 | 0.015480166202115732 | 0.0242 | 0.0123 | 0.7356 | 87 |

## Comparison vs Stage 3 vanilla BCE (test)

| Metric | Stage 3 vanilla | Stage 4A weighted | Δ |
| --- | ---: | ---: | ---: |
| BCE | 0.062200 | 0.415957 | 0.353757 |
| micro-AUPRC | 0.157836 | 0.150431 | -0.007406 |
| macro-AUPRC | 0.064279 | 0.236658 | 0.172379 |
| macro-AUROC | 0.654562 | 0.880121 | 0.225559 |
| micro-F1 @0.5 | 0.0000 | 0.1029 | 0.1029 |
| macro-F1 @0.5 | 0.0000 | 0.1249 | 0.1249 |
| # pos preds @0.5 | 0 | 99180 | |

| Rank | Name | S3 AUROC | S4 AUROC | S3 AUPRC | S4 AUPRC | S3 F1 | S4 F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Renal dialysis (procedure) | 0.9566 | 0.9933 | 0.8130 | 0.8489 | 0.0000 | 0.7491 |
| 2 | Medication Reconciliation (procedure) | 0.5762 | 0.7787 | 0.1151 | 0.2396 | 0.0000 | 0.2915 |
| 3 | Electrical cardioversion | 0.7756 | 0.9321 | 0.3146 | 0.3852 | 0.0000 | 0.3701 |
| 4 | Echocardiography (procedure) | 0.6318 | 0.8722 | 0.0305 | 0.1238 | 0.0000 | 0.1156 |
| 5 | Combined chemotherapy and radiation therapy (procedure) | 0.8624 | 0.9494 | 0.1394 | 0.6702 | 0.0000 | 0.4992 |
| 6 | Catheter ablation of tissue of heart | 0.7045 | 0.8865 | 0.0248 | 0.0553 | 0.0000 | 0.0792 |
| 7 | Viral sinusitis (disorder) | 0.5575 | 0.8302 | 0.0101 | 0.0336 | 0.0000 | 0.0592 |
| 8 | Placing subject in prone position (procedure) | 0.6979 | 0.9422 | 0.0165 | 0.4239 | 0.0000 | 0.0949 |
| 9 | Oxygen administration by mask (procedure) | 0.6894 | 0.9410 | 0.0148 | 0.4587 | 0.0000 | 0.0963 |
| 10 | Electrocardiographic procedure | 0.7223 | 0.9513 | 0.0126 | 0.4943 | 0.0000 | 0.0803 |
| 11 | Colonoscopy | 0.5769 | 0.8200 | 0.0089 | 0.0248 | 0.0000 | 0.0461 |
| 12 | Plain chest X-ray (procedure) | 0.5476 | 0.8642 | 0.0067 | 0.0253 | 0.0000 | 0.0410 |
| 13 | Acute viral pharyngitis (disorder) | 0.6232 | 0.8217 | 0.0082 | 0.0210 | 0.0000 | 0.0331 |
| 14 | Body mass index 30+ - obesity (finding) | 0.6005 | 0.9199 | 0.0049 | 0.0639 | 0.0000 | 0.0246 |
| 15 | Stroke | 0.5922 | 0.7145 | 0.0068 | 0.0143 | 0.0000 | 0.0255 |
| 16 | Spirometry (procedure) | 0.4127 | 0.8287 | 0.0051 | 0.0220 | 0.0000 | 0.0356 |
| 17 | Subcutaneous immunotherapy | 0.8200 | 0.9865 | 0.0138 | 0.6928 | 0.0000 | 0.1840 |
| 18 | Prediabetes | 0.5381 | 0.8483 | 0.0039 | 0.0705 | 0.0000 | 0.0217 |
| 19 | Heart failure education (procedure) | 0.8326 | 0.9956 | 0.0110 | 0.7672 | 0.0000 | 0.1107 |
| 20 | Acute bronchitis (disorder) | 0.5840 | 0.8093 | 0.0061 | 0.0175 | 0.0000 | 0.0243 |
| 21 | Anemia (disorder) | 0.6153 | 0.8613 | 0.0151 | 0.2015 | 0.0000 | 0.0340 |
| 22 | Digital examination of rectum | 0.6203 | 0.8273 | 0.0060 | 0.0135 | 0.0000 | 0.0302 |
| 23 | Mammography (procedure) | 0.5716 | 0.9416 | 0.0039 | 0.0567 | 0.0000 | 0.0269 |
| 24 | Bone density scan (procedure) | 0.6616 | 0.8526 | 0.0100 | 0.1765 | 0.0000 | 0.0258 |
| 25 | Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance | 0.5932 | 0.8344 | 0.0052 | 0.0155 | 0.0000 | 0.0242 |

## Is improvement distributed or dialysis-dominated?

- dialysis test AUPRC: `0.8488636168144681`
- mean AUPRC of other 24 classes: `0.21114955565310553`
- classes with AUPRC > Stage 3: `25/25`
- dialysis AUPRC / macro-AUPRC: `3.59`

AUPRC is **more distributed** than Stage 3: dialysis no longer accounts for nearly all ranking mass.

## Numerical issues

None recorded.
