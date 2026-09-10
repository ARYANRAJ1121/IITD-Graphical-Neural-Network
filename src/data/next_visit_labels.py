from __future__ import annotations

# Frozen next-visit targets: top-25 SNOMED concepts from experiments/label_feasibility.md
FROZEN_TOP25 = [
    ("265764009", "Renal dialysis (procedure)"),
    ("430193006", "Medication Reconciliation (procedure)"),
    ("180325003", "Electrical cardioversion"),
    ("40701008", "Echocardiography (procedure)"),
    ("703423002", "Combined chemotherapy and radiation therapy (procedure)"),
    ("18286008", "Catheter ablation of tissue of heart"),
    ("444814009", "Viral sinusitis (disorder)"),
    ("431182000", "Placing subject in prone position (procedure)"),
    ("371908008", "Oxygen administration by mask (procedure)"),
    ("29303009", "Electrocardiographic procedure"),
    ("73761001", "Colonoscopy"),
    ("399208008", "Plain chest X-ray (procedure)"),
    ("195662009", "Acute viral pharyngitis (disorder)"),
    ("162864005", "Body mass index 30+ - obesity (finding)"),
    ("230690007", "Stroke"),
    ("127783003", "Spirometry (procedure)"),
    ("180256009", "Subcutaneous immunotherapy"),
    ("15777000", "Prediabetes"),
    ("423475008", "Heart failure education (procedure)"),
    ("10509002", "Acute bronchitis (disorder)"),
    ("271737000", "Anemia (disorder)"),
    ("410006001", "Digital examination of rectum"),
    ("71651007", "Mammography (procedure)"),
    ("312681000", "Bone density scan (procedure)"),
    ("433112001", "Percutaneous mechanical thrombectomy of portal vein using fluoroscopic guidance"),
]

FROZEN_TOP25_CODES = [code for code, _ in FROZEN_TOP25]
