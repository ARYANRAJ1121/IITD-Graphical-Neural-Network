import orjson as json
import random
import base64
from pathlib import Path
import pandas as pd
from collections import Counter
import sys

def main():
    fhir_dir = Path("../fhir")
    bundles = list(fhir_dir.glob("*.json"))
    bundles = [b for b in bundles if b.name not in ["organizations.json", "practitioners.json"]]

    # 1. DOCUMENT/NOTE VALIDATION
    random.seed(42)
    sample_bundles = random.sample(bundles, min(50, len(bundles)))
    doc_refs = []
    for b in sample_bundles:
        with open(b, 'rb') as f:
            data = json.loads(f.read())
        for entry in data.get('entry', []):
            res = entry.get('resource', {})
            if res.get('resourceType') == 'DocumentReference':
                doc_refs.append(res)
                if len(doc_refs) >= 30:
                    break
        if len(doc_refs) >= 30:
            break

    print("=== 1. DOCUMENT/NOTE VALIDATION ===")
    note_texts = []
    for i, doc in enumerate(doc_refs[:30]):
        content = doc.get('content', [{}])[0].get('attachment', {})
        mime = content.get('contentType', 'unknown')
        data_b64 = content.get('data', '')
        try:
            text = base64.b64decode(data_b64).decode('utf-8')
        except:
            text = ""
        
        note_texts.append(text)
        encounter_refs = doc.get('context', {}).get('encounter', [])
        enc_specific = len(encounter_refs) > 0
        note_type = doc.get('type', {}).get('coding', [{}])[0].get('display', 'Unknown')
        
        print(f"Note {i+1}: Type={note_type}, MIME={mime}, Length={len(text)}, Encounter-specific={enc_specific}")
        print(f"Excerpt: {text[:100].replace(chr(10), ' ')}")

    print(f"\nUnique note texts out of {len(note_texts)}: {len(set(note_texts))}")

    # 2. ENCOUNTER VALIDATION
    print("\n=== 2. ENCOUNTER VALIDATION ===")
    enc_stats = pd.read_csv("data/processed/encounter_statistics.csv")
    print("Average resources per encounter across entire dataset:")
    print(f"Condition: {enc_stats['condition_count'].mean():.2f}")
    print(f"Observation: {enc_stats['observation_count'].mean():.2f}")
    print(f"Procedure: {enc_stats['procedure_count'].mean():.2f}")
    print(f"MedicationRequest: {enc_stats['medication_request_count'].mean():.2f}")
    print(f"DiagnosticReport: {enc_stats['diagnostic_report_count'].mean():.2f}")
    print(f"DocumentReference: {enc_stats['document_reference_count'].mean():.2f}")

    # 3. NODE VALIDATION
    print("\n=== 3. NODE VALIDATION ===")
    concepts = Counter()
    missing_codes = 0
    encounters_with_concept_type = {
        'Condition': set(),
        'MedicationRequest': set(),
        'Procedure': set(),
        'Observation': set(),
        'DiagnosticReport': set()
    }

    # Process all bundles for accurate node validation
    for b in bundles:
        with open(b, 'rb') as f:
            data = json.loads(f.read())
        
        for entry in data.get('entry', []):
            res = entry.get('resource', {})
            rt = res.get('resourceType')
            if rt in encounters_with_concept_type:
                # Find encounter ref
                enc_ref = res.get('encounter', {}).get('reference', '')
                if not enc_ref and rt == 'DiagnosticReport':
                    enc_ref = res.get('encounter', {}).get('reference', '')
                
                if enc_ref:
                    enc_id = enc_ref.replace('urn:uuid:', '')
                    encounters_with_concept_type[rt].add(enc_id)

                codeable = None
                if rt == 'MedicationRequest':
                    codeable = res.get('medicationCodeableConcept')
                else:
                    codeable = res.get('code')
                
                if codeable and 'coding' in codeable and codeable['coding']:
                    coding = codeable['coding'][0]
                    sys_str = coding.get('system', '')
                    code_str = coding.get('code', '')
                    display = coding.get('display', '')
                    if not code_str or not display:
                        missing_codes += 1
                    concepts[f"{sys_str}|{code_str}|{display}"] += 1
                else:
                    missing_codes += 1

    print(f"Total candidate node instances: {sum(concepts.values())}")
    print(f"Total unique concepts: {len(concepts)}")
    print(f"Concepts without valid code/display: {missing_codes}")
    for k, v in encounters_with_concept_type.items():
        print(f"Encounters containing {k}: {len(v)}")
    
    print("\nTop 20 concepts:")
    for c, count in concepts.most_common(20):
        print(f"  {count}: {c}")

    # 4. TEMPORAL / LABEL FEASIBILITY
    print("\n=== 4. TEMPORAL / LABEL FEASIBILITY ===")
    patients = pd.read_csv("data/processed/patient_statistics.csv")
    print(f"Total patients: {len(patients)}")
    patients_with_multi = patients[patients['encounter_count'] >= 2]
    print(f"Patients with >=2 encounters: {len(patients_with_multi)}")
    
    # Let's count encounters with conditions (for next-visit phenotype labeling)
    enc_with_cond = enc_stats[enc_stats['condition_count'] > 0]
    print(f"Encounters with >=1 Condition: {len(enc_with_cond)} out of {len(enc_stats)} ({len(enc_with_cond)/len(enc_stats)*100:.2f}%)")

if __name__ == '__main__':
    main()
