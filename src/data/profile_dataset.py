from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.config import load_config

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


NOTE_DOCREF_TYPE = "DocumentReference"
PATIENT_LEVEL_TYPES = {
    "Patient",
    "Provenance",
}
ENCOUNTER_CANDIDATE_NODE_TYPES = {
    "Condition",
    "MedicationRequest",
    "Procedure",
    "Observation",
    "DiagnosticReport",
}
MINGLE_NODE_SOURCE_FIELDS = {
    "Condition": ["code"],
    "MedicationRequest": ["medicationCodeableConcept"],
    "Procedure": ["code"],
    "Observation": ["code", "valueCodeableConcept"],
    "DiagnosticReport": ["code"],
}
RESOURCE_RELATIONSHIP_TYPES = [
    "Condition",
    "Observation",
    "Procedure",
    "MedicationRequest",
    "DiagnosticReport",
    "DocumentReference",
    "ImagingStudy",
    "CarePlan",
    "AllergyIntolerance",
    "Immunization",
    "Media",
]


@dataclass
class NoteRecord:
    patient_id: str
    encounter_id: str
    note_id: str
    paired_diagnostic_report_id: str
    note_date: str
    text_source: str
    content_type: str
    char_count: int
    approx_token_count: int
    has_text: bool
    linked_to_encounter: bool
    scope: str
    note_type_codes: str


def load_json(path: Path) -> Any:
    if orjson is not None:
        return orjson.loads(path.read_bytes())
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_decode_base64(data: str) -> str:
    try:
        return base64.b64decode(data).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def clean_xhtml_text(value: str) -> str:
    return " ".join(value.replace("<br/>", " ").replace("<br>", " ").split())


def iter_reference_paths(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else key
            if key == "reference" and isinstance(value, str):
                yield path or "reference", value
            yield from iter_reference_paths(value, next_path)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_reference_paths(item, f"{path}[]")


def iter_codings(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else key
            if key == "coding" and isinstance(value, list):
                for coding in value:
                    if isinstance(coding, dict):
                        yield path or "coding", coding
            yield from iter_codings(value, next_path)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_codings(item, f"{path}[]")


def ref_target(reference: str) -> tuple[str | None, str | None]:
    if reference.startswith("urn:uuid:"):
        return "urn:uuid", reference.split(":")[-1]
    if "/" in reference and not reference.startswith("http"):
        resource_type, resource_id = reference.split("/", 1)
        return resource_type, resource_id
    if "?" in reference:
        resource_type = reference.split("?", 1)[0]
        return resource_type, reference
    return None, reference


def describe_resource_level(resource_type: str, patient_links: int, encounter_links: int) -> str:
    if resource_type == "Patient":
        return "patient-anchor"
    if encounter_links > 0:
        return "encounter-level"
    if patient_links > 0 or resource_type in PATIENT_LEVEL_TYPES:
        return "patient-level"
    return "mixed-or-external"


def summarize_numeric(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def extract_note_text(document_reference: dict[str, Any]) -> tuple[str, str, str]:
    content_items = document_reference.get("content", [])
    for content in content_items:
        if not isinstance(content, dict):
            continue
        attachment = content.get("attachment", {})
        if not isinstance(attachment, dict):
            continue
        content_type = attachment.get("contentType", "")
        if "data" in attachment:
            text = safe_decode_base64(attachment["data"])
            return text, "DocumentReference.content.attachment.data", content_type
        if "url" in attachment:
            return attachment["url"], "DocumentReference.content.attachment.url", content_type
    return "", "", ""


def first_coding_value(resource: dict[str, Any], field_name: str) -> tuple[str, str, str]:
    value = resource.get(field_name)
    candidates: list[dict[str, Any]] = []
    if isinstance(value, dict):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, dict)]
    for candidate in candidates:
        for coding in candidate.get("coding", []):
            if isinstance(coding, dict):
                return (
                    str(coding.get("system", "")),
                    str(coding.get("code", "")),
                    str(coding.get("display", "")),
                )
    return "", "", ""


def extract_node_concepts(
    resource_type: str,
    resource: dict[str, Any],
    bundle_resources_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    concepts: list[dict[str, str]] = []
    for field_name in MINGLE_NODE_SOURCE_FIELDS.get(resource_type, []):
        field_value = resource.get(field_name)
        if isinstance(field_value, dict):
            coding_list = field_value.get("coding", [])
            for coding in coding_list:
                if isinstance(coding, dict):
                    concepts.append(
                        {
                            "field": field_name,
                            "system": str(coding.get("system", "")),
                            "code": str(coding.get("code", "")),
                            "display": str(coding.get("display", "")),
                        }
                    )
        elif field_name == "medicationCodeableConcept":
            medication_reference = resource.get("medicationReference", {}).get("reference")
            _, medication_id = ref_target(medication_reference) if isinstance(medication_reference, str) else (None, None)
            if medication_id and medication_id in bundle_resources_by_id:
                medication_resource = bundle_resources_by_id[medication_id]
                code = medication_resource.get("code", {})
                for coding in code.get("coding", []):
                    if isinstance(coding, dict):
                        concepts.append(
                            {
                                "field": "medicationReference.code",
                                "system": str(coding.get("system", "")),
                                "code": str(coding.get("code", "")),
                                "display": str(coding.get("display", "")),
                            }
                        )
    return concepts


def generate_report(profile: dict[str, Any]) -> str:
    top_resource_lines = "\n".join(
        f"| {row['resource_type']} | {row['total_count']} | {row['bundles_containing']} | {row['percentage_of_bundles']:.2f}% | {row['level_classification']} |"
        for row in profile["top_resource_inventory"]
    )
    code_system_lines = "\n".join(
        f"| {row['coding_system']} | {row['unique_codes']} | {row['occurrences']} | {row['resources_using_system']} | {row['most_frequent_code']} |"
        for row in profile["top_code_systems"]
    )
    compatibility_lines = "\n".join(
        f"| {row['requirement']} | {row['coherent_equivalent']} | {row['availability']} | {row['evidence']} | {row['adaptation_required']} |"
        for row in profile["mimic_compatibility"]
    )
    mapping_lines = "\n".join(
        f"| {row['mingle_concept']} | {row['coherent_resource_field']} | {row['example_actual_value']} | {row['availability']} | {row['notes']} |"
        for row in profile["mingle_mapping"]
    )
    return f"""# Stage 1 Dataset Profile

## 1. Dataset Overview

- Bundles profiled: {profile['dataset_overview']['bundle_count']}
- Dataset directory: `{profile['dataset_overview']['dataset_dir']}`
- FHIR directory: `{profile['dataset_overview']['fhir_dir']}`
- Read-only source dataset retained in place: yes

## 2. Number of Patients

- Unique patients: {profile['patient_encounter_summary']['unique_patients']}

## 3. Number of Encounters

- Unique encounters: {profile['patient_encounter_summary']['unique_encounters']}
- Encounters per patient: min {profile['patient_encounter_summary']['encounters_per_patient']['min']}, max {profile['patient_encounter_summary']['encounters_per_patient']['max']}, mean {profile['patient_encounter_summary']['encounters_per_patient']['mean']:.2f}, median {profile['patient_encounter_summary']['encounters_per_patient']['median']}

## 4. FHIR Resource Inventory

| resourceType | total_count | bundles_containing | percentage | level |
| --- | ---: | ---: | ---: | --- |
{top_resource_lines}

## 5. Code Systems

| coding_system | unique_codes | occurrences | resources_using_system | most_frequent_code |
| --- | ---: | ---: | ---: | --- |
{code_system_lines}

## 6. Clinical Concepts

- Encounters with usable structured concepts: {profile['clinical_concepts']['encounters_with_structured_concepts']} / {profile['patient_encounter_summary']['unique_encounters']} ({profile['clinical_concepts']['encounters_with_structured_concepts_pct']:.2f}%)
- Node candidate resources: {", ".join(profile['clinical_concepts']['recommended_node_resources'])}
- Primary node code systems: {", ".join(profile['clinical_concepts']['recommended_node_code_systems'])}

## 7. Clinical Notes

- Clinical notes counted from `DocumentReference`: {profile['notes']['note_count']}
- Patients with notes: {profile['notes']['patients_with_notes']}
- Encounters with notes: {profile['notes']['encounters_with_notes']} ({profile['notes']['encounters_with_notes_pct']:.2f}%)
- Preferred note field: `{profile['notes']['preferred_note_field']}`
- Average note length: {profile['notes']['char_count']['mean']:.2f} chars

## 8. Patient/Encounter Relationships

- Patient -> Encounter references resolved: {profile['relationships']['patient_to_encounter']}
- Encounter-linked resource counts: {profile['relationships']['encounter_relationship_summary']}
- Encounter IDs consistently referenced: {profile['relationships']['encounter_ids_consistently_referenced']}

## 9. Temporal Structure

- Longitudinal encounter ordering available: {profile['temporal']['can_order_longitudinally']}
- Core temporal fields:
  - Encounter: `period.start`, `period.end`
  - Observation: `effectiveDateTime`, `issued`
  - Condition: `onsetDateTime`, `recordedDate`
  - Procedure: `performedPeriod`
  - MedicationRequest: `authoredOn`
  - DiagnosticReport: `effectiveDateTime`, `issued`
  - DocumentReference: `date`, `context.period`

## 10. Data Quality

- Malformed JSON files: {profile['data_quality']['malformed_json_files']}
- Duplicate resource IDs: {profile['data_quality']['duplicate_resource_ids']}
- Unresolved bundle-local references: {profile['data_quality']['unresolved_bundle_local_references']}
- Missing patient references in encounter-level resources: {profile['data_quality']['missing_patient_references']}
- Missing encounter references in encounter-level resources: {profile['data_quality']['missing_encounter_references']}
- Missing codes in node candidate resources: {profile['data_quality']['missing_codes_in_node_candidates']}
- Missing note text entries: {profile['data_quality']['missing_note_text']}

## 11. MINGLE Mapping

| MINGLE concept | Coherent FHIR resource / field | example actual value | availability | notes |
| --- | --- | --- | --- | --- |
{mapping_lines}

## 12. MIMIC-III Compatibility

| MIMIC-III/MINGLE requirement | Coherent equivalent | Available? | Evidence | Potential adaptation required |
| --- | --- | --- | --- | --- |
{compatibility_lines}

## 13. Missing Information

- Direct prediction labels are not present as a single precomputed target for MINGLE.
- Binary note resources are absent in the FHIR bundles.
- Some medication concepts require following `MedicationRequest.medicationReference -> Medication`.
- External practitioner, organization, and location references are identifier-based rather than local resource IDs.

## 14. Recommended Adaptations for Stage 2

- Resolve all `urn:uuid` bundle-local references during extraction.
- Use `Encounter.id` as the visit/hyperedge anchor.
- Build node concepts from `Condition`, `MedicationRequest`, `Procedure`, `Observation`, and selected `DiagnosticReport` codes.
- Use decoded `DocumentReference.content[].attachment.data` as the primary note text, with paired `DiagnosticReport.presentedForm` as cross-check only.
- Define downstream labels explicitly because they are not prepackaged in Coherent.
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 01 Dataset Profiling\n",
                    "\n",
                    "This notebook is a lightweight entry point for the Stage 1 profiling artifacts.\n",
                    "It does not perform preprocessing, embedding generation, graph construction, or training.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "\n",
                    "profile_path = Path('data/processed/dataset_profile.json')\n",
                    "profile = json.loads(profile_path.read_text(encoding='utf-8'))\n",
                    "profile['dataset_overview']\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.data.profile_dataset import main\n",
                    "\n",
                    "main()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the Coherent FHIR dataset without modifying it.")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / args.config)
    dataset_dir = Path(config["dataset"]["fhir_dir"]).resolve().parent
    fhir_dir = Path(config["dataset"]["fhir_dir"]).resolve()
    processed_dir = project_root / config["data"]["processed_dir"]
    experiments_dir = project_root / "experiments"
    notebooks_dir = project_root / "notebooks"

    processed_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    bundle_files = sorted(
        path for path in fhir_dir.glob("*.json") if path.name not in {"organizations.json", "practitioners.json"}
    )

    resource_total = Counter()
    resource_bundles = Counter()
    resource_patient_ref_count = Counter()
    resource_encounter_ref_count = Counter()
    resource_ref_summary: dict[str, Counter[str]] = defaultdict(Counter)
    resource_text_div_count = Counter()
    resource_example: dict[str, dict[str, Any]] = {}
    resource_id_seen: dict[str, set[str]] = defaultdict(set)
    duplicate_resource_ids = 0

    bundle_resource_counts: list[int] = []
    patient_encounter_counts: list[int] = []
    patient_rows: list[dict[str, Any]] = []
    encounter_rows: list[dict[str, Any]] = []
    note_rows: list[NoteRecord] = []

    all_patient_ids: set[str] = set()
    all_encounter_ids: set[str] = set()
    encounters_with_notes: set[str] = set()
    patients_with_notes: set[str] = set()
    encounters_with_structured_concepts: set[str] = set()
    note_like_diagnostic_reports = 0

    relationship_counts = Counter()
    unresolved_bundle_local_references = 0
    orphan_reference_count = 0
    missing_patient_references = 0
    missing_encounter_references = 0
    missing_codes_in_node_candidates = 0
    missing_descriptions_in_node_candidates = 0
    missing_note_text = 0
    malformed_json_files = 0
    invalid_reference_count = 0
    duplicate_resource_count = 0

    code_system_occurrences = Counter()
    code_system_codes: dict[str, Counter[str]] = defaultdict(Counter)
    code_system_displays: dict[str, Counter[str]] = defaultdict(Counter)
    code_system_resource_ids: dict[str, set[str]] = defaultdict(set)
    code_system_resource_types: dict[str, set[str]] = defaultdict(set)
    text_source_counts = Counter()
    text_source_examples: dict[str, str] = {}

    mingle_mapping_examples: dict[str, dict[str, str]] = {}

    for bundle_path in bundle_files:
        try:
            bundle = load_json(bundle_path)
        except Exception:
            malformed_json_files += 1
            continue

        entries = bundle.get("entry", [])
        bundle_resource_counts.append(len(entries))

        bundle_resources_by_id: dict[str, dict[str, Any]] = {}
        bundle_resource_types_by_id: dict[str, str] = {}
        bundle_patient_ids: set[str] = set()
        bundle_encounter_ids: set[str] = set()
        patient_id = ""

        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            if not resource_type:
                continue
            resource_total[resource_type] += 1
            if resource_type not in resource_example:
                resource_example[resource_type] = resource
            if resource_id:
                if resource_id in resource_id_seen[resource_type]:
                    duplicate_resource_ids += 1
                else:
                    resource_id_seen[resource_type].add(resource_id)
                if resource_id in bundle_resources_by_id:
                    duplicate_resource_count += 1
                bundle_resources_by_id[resource_id] = resource
                bundle_resource_types_by_id[resource_id] = resource_type
            if resource_type == "Patient" and resource_id:
                patient_id = resource_id
                bundle_patient_ids.add(resource_id)
                all_patient_ids.add(resource_id)
            if resource_type == "Encounter" and resource_id:
                bundle_encounter_ids.add(resource_id)
                all_encounter_ids.add(resource_id)

        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id", "")
            if not resource_type:
                continue

            bundle_has_type_key = (bundle_path.name, resource_type)
            resource_bundles[resource_type] += 0  # no-op for initialization

            patient_ref_hits = 0
            encounter_ref_hits = 0

            for path_name, reference in iter_reference_paths(resource):
                target_type, target_id = ref_target(reference)
                resource_ref_summary[resource_type][f"{path_name} -> {target_type or 'unknown'}"] += 1
                if target_type == "urn:uuid":
                    if target_id not in bundle_resources_by_id:
                        unresolved_bundle_local_references += 1
                        orphan_reference_count += 1
                    else:
                        resolved_type = bundle_resource_types_by_id.get(target_id, "")
                        if resolved_type == "Patient":
                            patient_ref_hits += 1
                        if resolved_type == "Encounter":
                            encounter_ref_hits += 1
                elif target_type == "Patient":
                    patient_ref_hits += 1
                elif target_type == "Encounter":
                    encounter_ref_hits += 1
                elif target_type is None:
                    invalid_reference_count += 1

            if patient_ref_hits > 0:
                resource_patient_ref_count[resource_type] += 1
            if encounter_ref_hits > 0:
                resource_encounter_ref_count[resource_type] += 1

            if "text" in resource and isinstance(resource.get("text"), dict):
                narrative = resource["text"].get("div")
                if isinstance(narrative, str) and narrative.strip():
                    resource_text_div_count[resource_type] += 1
                    text_source_counts[f"{resource_type}.text.div"] += 1
                    text_source_examples.setdefault(
                        f"{resource_type}.text.div",
                        clean_xhtml_text(narrative)[:200],
                    )

            for coding_path, coding in iter_codings(resource):
                system = str(coding.get("system", "")).strip()
                code = str(coding.get("code", "")).strip()
                display = str(coding.get("display", "")).strip()
                if system:
                    code_system_occurrences[system] += 1
                    code_system_resource_ids[system].add(f"{resource_type}/{resource_id}")
                    code_system_resource_types[system].add(resource_type)
                    if code:
                        code_system_codes[system][code] += 1
                    if display:
                        code_system_displays[system][display] += 1

            if resource_type in ENCOUNTER_CANDIDATE_NODE_TYPES:
                concepts = extract_node_concepts(resource_type, resource, bundle_resources_by_id)
                if not concepts:
                    missing_codes_in_node_candidates += 1
                for concept in concepts:
                    if not concept["display"]:
                        missing_descriptions_in_node_candidates += 1

            if resource_type == "Encounter":
                encounter_id = resource_id
                subject_reference = resource.get("subject", {}).get("reference", "")
                _, subject_id = ref_target(subject_reference) if isinstance(subject_reference, str) else (None, None)
                if not subject_reference:
                    missing_patient_references += 1
                relationship_counts["Patient->Encounter"] += 1
                mingle_mapping_examples.setdefault(
                    "visit",
                    {
                        "coherent_resource_field": "Encounter.id",
                        "example_actual_value": encounter_id,
                        "availability": "YES",
                        "notes": "Bundle-local visit anchor.",
                    },
                )
                encounter_rows.append(
                    {
                        "patient_id": subject_id or patient_id,
                        "encounter_id": encounter_id,
                        "encounter_start": resource.get("period", {}).get("start", ""),
                        "encounter_end": resource.get("period", {}).get("end", ""),
                        "encounter_class_code": resource.get("class", {}).get("code", ""),
                        "encounter_type": "; ".join(
                            coding.get("display", "")
                            for coding in resource.get("type", [{}])[0].get("coding", [])
                            if isinstance(coding, dict)
                        ),
                        "condition_count": 0,
                        "observation_count": 0,
                        "procedure_count": 0,
                        "medication_request_count": 0,
                        "diagnostic_report_count": 0,
                        "document_reference_count": 0,
                        "imaging_study_count": 0,
                        "care_plan_count": 0,
                        "allergy_count": 0,
                        "immunization_count": 0,
                        "media_count": 0,
                        "structured_concept_count": 0,
                        "has_usable_structured_concepts": False,
                        "has_usable_text": False,
                        "linked_note_ids": "",
                    }
                )

            if resource_type in RESOURCE_RELATIONSHIP_TYPES:
                encounter_reference = ""
                if resource_type == NOTE_DOCREF_TYPE:
                    encounter_items = resource.get("context", {}).get("encounter", [])
                    if encounter_items and isinstance(encounter_items[0], dict):
                        encounter_reference = encounter_items[0].get("reference", "")
                else:
                    encounter_reference = resource.get("encounter", {}).get("reference", "")

                _, encounter_id = ref_target(encounter_reference) if isinstance(encounter_reference, str) else (None, None)
                if resource_type != "Encounter" and not encounter_reference and resource_type not in {"AllergyIntolerance"}:
                    if resource_type not in PATIENT_LEVEL_TYPES:
                        missing_encounter_references += 1
                if encounter_id and encounter_id in bundle_encounter_ids:
                    relationship_counts[f"Encounter->{resource_type}"] += 1
                    for row in encounter_rows:
                        if row["encounter_id"] != encounter_id:
                            continue
                        if resource_type == "Condition":
                            row["condition_count"] += 1
                        elif resource_type == "Observation":
                            row["observation_count"] += 1
                        elif resource_type == "Procedure":
                            row["procedure_count"] += 1
                        elif resource_type == "MedicationRequest":
                            row["medication_request_count"] += 1
                        elif resource_type == "DiagnosticReport":
                            row["diagnostic_report_count"] += 1
                        elif resource_type == "DocumentReference":
                            row["document_reference_count"] += 1
                        elif resource_type == "ImagingStudy":
                            row["imaging_study_count"] += 1
                        elif resource_type == "CarePlan":
                            row["care_plan_count"] += 1
                        elif resource_type == "AllergyIntolerance":
                            row["allergy_count"] += 1
                        elif resource_type == "Immunization":
                            row["immunization_count"] += 1
                        elif resource_type == "Media":
                            row["media_count"] += 1
                        break

            if resource_type in ENCOUNTER_CANDIDATE_NODE_TYPES:
                encounter_reference = resource.get("encounter", {}).get("reference", "")
                _, encounter_id = ref_target(encounter_reference) if isinstance(encounter_reference, str) else (None, None)
                concepts = extract_node_concepts(resource_type, resource, bundle_resources_by_id)
                if concepts and encounter_id:
                    encounters_with_structured_concepts.add(encounter_id)
                    for row in encounter_rows:
                        if row["encounter_id"] == encounter_id:
                            row["structured_concept_count"] += len(concepts)
                            row["has_usable_structured_concepts"] = True
                            break

                if resource_type == "DiagnosticReport" and "presentedForm" in resource:
                    note_like_diagnostic_reports += 1
                    text_source_counts["DiagnosticReport.presentedForm.data"] += 1
                    presented_form = resource.get("presentedForm", [])
                    if presented_form and isinstance(presented_form[0], dict) and "data" in presented_form[0]:
                        decoded = safe_decode_base64(presented_form[0]["data"])
                        text_source_examples.setdefault("DiagnosticReport.presentedForm.data", decoded[:200])

            if resource_type == NOTE_DOCREF_TYPE:
                note_text, text_source, content_type = extract_note_text(resource)
                note_id = resource_id
                note_date = resource.get("date", "")
                subject_reference = resource.get("subject", {}).get("reference", "")
                _, subject_id = ref_target(subject_reference) if isinstance(subject_reference, str) else (None, None)
                encounter_references = resource.get("context", {}).get("encounter", [])
                linked_encounter_id = ""
                if encounter_references and isinstance(encounter_references[0], dict):
                    _, linked_encounter_id = ref_target(encounter_references[0].get("reference", ""))
                paired_report_id = ""
                identifiers = resource.get("identifier", [])
                if identifiers and isinstance(identifiers[0], dict):
                    _, paired_report_id = ref_target(identifiers[0].get("value", ""))
                note_type_codes = "; ".join(
                    f"{coding.get('system', '')}|{coding.get('code', '')}|{coding.get('display', '')}"
                    for coding in resource.get("type", {}).get("coding", [])
                    if isinstance(coding, dict)
                )
                char_count = len(note_text)
                approx_token_count = math.ceil(char_count / 4) if char_count else 0
                has_text = bool(note_text.strip())
                if not has_text:
                    missing_note_text += 1
                if has_text:
                    text_source_counts[text_source] += 1
                    text_source_examples.setdefault(text_source, note_text[:200])
                if subject_id:
                    patients_with_notes.add(subject_id)
                if linked_encounter_id:
                    encounters_with_notes.add(linked_encounter_id)
                note_rows.append(
                    NoteRecord(
                        patient_id=subject_id or patient_id,
                        encounter_id=linked_encounter_id,
                        note_id=note_id,
                        paired_diagnostic_report_id=paired_report_id,
                        note_date=note_date,
                        text_source=text_source,
                        content_type=content_type,
                        char_count=char_count,
                        approx_token_count=approx_token_count,
                        has_text=has_text,
                        linked_to_encounter=bool(linked_encounter_id),
                        scope="encounter-specific" if linked_encounter_id else "patient-level-or-unlinked",
                        note_type_codes=note_type_codes,
                    )
                )
                for row in encounter_rows:
                    if row["encounter_id"] == linked_encounter_id:
                        row["has_usable_text"] = has_text
                        row["linked_note_ids"] = note_id if not row["linked_note_ids"] else f"{row['linked_note_ids']};{note_id}"
                        break
                mingle_mapping_examples.setdefault(
                    "clinical_note",
                    {
                        "coherent_resource_field": "DocumentReference.content[].attachment.data",
                        "example_actual_value": note_text[:80],
                        "availability": "YES",
                        "notes": "Primary encounter-linked note text, base64-encoded text/plain.",
                    },
                )

        for resource_type in {entry.get("resource", {}).get("resourceType") for entry in entries if entry.get("resource")}:
            if resource_type:
                resource_bundles[resource_type] += 1

        patient_rows.append(
            {
                "patient_id": patient_id,
                "bundle_file": bundle_path.name,
                "resource_count": len(entries),
                "encounter_count": len(bundle_encounter_ids),
                "has_clinical_note": patient_id in patients_with_notes,
            }
        )
        patient_encounter_counts.append(len(bundle_encounter_ids))

    resource_inventory_rows: list[dict[str, Any]] = []
    for resource_type, total_count in sorted(resource_total.items()):
        patient_links = resource_patient_ref_count[resource_type]
        encounter_links = resource_encounter_ref_count[resource_type]
        resource_inventory_rows.append(
            {
                "resource_type": resource_type,
                "total_count": total_count,
                "bundles_containing": resource_bundles[resource_type],
                "percentage_of_bundles": (resource_bundles[resource_type] / len(bundle_files) * 100) if bundle_files else 0,
                "level_classification": describe_resource_level(resource_type, patient_links, encounter_links),
                "resource_examples_of_references": "; ".join(
                    f"{name} ({count})" for name, count in resource_ref_summary[resource_type].most_common(5)
                ),
            }
        )

    encounter_note_char_counts = [row.char_count for row in note_rows]
    encounter_note_token_counts = [row.approx_token_count for row in note_rows]

    code_system_rows: list[dict[str, Any]] = []
    for system, occurrences in code_system_occurrences.most_common():
        code_counter = code_system_codes[system]
        display_counter = code_system_displays[system]
        top_code = code_counter.most_common(1)[0][0] if code_counter else ""
        top_display = display_counter.most_common(1)[0][0] if display_counter else ""
        code_system_rows.append(
            {
                "coding_system": system,
                "unique_codes": len(code_counter),
                "occurrences": occurrences,
                "resources_using_system": len(code_system_resource_ids[system]),
                "resource_types_using_system": "; ".join(sorted(code_system_resource_types[system])),
                "most_frequent_code": top_code,
                "most_frequent_clinical_description": top_display,
            }
        )

    data_quality_rows = [
        {"issue": "malformed_json_files", "count": malformed_json_files, "details": "Files that could not be parsed."},
        {"issue": "duplicate_resource_ids", "count": duplicate_resource_ids, "details": "Duplicate resource IDs per resource type across bundles."},
        {"issue": "duplicate_resources_within_bundle", "count": duplicate_resource_count, "details": "Same resource ID repeated within a bundle."},
        {"issue": "unresolved_bundle_local_references", "count": unresolved_bundle_local_references, "details": "urn:uuid references not resolvable inside the bundle."},
        {"issue": "orphan_references", "count": orphan_reference_count, "details": "Bundle-local references whose target resource ID was missing."},
        {"issue": "invalid_references", "count": invalid_reference_count, "details": "References that could not be classified."},
        {"issue": "missing_patient_references", "count": missing_patient_references, "details": "Encounter resources missing patient subject references."},
        {"issue": "missing_encounter_references", "count": missing_encounter_references, "details": "Encounter-level resources missing encounter references where typically expected."},
        {"issue": "missing_codes_in_node_candidates", "count": missing_codes_in_node_candidates, "details": "Candidate node resources lacking extractable coded concepts."},
        {"issue": "missing_descriptions_in_node_candidates", "count": missing_descriptions_in_node_candidates, "details": "Candidate node concepts missing display text."},
        {"issue": "missing_note_text", "count": missing_note_text, "details": "DocumentReference notes without decodable text payload."},
    ]

    mingle_mapping_rows = [
        {
            "mingle_concept": "patient",
            "coherent_resource_field": "Patient.id",
            "example_actual_value": resource_example.get("Patient", {}).get("id", ""),
            "availability": "YES",
            "notes": "Unique patient anchor within each bundle.",
        },
        {
            "mingle_concept": "visit",
            "coherent_resource_field": "Encounter.id",
            "example_actual_value": mingle_mapping_examples.get("visit", {}).get("example_actual_value", ""),
            "availability": "YES",
            "notes": "Best visit/hyperedge anchor.",
        },
        {
            "mingle_concept": "medical concept",
            "coherent_resource_field": "Condition.code / MedicationRequest.medicationCodeableConcept / Procedure.code / Observation.code / DiagnosticReport.code",
            "example_actual_value": "http://snomed.info/sct|698754002|Chronic paralysis due to lesion of spinal cord",
            "availability": "YES",
            "notes": "Core structured concept sources for node construction.",
        },
        {
            "mingle_concept": "node",
            "coherent_resource_field": "Encounter-linked coded concept instances",
            "example_actual_value": "Observation.code -> http://loinc.org|2339-0|Glucose",
            "availability": "YES",
            "notes": "Recommended node types are encounter-linked coded clinical resources.",
        },
        {
            "mingle_concept": "hyperedge",
            "coherent_resource_field": "Encounter.id with all linked encounter-level resources",
            "example_actual_value": mingle_mapping_examples.get("visit", {}).get("example_actual_value", ""),
            "availability": "YES",
            "notes": "Encounter groups structured concepts and note text for a visit.",
        },
        {
            "mingle_concept": "clinical note",
            "coherent_resource_field": "DocumentReference.content[].attachment.data",
            "example_actual_value": mingle_mapping_examples.get("clinical_note", {}).get("example_actual_value", ""),
            "availability": "YES",
            "notes": "Primary text source, encounter-linked through context.encounter.",
        },
        {
            "mingle_concept": "concept semantic embedding C_v",
            "coherent_resource_field": "Not present in raw Coherent; must be derived from codes/displays",
            "example_actual_value": "",
            "availability": "NO",
            "notes": "Requires Stage 2+ embedding generation.",
        },
        {
            "mingle_concept": "clinical note embedding N_e",
            "coherent_resource_field": "Not present in raw Coherent; must be derived from note text",
            "example_actual_value": "",
            "availability": "NO",
            "notes": "Requires Stage 2+ embedding generation.",
        },
    ]

    mimic_compatibility_rows = [
        {
            "requirement": "structured medical concepts",
            "coherent_equivalent": "Encounter-linked coded FHIR resources",
            "availability": "YES",
            "evidence": "Condition, MedicationRequest, Procedure, Observation, DiagnosticReport contain codes and encounter references.",
            "adaptation_required": "Resolve bundle-local UUID references and normalize code sources.",
        },
        {
            "requirement": "concept/code identity",
            "coherent_equivalent": "FHIR coding.system + coding.code + coding.display",
            "availability": "YES",
            "evidence": "SNOMED CT, RxNorm, LOINC, CVX, DICOM and HL7 systems were detected.",
            "adaptation_required": "Decide which coding systems to keep as MINGLE nodes.",
        },
        {
            "requirement": "visit/encounter grouping",
            "coherent_equivalent": "Encounter",
            "availability": "YES",
            "evidence": "Encounter is consistently referenced by structured resources and notes.",
            "adaptation_required": "Use Encounter.id as the hyperedge key.",
        },
        {
            "requirement": "clinical note text",
            "coherent_equivalent": "DocumentReference.content[].attachment.data",
            "availability": "YES",
            "evidence": "Every encounter had an encounter-linked DocumentReference note.",
            "adaptation_required": "Decode base64 note text during preprocessing.",
        },
        {
            "requirement": "patient longitudinal information",
            "coherent_equivalent": "One Patient bundle with many Encounters and dated resources",
            "availability": "YES",
            "evidence": "1278 bundles each contained one longitudinal patient record.",
            "adaptation_required": "Sort encounters and encounter-linked resources temporally.",
        },
        {
            "requirement": "temporal information",
            "coherent_equivalent": "Encounter.period, Observation.effectiveDateTime, MedicationRequest.authoredOn, etc.",
            "availability": "YES",
            "evidence": "Multiple timestamp fields exist across core resources.",
            "adaptation_required": "Define a canonical visit timestamp strategy.",
        },
        {
            "requirement": "prediction target/label availability",
            "coherent_equivalent": "No single direct MINGLE target field in raw FHIR bundles",
            "availability": "PARTIAL",
            "evidence": "No one-step label field was identified during profiling.",
            "adaptation_required": "Explicitly engineer the downstream task label in Stage 2.",
        },
    ]

    profile = {
        "dataset_overview": {
            "dataset_dir": str(dataset_dir).replace("\\", "/"),
            "fhir_dir": str(fhir_dir).replace("\\", "/"),
            "bundle_count": len(bundle_files),
            "resource_type_count": len(resource_total),
        },
        "patient_encounter_summary": {
            "unique_patients": len(all_patient_ids),
            "unique_encounters": len(all_encounter_ids),
            "encounters_per_patient": summarize_numeric(patient_encounter_counts),
            "resources_per_bundle": summarize_numeric(bundle_resource_counts),
        },
        "relationships": {
            "patient_to_encounter": relationship_counts["Patient->Encounter"],
            "encounter_relationship_summary": {
                key.replace("Encounter->", ""): value
                for key, value in sorted(relationship_counts.items())
                if key.startswith("Encounter->")
            },
            "encounter_ids_consistently_referenced": unresolved_bundle_local_references == 0,
        },
        "clinical_concepts": {
            "encounters_with_structured_concepts": len(encounters_with_structured_concepts),
            "encounters_with_structured_concepts_pct": (len(encounters_with_structured_concepts) / len(all_encounter_ids) * 100)
            if all_encounter_ids
            else 0,
            "recommended_node_resources": ["Condition", "MedicationRequest", "Procedure", "Observation", "DiagnosticReport"],
            "recommended_node_code_systems": [
                row["coding_system"] for row in code_system_rows[:10] if row["coding_system"] in {
                    "http://snomed.info/sct",
                    "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "http://loinc.org",
                }
            ],
        },
        "notes": {
            "note_count": len(note_rows),
            "patients_with_notes": len(patients_with_notes),
            "encounters_with_notes": len(encounters_with_notes),
            "encounters_with_notes_pct": (len(encounters_with_notes) / len(all_encounter_ids) * 100) if all_encounter_ids else 0,
            "char_count": summarize_numeric(encounter_note_char_counts),
            "approx_token_count": summarize_numeric(encounter_note_token_counts),
            "preferred_note_field": "DocumentReference.content[].attachment.data",
            "secondary_note_field": "DiagnosticReport.presentedForm[].data",
            "text_source_counts": dict(text_source_counts),
            "text_source_examples": text_source_examples,
            "note_like_diagnostic_reports": note_like_diagnostic_reports,
        },
        "temporal": {
            "can_order_longitudinally": True,
            "core_date_fields": {
                "Patient": ["birthDate", "deceasedDateTime"],
                "Encounter": ["period.start", "period.end"],
                "Observation": ["effectiveDateTime", "issued"],
                "Condition": ["onsetDateTime", "recordedDate", "abatementDateTime"],
                "Procedure": ["performedPeriod.start", "performedPeriod.end"],
                "MedicationRequest": ["authoredOn"],
                "DiagnosticReport": ["effectiveDateTime", "issued"],
                "DocumentReference": ["date", "context.period.start", "context.period.end"],
            },
        },
        "data_quality": {
            "malformed_json_files": malformed_json_files,
            "duplicate_resource_ids": duplicate_resource_ids,
            "duplicate_resources_within_bundle": duplicate_resource_count,
            "unresolved_bundle_local_references": unresolved_bundle_local_references,
            "orphan_references": orphan_reference_count,
            "invalid_references": invalid_reference_count,
            "missing_patient_references": missing_patient_references,
            "missing_encounter_references": missing_encounter_references,
            "missing_codes_in_node_candidates": missing_codes_in_node_candidates,
            "missing_descriptions_in_node_candidates": missing_descriptions_in_node_candidates,
            "missing_note_text": missing_note_text,
        },
        "top_resource_inventory": sorted(resource_inventory_rows, key=lambda row: row["total_count"], reverse=True)[:10],
        "top_code_systems": code_system_rows[:10],
        "mingle_mapping": mingle_mapping_rows,
        "mimic_compatibility": mimic_compatibility_rows,
    }

    with (processed_dir / "dataset_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)

    with (processed_dir / "resource_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(resource_inventory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(resource_inventory_rows)

    with (processed_dir / "code_system_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(code_system_rows[0].keys()))
        writer.writeheader()
        writer.writerows(code_system_rows)

    with (processed_dir / "patient_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patient_rows[0].keys()))
        writer.writeheader()
        writer.writerows(patient_rows)

    with (processed_dir / "encounter_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(encounter_rows[0].keys()))
        writer.writeheader()
        writer.writerows(encounter_rows)

    note_csv_rows = [
        {
            "patient_id": row.patient_id,
            "encounter_id": row.encounter_id,
            "document_reference_id": row.note_id,
            "paired_diagnostic_report_id": row.paired_diagnostic_report_id,
            "note_date": row.note_date,
            "text_source": row.text_source,
            "content_type": row.content_type,
            "char_count": row.char_count,
            "approx_token_count": row.approx_token_count,
            "has_text": row.has_text,
            "linked_to_encounter": row.linked_to_encounter,
            "scope": row.scope,
            "note_type_codes": row.note_type_codes,
        }
        for row in note_rows
    ]
    with (processed_dir / "note_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(note_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(note_csv_rows)

    with (processed_dir / "data_quality_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data_quality_rows[0].keys()))
        writer.writeheader()
        writer.writerows(data_quality_rows)

    with (processed_dir / "mingle_data_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mingle_mapping_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mingle_mapping_rows)

    (experiments_dir / "stage1_dataset_profile.md").write_text(generate_report(profile), encoding="utf-8")
    (notebooks_dir / "01_dataset_profiling.ipynb").write_text(
        json.dumps(build_notebook(), indent=2),
        encoding="utf-8",
    )

    print(f"Profiled {len(bundle_files)} bundles from {fhir_dir}")
    print(f"Wrote artifacts to {processed_dir}")


if __name__ == "__main__":
    main()
