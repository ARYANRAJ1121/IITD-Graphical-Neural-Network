# Dataset profile

## 1. Dataset Overview

- Bundles profiled: 1278
- Dataset directory: `C:/Users/Aryan Raj/OneDrive/Desktop/coherent-11-07-2022`
- FHIR directory: `C:/Users/Aryan Raj/OneDrive/Desktop/coherent-11-07-2022/fhir`
- Read-only source dataset retained in place: yes

## 2. Number of Patients

- Unique patients: 1278

## 3. Number of Encounters

- Unique encounters: 143946
- Encounters per patient: min 2, max 2162, mean 112.63, median 65.0

## 4. FHIR Resource Inventory

| resourceType | total_count | bundles_containing | percentage | level |
| --- | ---: | ---: | ---: | --- |
| Observation | 669898 | 1260 | 98.59% | encounter-level |
| Claim | 353347 | 1278 | 100.00% | encounter-level |
| MedicationRequest | 209401 | 1267 | 99.14% | encounter-level |
| DiagnosticReport | 201152 | 1278 | 100.00% | encounter-level |
| DocumentReference | 143946 | 1278 | 100.00% | encounter-level |
| Encounter | 143946 | 1278 | 100.00% | patient-level |
| ExplanationOfBenefit | 143946 | 1278 | 100.00% | encounter-level |
| Procedure | 56092 | 1259 | 98.51% | encounter-level |
| Condition | 15956 | 1278 | 100.00% | encounter-level |
| Immunization | 11900 | 1167 | 91.31% | encounter-level |

## 5. Code Systems

| coding_system | unique_codes | occurrences | resources_using_system | most_frequent_code |
| --- | ---: | ---: | ---: | --- |
| http://loinc.org | 178 | 1648018 | 1013924 | 34117-2 |
| http://snomed.info/sct | 518 | 993609 | 771310 | 162673000 |
| http://terminology.hl7.org/CodeSystem/observation-category | 8 | 669898 | 669898 | laboratory |
| http://terminology.hl7.org/CodeSystem/claim-type | 2 | 497293 | 497293 | institutional |
| https://bluebutton.cms.gov/resources/codesystem/adjudication | 6 | 407952 | 46036 | https://bluebutton.cms.gov/resources/variables/line_coinsrnc_amt |
| http://terminology.hl7.org/CodeSystem/processpriority | 1 | 353347 | 353347 | normal |
| https://bluebutton.cms.gov/resources/variables/line_cms_type_srvc_cd | 1 | 227894 | 143946 | 1 |
| http://terminology.hl7.org/CodeSystem/ex-serviceplace | 3 | 227894 | 143946 | 21 |
| http://www.nlm.nih.gov/research/umls/rxnorm | 135 | 210751 | 210751 | 314076 |
| http://terminology.hl7.org/CodeSystem/v3-ParticipationType | 1 | 143946 | 143946 | PPRF |

## 6. Clinical Concepts

- Encounters with usable structured concepts: 143946 / 143946 (100.00%)
- Node candidate resources: Condition, MedicationRequest, Procedure, Observation, DiagnosticReport
- Primary node code systems: http://loinc.org, http://snomed.info/sct, http://www.nlm.nih.gov/research/umls/rxnorm

## 7. Clinical Notes

- Clinical notes counted from `DocumentReference`: 143946
- Patients with notes: 1278
- Encounters with notes: 143946 (100.00%)
- Preferred note field: `DocumentReference.content[].attachment.data`
- Average note length: 943.13 chars

## 8. Patient/Encounter Relationships

- Patient -> Encounter references resolved: 143946
- Encounter-linked resource counts: {'CarePlan': 6135, 'Condition': 15956, 'DiagnosticReport': 201152, 'DocumentReference': 143946, 'ImagingStudy': 3752, 'Immunization': 11900, 'Media': 1072, 'MedicationRequest': 209401, 'Observation': 669898, 'Procedure': 56092}
- Encounter IDs consistently referenced: True

## 9. Temporal Structure

- Longitudinal encounter ordering available: True
- Core temporal fields:
  - Encounter: `period.start`, `period.end`
  - Observation: `effectiveDateTime`, `issued`
  - Condition: `onsetDateTime`, `recordedDate`
  - Procedure: `performedPeriod`
  - MedicationRequest: `authoredOn`
  - DiagnosticReport: `effectiveDateTime`, `issued`
  - DocumentReference: `date`, `context.period`

## 10. Data Quality

- Malformed JSON files: 0
- Duplicate resource IDs: 0
- Unresolved bundle-local references: 0
- Missing patient references in encounter-level resources: 0
- Missing encounter references in encounter-level resources: 0
- Missing codes in node candidate resources: 0
- Missing note text entries: 0

## 11. MINGLE Mapping

| MINGLE concept | Coherent FHIR resource / field | example actual value | availability | notes |
| --- | --- | --- | --- | --- |
| patient | Patient.id | b8dd1798-beef-094d-1be4-f90ee0e6b7d5 | YES | Unique patient anchor within each bundle. |
| visit | Encounter.id | 73575f5e-0ed3-e612-44f4-1fe2e7b0d0a2 | YES | Best visit/hyperedge anchor. |
| medical concept | Condition.code / MedicationRequest.medicationCodeableConcept / Procedure.code / Observation.code / DiagnosticReport.code | http://snomed.info/sct|698754002|Chronic paralysis due to lesion of spinal cord | YES | Core structured concept sources for node construction. |
| node | Encounter-linked coded concept instances | Observation.code -> http://loinc.org|2339-0|Glucose | YES | Recommended node types are encounter-linked coded clinical resources. |
| hyperedge | Encounter.id with all linked encounter-level resources | 73575f5e-0ed3-e612-44f4-1fe2e7b0d0a2 | YES | Encounter groups structured concepts and note text for a visit. |
| clinical note | DocumentReference.content[].attachment.data | 
1926-06-19

# Chief Complaint
No complaints.

# History of Present Illness
Abe6 | YES | Primary text source, encounter-linked through context.encounter. |
| concept semantic embedding C_v | Not present in raw Coherent; must be derived from codes/displays |  | NO | Requires later hypergraph / embedding embedding generation. |
| clinical note embedding N_e | Not present in raw Coherent; must be derived from note text |  | NO | Requires later hypergraph / embedding embedding generation. |

## 12. MIMIC-III Compatibility

| MIMIC-III/MINGLE requirement | Coherent equivalent | Available? | Evidence | Potential adaptation required |
| --- | --- | --- | --- | --- |
| structured medical concepts | Encounter-linked coded FHIR resources | YES | Condition, MedicationRequest, Procedure, Observation, DiagnosticReport contain codes and encounter references. | Resolve bundle-local UUID references and normalize code sources. |
| concept/code identity | FHIR coding.system + coding.code + coding.display | YES | SNOMED CT, RxNorm, LOINC, CVX, DICOM and HL7 systems were detected. | Decide which coding systems to keep as MINGLE nodes. |
| visit/encounter grouping | Encounter | YES | Encounter is consistently referenced by structured resources and notes. | Use Encounter.id as the hyperedge key. |
| clinical note text | DocumentReference.content[].attachment.data | YES | Every encounter had an encounter-linked DocumentReference note. | Decode base64 note text during preprocessing. |
| patient longitudinal information | One Patient bundle with many Encounters and dated resources | YES | 1278 bundles each contained one longitudinal patient record. | Sort encounters and encounter-linked resources temporally. |
| temporal information | Encounter.period, Observation.effectiveDateTime, MedicationRequest.authoredOn, etc. | YES | Multiple timestamp fields exist across core resources. | Define a canonical visit timestamp strategy. |
| prediction target/label availability | No single direct MINGLE target field in raw FHIR bundles | PARTIAL | No one-step label field was identified during profiling. | Explicitly engineer the downstream task label in Hypergraph construction. |

## 13. Missing Information

- Direct prediction labels are not present as a single precomputed target for MINGLE.
- Binary note resources are absent in the FHIR bundles.
- Some medication concepts require following `MedicationRequest.medicationReference -> Medication`.
- External practitioner, organization, and location references are identifier-based rather than local resource IDs.

## 14. Recommended Adaptations for Hypergraph construction

- Resolve all `urn:uuid` bundle-local references during extraction.
- Use `Encounter.id` as the visit/hyperedge anchor.
- Build node concepts from `Condition`, `MedicationRequest`, `Procedure`, `Observation`, and selected `DiagnosticReport` codes.
- Use decoded `DocumentReference.content[].attachment.data` as the primary note text, with paired `DiagnosticReport.presentedForm` as cross-check only.
- Define downstream labels explicitly because they are not prepackaged in Coherent.
