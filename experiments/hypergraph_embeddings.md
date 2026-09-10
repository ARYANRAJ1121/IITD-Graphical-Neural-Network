# Hypergraph construction and semantic embeddings

## 1. Implementation Summary
Successfully replicated the MINGLE methodology for constructing the hypergraph and computing semantic embeddings on the Coherent synthetic dataset.
- **Nodes**: Extracted from unique clinical concepts (`Condition`, `MedicationRequest`, `Procedure`, `Observation`), strictly excluding generic `DiagnosticReport` nodes identified in Dataset profile.
- **Hyperedges (Visits)**: Extracted from `Encounter` IDs and chronologically sorted per patient.
- **Structural Embeddings ($s_v$)**: Calculated using `networkx` and `gensim` (DeepWalk/Word2Vec) based on the co-occurrence graph of nodes within encounters.
- **Semantic Embeddings ($C_v$, $N_e$)**: Computed via `sentence-transformers` using the selected model. Notes were encoded in batches of 5,000 to manage memory constraints.
- **Aggregation**: The final node representations ($X_v$) were constructed by concatenating structural and semantic embeddings, exactly replicating the original paper's fusion strategy.

## 2. Files Created
All files are saved in `data/processed/`:
- `concept_nodes.csv`: The list of 564 unique medical concept nodes.
- `encounter_hyperedges.csv`: The list of 143,946 chronologically-sorted encounters.
- `node_encounter_edges.csv`: 923,650 mapping edges linking nodes to hyperedges.
- `clinical_notes.csv`: The Base64-decoded raw textual notes per encounter.
- `node_embeddings.npy`: The full concatenated node feature matrix $X_v$.
- `note_embeddings.npy`: The full sequence of clinical note embeddings $N_e$.
- `embedding_metadata.json`: Dimensionality and model hyperparameters.
- `hypergraph_statistics.json`: Graph structure metrics.

## 3. Embedding Matrix Dimensions
| Matrix | Description | Dimensions (Rows x Cols) |
| --- | --- | --- |
| **$s_v$** | DeepWalk Structural Embeddings | `(564, 64)` |
| **$C_v$** | Concept Semantic Embeddings | `(564, 768)` |
| **$X_v$** | Fused Node Embeddings ($s_v \oplus C_v$) | `(564, 832)` |
| **$N_e$** | Clinical Note Semantic Embeddings | `(143946, 768)` |

## 4. Hypergraph Structure Metrics
- **4. Number of unique nodes**: 564
- **5. Number of hyperedges**: 143,946
- **6. Node-hyperedge connections**: 923,650

## 7. DeepWalk Dimensions
- `d1` (Structural dimensionality): **64**
- Number of walks: **10**, Walk length: **10**
- Window size: **5**, Epochs: **5**

## 8. BiomedBERT Dimensions
- Model: `NeuML/biomedbert-base-embeddings`
- `d2` (Semantic dimensionality): **768**
- Max sequence length: **512 tokens**
- Pooling method: **Mean pooling**

## 9. Runtime Performance
- **Model Load Time**: ~2.5 seconds
- **DeepWalk Graph Construction & Training**: ~45 seconds
- **Concept Semantic Embedding ($C_v$)**: ~10 seconds
- **Clinical Note Semantic Embedding ($N_e$)**: ~5 hours 25 minutes (Processed locally on CPU with batching & checkpointing)

## 10. Validation Results
- **Node $\leftrightarrow$ Hyperedge Consistency**: Passed (923,650 edges perfectly align with ID sets)
- **Orphan Nodes**: **0** (All 564 concepts belong to at least one encounter)
- **Matrix Shape Alignment**: Passed (Row counts perfectly match valid node/encounter counts)
- **NaN/Inf Check**: Passed (No corrupt or exploding gradients in `npy` tensors)
- **Temporal Ordering**: Passed (Encounters strictly sorted by `patient_id` $\rightarrow$ `start_date`)

## 11. Deviations from the Original MINGLE Paper
> [!IMPORTANT]
> This run utilized a single controlled deviation explicitly requested for local/free computation:
> 
> **Original Implementation**: OpenAI's API (`text-embedding-ada-002`, 1536d)
> **Our Implementation**: Open-source local transformer (`NeuML/biomedbert-base-embeddings`, 768d)
> 
> As a result, the final $X_v$ dimension is 832 (instead of 1600). The $N_e$ dimension is 768 (instead of 1536). All message passing algorithms in Vanilla BCE will inherit these dimensions natively. No architectural changes were required to accommodate this.
