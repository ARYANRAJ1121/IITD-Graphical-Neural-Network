# Stage 2 Replication Audit

This report verifies the Coherent dataset Stage 2 implementation against the original MINGLE paper methodology.

### 1. Is $S_v$ actually generated using DeepWalk on the same type of node co-occurrence structure described by MINGLE?
**Yes.** We constructed a node co-occurrence graph using `networkx` where edges denote that two medical concepts (nodes) co-occurred within the same encounter (visit/hyperedge). We then generated random walks of length 10 over this graph and fed them into a `Word2Vec` skip-gram model, which perfectly replicates the DeepWalk methodology used in MINGLE to generate structural embeddings $s_v$.

### 2. Is $C_v$ generated from the concept NAME/DISPLAY text rather than the raw medical code?
**Yes.** We extracted the `display` attribute from the FHIR `CodeableConcept.coding` elements (e.g., extracting "Essential hypertension" instead of just the ICD/SNOMED code string) and passed this plain English text directly into the transformer model to generate the semantic concept embeddings $C_v$.

### 3. Is $N_e$ generated from the clinical note associated with each hyperedge/visit?
**Yes.** We extracted the Base64-encoded attachment from the FHIR `DocumentReference` resource, decoded it into raw text, and linked it strictly to the `Encounter.id` specified in the document's `context`. These texts were then embedded to form $N_e$.

### 4. Are the note embeddings correctly aligned one-to-one with Encounter.id?
**Yes.** The script built a chronological list of `valid_encounters`. When generating the $N_e$ batch embeddings, it iterated exactly through this chronologically sorted list of `valid_encounters`, appending empty strings for any encounters that lacked a note. Therefore, Row $i$ of `note_embeddings.npy` corresponds exactly to Row $i$ of `encounter_hyperedges.csv`.

### 5. Is $X_v^{(0)}$ actually $[S_v ; C_v]$?
**Yes.** In `construct_graph_stage2.py`, we explicitly performed `X_v = np.concatenate([structural_embeddings, semantic_concept_embeddings], axis=1)`, which is the exact tensor concatenation $s_v \oplus c_v$ defined in the paper.

### 6. Are self-loops being represented as required by MINGLE?
**No, not yet.** The MINGLE paper specifies that self-loops are added to the hypergraph to ensure nodes retain their own features during message passing. Currently, `node_encounter_edges.csv` only contains genuine visit relationships. Self-loops are a topological addition (adding an identity matrix to the incidence matrix $H$) that must be handled dynamically inside the PyTorch `forward()` pass in Stage 3. 

### 7. Have we actually constructed $H_e = \text{MLP}_1([N_e ; C_v])$ yet, or is that still part of Stage 3?
**That is strictly part of Stage 3.** $\text{MLP}_1$ is a neural network component with trainable weights. In Stage 2, we have only generated the fixed initial features $N_e$ and $C_v$ that will eventually be fed into that MLP during training.

### 8. Have we implemented the MINGLE hyperedge/node message-passing equations yet, or not?
**Not yet.** Stage 2 was exclusively focused on raw data extraction, structural graph building, and initial semantic embedding computation via Frozen Language Models. The actual hypergraph convolution/message-passing architecture belongs in Stage 3.

### 9. What exactly differs from the original MINGLE implementation?
There is only one methodological deviation from the original paper regarding Stage 2:
* **Embedding Model**: The MINGLE paper used OpenAI's paid `text-embedding-ada-002` API, which yields 1536-dimensional vectors. Per your instructions, we used a local open-source transformer (`NeuML/biomedbert-base-embeddings`) running locally on CPU. This yields 768-dimensional vectors. 
* *Note: This does not break the MINGLE architecture, it merely halves the semantic input dimensions that Stage 3 will ingest.*

### 10. Exact dimensions of $S_v$, $C_v$, $N_e$, and $X_v^{(0)}$
* **$S_v$ (DeepWalk)**: `(564, 64)` — 564 unique nodes, 64 structural dimensions.
* **$C_v$ (Concept Semantic)**: `(564, 768)` — 564 unique nodes, 768 semantic dimensions from BiomedBERT.
* **$N_e$ (Note Semantic)**: `(143946, 768)` — 143,946 visits, 768 semantic dimensions from BiomedBERT.
* **$X_v^{(0)}$ (Initial Node Features)**: `(564, 832)` — 564 unique nodes, 832 fused dimensions ($64 + 768$).
