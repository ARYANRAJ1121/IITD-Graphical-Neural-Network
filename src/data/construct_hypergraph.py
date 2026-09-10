import orjson as json
import random
import base64
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import sys
import time
import os
import networkx as nx
from gensim.models import Word2Vec
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def generate_random_walks(graph, num_walks, walk_length):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(graph.neighbors(cur))
                if len(neighbors) > 0:
                    walk.append(random.choice(neighbors))
                else:
                    break
            walks.append(walk)
    return walks

def main():
    print("=== Hypergraph construction + semantic embedding ===")
    
    fhir_dir = Path("../fhir")
    if not fhir_dir.exists():
        fhir_dir = Path("fhir")
    
    bundles = list(fhir_dir.glob("*.json"))
    bundles = [b for b in bundles if b.name not in ["organizations.json", "practitioners.json"]]
    
    # 1. Parse Graph Structure
    print(f"Parsing {len(bundles)} FHIR bundles...")
    
    encounters = {} # enc_id -> {patient_id, start_date}
    concept_counts = Counter()
    concept_name_map = {} # concept_id -> display
    encounter_nodes = defaultdict(set) # enc_id -> set of concept_ids
    clinical_notes = {} # enc_id -> text
    
    node_types = ['Condition', 'MedicationRequest', 'Procedure', 'Observation'] # Exclude DiagnosticReport

    for b in bundles:
        with open(b, 'rb') as f:
            data = json.loads(f.read())
            
        patient_id = None
        for entry in data.get('entry', []):
            res = entry.get('resource', {})
            if res.get('resourceType') == 'Patient':
                patient_id = res.get('id')
                break
                
        for entry in data.get('entry', []):
            res = entry.get('resource', {})
            rt = res.get('resourceType')
            
            if rt == 'Encounter':
                enc_id = res.get('id')
                start_date = res.get('period', {}).get('start', '')
                encounters[enc_id] = {'patient_id': patient_id, 'start_date': start_date}
                
            elif rt in node_types:
                enc_ref = res.get('encounter', {}).get('reference', '')
                if enc_ref:
                    enc_id = enc_ref.replace('urn:uuid:', '')
                    
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
                        
                        if code_str and display:
                            concept_id = f"{sys_str}|{code_str}"
                            concept_name_map[concept_id] = display
                            encounter_nodes[enc_id].add(concept_id)
                            concept_counts[concept_id] += 1
                            
            elif rt == 'DocumentReference':
                enc_ref = res.get('context', {}).get('encounter', [])
                if enc_ref:
                    enc_id = enc_ref[0].get('reference', '').replace('urn:uuid:', '')
                    data_b64 = res.get('content', [{}])[0].get('attachment', {}).get('data', '')
                    if data_b64:
                        try:
                            text = base64.b64decode(data_b64).decode('utf-8')
                            clinical_notes[enc_id] = text
                        except:
                            pass

    # Process and filter Encounters
    print("\n--- Constructing Hyperedges (Encounters) ---")
    valid_encounters = []
    # Sort encounters by patient_id, then start_date (Chronological ordering)
    for enc_id, info in encounters.items():
        valid_encounters.append({
            'encounter_id': enc_id,
            'patient_id': info['patient_id'],
            'start_date': info['start_date']
        })
        
    valid_encounters.sort(key=lambda x: (x['patient_id'], x['start_date']))
    valid_enc_ids = [x['encounter_id'] for x in valid_encounters]
    
    print(f"Total chronological hyperedges (encounters): {len(valid_encounters)}")
    
    # Process Nodes
    print("\n--- Constructing Nodes (Medical Concepts) ---")
    valid_nodes = list(concept_name_map.keys())
    print(f"Total unique concept nodes: {len(valid_nodes)}")
    
    # Edge lists
    node_encounter_edges = []
    for enc_id in valid_enc_ids:
        for node_id in encounter_nodes[enc_id]:
            node_encounter_edges.append({'encounter_id': enc_id, 'node_id': node_id})
    print(f"Total node-hyperedge connections: {len(node_encounter_edges)}")
    
    # 2. Structural Embeddings (DeepWalk)
    print("\n--- Generating DeepWalk Structural Embeddings (s_v) ---")
    G = nx.Graph()
    G.add_nodes_from(valid_nodes)
    
    # Add co-occurrence edges
    print("Building co-occurrence graph...")
    for enc_id, nodes in encounter_nodes.items():
        nodes_list = list(nodes)
        for i in range(len(nodes_list)):
            for j in range(i+1, len(nodes_list)):
                u, v = nodes_list[i], nodes_list[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
                    
    print(f"Co-occurrence graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    walks = generate_random_walks(G, num_walks=10, walk_length=10)
    # Convert node IDs to strings for Word2Vec
    walks = [[str(node) for node in walk] for walk in walks]
    
    print("Training Word2Vec model...")
    d1 = 64
    w2v = Word2Vec(walks, vector_size=d1, window=5, min_count=0, sg=1, workers=4, epochs=5)
    
    structural_embeddings = np.zeros((len(valid_nodes), d1))
    for i, node_id in enumerate(valid_nodes):
        structural_embeddings[i] = w2v.wv[str(node_id)]
        
    print(f"Structural embeddings generated, shape: {structural_embeddings.shape}")
    
    # 3. Semantic Embeddings
    print("\n--- Generating Semantic Embeddings ---")
    model_name = "NeuML/biomedbert-base-embeddings"
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Node Semantics (C_v)
    print("Encoding Concept Names (C_v)...")
    concept_names = [concept_name_map[node] for node in valid_nodes]
    semantic_concept_embeddings = model.encode(concept_names, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    print(f"Semantic concept embeddings shape: {semantic_concept_embeddings.shape}")
    
    # Concatenate structural and semantic representations as per MINGLE
    # X_v^{(0)} = [s_v ; c_v]
    X_v = np.concatenate([structural_embeddings, semantic_concept_embeddings], axis=1)
    print(f"Final concatenated node representations (X_v) shape: {X_v.shape}")
    
    # Note Semantics (N_e)
    print("Encoding Clinical Notes (N_e)...")
    # We want 1-to-1 mapping with valid_encounters
    note_texts = []
    for enc_id in valid_enc_ids:
        note_texts.append(clinical_notes.get(enc_id, ""))
        
    # Checkpointing and batching
    os.makedirs('data/processed', exist_ok=True)
    note_emb_path = 'data/processed/note_embeddings.npy'
    
    if os.path.exists(note_emb_path):
        print(f"Found existing {note_emb_path}, loading...")
        N_e = np.load(note_emb_path)
    else:
        print("Generating new note embeddings in batches with checkpointing...")
        batch_size = 5000
        N_e_list = []
        for i in range(0, len(note_texts), batch_size):
            chunk_file = f'data/processed/note_embeddings_chunk_{i}.npy'
            if os.path.exists(chunk_file):
                print(f"Loading existing chunk {chunk_file}...")
                emb_batch = np.load(chunk_file)
            else:
                batch = note_texts[i:i+batch_size]
                emb_batch = model.encode(batch, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
                np.save(chunk_file, emb_batch)
                print(f"Processed and saved chunk {min(i+batch_size, len(note_texts))}/{len(note_texts)} notes")
            N_e_list.append(emb_batch)
            
        N_e = np.vstack(N_e_list)
        np.save(note_emb_path, N_e)
        
        # Cleanup chunks
        for i in range(0, len(note_texts), batch_size):
            chunk_file = f'data/processed/note_embeddings_chunk_{i}.npy'
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        
    print(f"Clinical note embeddings (N_e) shape: {N_e.shape}")
    
    # Validation checks
    print("\n--- Validation Audit ---")
    assert len(valid_encounters) == N_e.shape[0], "Mismatch in encounter count vs note embeddings count"
    assert len(valid_nodes) == X_v.shape[0], "Mismatch in node count vs node embeddings count"
    assert not np.isnan(X_v).any(), "NaN found in X_v"
    assert not np.isnan(N_e).any(), "NaN found in N_e"
    assert not np.isinf(X_v).any(), "Inf found in X_v"
    assert not np.isinf(N_e).any(), "Inf found in N_e"
    
    # Ensure no orphans
    nodes_in_edges = set(e['node_id'] for e in node_encounter_edges)
    encs_in_edges = set(e['encounter_id'] for e in node_encounter_edges)
    
    # Encounters can be orphans if they have notes but no diagnoses (common in Coherent)
    orphan_nodes = set(valid_nodes) - nodes_in_edges
    print(f"Orphan nodes: {len(orphan_nodes)}")
    assert len(orphan_nodes) == 0, "Orphan nodes found!"
    
    # Save outputs
    print("\n--- Saving Final Outputs ---")
    
    pd.DataFrame({'node_id': valid_nodes, 'display': concept_names}).to_csv('data/processed/concept_nodes.csv', index=False)
    pd.DataFrame(valid_encounters).to_csv('data/processed/encounter_hyperedges.csv', index=False)
    pd.DataFrame(node_encounter_edges).to_csv('data/processed/node_encounter_edges.csv', index=False)
    pd.DataFrame({'encounter_id': valid_enc_ids, 'note_text': note_texts}).to_csv('data/processed/clinical_notes.csv', index=False)
    
    np.save('data/processed/node_embeddings.npy', X_v)
    
    metadata = {
        'model_name': model_name,
        'embedding_dimension': model.get_sentence_embedding_dimension(),
        'max_seq_length': model.max_seq_length,
        'pooling_method': 'mean', # Default for biomedbert
        'structural_dimension_d1': d1,
        'semantic_dimension_d2': model.get_sentence_embedding_dimension(),
        'total_node_dimension_d': X_v.shape[1],
        'note_embedding_dimension': N_e.shape[1]
    }
    with open('data/processed/embedding_metadata.json', 'w') as f:
        f.write(json.dumps(metadata, option=json.OPT_INDENT_2).decode('utf-8'))
        
    stats = {
        'num_hyperedges': len(valid_encounters),
        'num_nodes': len(valid_nodes),
        'num_node_hyperedge_connections': len(node_encounter_edges),
        'orphan_nodes': len(orphan_nodes)
    }
    with open('data/processed/hypergraph_statistics.json', 'w') as f:
        f.write(json.dumps(stats, option=json.OPT_INDENT_2).decode('utf-8'))
        
    print("Hypergraph construction completed successfully!")

if __name__ == '__main__':
    main()
