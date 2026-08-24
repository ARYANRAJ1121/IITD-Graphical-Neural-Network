import time
import psutil
import os
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import json

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def main():
    print("=== Pilot Stage 2: Embedding Generation ===")
    
    model_name = "NeuML/biomedbert-base-embeddings"
    print(f"Loading model: {model_name}...")
    
    start_load = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds.")
    
    # Extract model info
    print("\n--- Model Information ---")
    print(f"Model name: {model_name}")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"Max sequence length: {model.max_seq_length}")
    print(f"Device: {model.device}")
    
    # Generate mock concepts and notes for the pilot
    print("\n--- Generating 100 mock concepts and 100 notes ---")
    concepts = [f"Sample medical concept {i} for testing" for i in range(100)]
    notes = [
        f"Chief Complaint No complaints. History of Present Illness Patient {i} is a 45 year-old with hypertension and diabetes. Current medications include lisinopril and metformin."
        for i in range(100)
    ]
    
    # Encode concepts
    print("\n--- Encoding Concepts (C_v) ---")
    mem_before = get_memory_usage()
    start_time = time.time()
    c_v_embeddings = model.encode(concepts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    concept_time = time.time() - start_time
    mem_after = get_memory_usage()
    
    print(f"Concept Encoding Time: {concept_time:.2f} seconds")
    print(f"Memory increase: {mem_after - mem_before:.2f} MB")
    print(f"Shape: {c_v_embeddings.shape}")
    print(f"Contains NaN: {np.isnan(c_v_embeddings).any()}")
    print(f"Contains Inf: {np.isinf(c_v_embeddings).any()}")
    
    # Encode notes
    print("\n--- Encoding Notes (N_e) ---")
    mem_before = get_memory_usage()
    start_time = time.time()
    n_e_embeddings = model.encode(notes, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    note_time = time.time() - start_time
    mem_after = get_memory_usage()
    
    print(f"Note Encoding Time: {note_time:.2f} seconds")
    print(f"Memory increase: {mem_after - mem_before:.2f} MB")
    print(f"Shape: {n_e_embeddings.shape}")
    print(f"Contains NaN: {np.isnan(n_e_embeddings).any()}")
    print(f"Contains Inf: {np.isinf(n_e_embeddings).any()}")
    
    print("\nPilot validation successful!")

if __name__ == "__main__":
    main()
