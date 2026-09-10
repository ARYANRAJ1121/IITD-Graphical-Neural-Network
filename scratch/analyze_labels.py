import pandas as pd
from collections import Counter
import json

def main():
    print("Loading data...")
    encounters = pd.read_csv('data/processed/encounter_hyperedges.csv')
    edges = pd.read_csv('data/processed/node_encounter_edges.csv')
    nodes = pd.read_csv('data/processed/concept_nodes.csv')
    
    # Sort encounters just in case, but they should be chronological
    encounters['start_date'] = pd.to_datetime(encounters['start_date'], utc=True)
    encounters = encounters.sort_values(by=['patient_id', 'start_date'])
    
    # Get condition nodes only (SNOMED usually corresponds to conditions/procedures)
    condition_nodes = nodes[nodes['node_id'].str.contains('snomed', case=False, na=False)]
    condition_node_ids = set(condition_nodes['node_id'])
    
    # Count frequencies of conditions across all edges
    print("Finding top 25 conditions...")
    condition_edges = edges[edges['node_id'].isin(condition_node_ids)]
    top_25_nodes = condition_edges['node_id'].value_counts().head(25).index.tolist()
    
    top_25_names = nodes.set_index('node_id').loc[top_25_nodes, 'display'].tolist()
    
    # Map encounters to their labels (which of top 25 they contain)
    enc_to_labels = {e_id: set() for e_id in encounters['encounter_id']}
    for _, row in edges[edges['node_id'].isin(top_25_nodes)].iterrows():
        enc_to_labels[row['encounter_id']].add(row['node_id'])
    
    # Form t -> t+1 pairs
    print("Forming pairs...")
    enc_list = encounters.to_dict('records')
    valid_pairs = 0
    total_encounters = len(enc_list)
    
    label_counts_next = {node_id: 0 for node_id in top_25_nodes}
    
    for i in range(len(enc_list) - 1):
        e_curr = enc_list[i]
        e_next = enc_list[i+1]
        
        if e_curr['patient_id'] == e_next['patient_id']:
            valid_pairs += 1
            labels_next = enc_to_labels[e_next['encounter_id']]
            for node_id in top_25_nodes:
                if node_id in labels_next:
                    label_counts_next[node_id] += 1
                    
    # Generate Report
    report = [
        "# Vanilla BCE Label Feasibility Analysis",
        "",
        "## Dataset Statistics",
        f"- Total Encounters (Hyperedges): {total_encounters}",
        f"- Valid $t \rightarrow t+1$ training pairs: {valid_pairs}",
        f"- Terminal visits (no next visit, excluded from loss): {total_encounters - valid_pairs}",
        "",
        "## Top 25 Candidate Conditions (Target Labels)",
        "| Rank | Node ID | Condition Name | Positive Examples in $t+1$ | Prevalence (%) |",
        "|------|---------|----------------|----------------------------|----------------|"
    ]
    
    for i, node_id in enumerate(top_25_nodes):
        count = label_counts_next[node_id]
        prev = (count / valid_pairs) * 100 if valid_pairs > 0 else 0
        name = top_25_names[i]
        report.append(f"| {i+1} | `{node_id.split('|')[-1]}` | {name} | {count} | {prev:.2f}% |")
    
    report.append("")
    report.append("## Conclusion")
    
    zero_classes = sum(1 for v in label_counts_next.values() if v == 0)
    low_classes = sum(1 for v in label_counts_next.values() if v < (valid_pairs * 0.01))
    
    if zero_classes > 0 or low_classes > 5:
        report.append(f"**WARNING: Label Sparsity Detected.** There are {zero_classes} classes with zero positive examples, and {low_classes} classes with <1% prevalence in the next-visit targets. This task may lead to degenerate model training or all-zero predictions.")
    else:
        report.append("**FEASIBLE:** The target labels have sufficient representation across the dataset to train a meaningful prediction head without encountering zero-class errors.")
        
    with open('experiments/label_feasibility.md', 'w') as f:
        f.write("\n".join(report))
        
    print("Report saved to experiments/label_feasibility.md")

if __name__ == '__main__':
    main()
