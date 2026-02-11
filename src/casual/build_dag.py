# src/casual/build_dag.py

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
CSV_FILE = "data/processed/all_sessions.csv"
OUTPUT_DIR = "data/processed"
MIN_PROB_THRESHOLD = 0.01  # Optional: filter weak edges

# -----------------------------
# Step 1: Load processed CSV
# -----------------------------
def load_data():
    """Load and validate session data"""
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"❌ File not found: {CSV_FILE}")
    
    df = pd.read_csv(CSV_FILE)
    
    if df.empty:
        raise ValueError("❌ CSV file is empty!")
    
    print(f"✅ Loaded {len(df)} events from {len(df['session_id'].unique())} sessions")
    return df.sort_values(by=["session_id", "timestamp"])

# -----------------------------
# Step 2: Build DAG
# -----------------------------
def build_dag(df):
    """Build Directed Acyclic Graph from event sequences"""
    edges = defaultdict(int)
    
    for session_id, group in df.groupby("session_id"):
        events = group["event"].tolist()
        for i in range(len(events) - 1):
            src, dst = events[i], events[i + 1]
            if src != dst:  # Skip self-loops
                edges[(src, dst)] += 1
    
    print(f"✅ Found {len(edges)} unique transitions")
    return edges

# -----------------------------
# Step 3: Compute probabilities
# -----------------------------
def compute_probabilities(edges):
    """Calculate Markov transition probabilities"""
    total_from = defaultdict(int)
    for (src, dst), count in edges.items():
        total_from[src] += count
    
    probabilities = {}
    for (src, dst), count in edges.items():
        probabilities[(src, dst)] = count / total_from[src]
    
    return probabilities, total_from

# -----------------------------
# Step 4: Create NetworkX graph
# -----------------------------
def create_graph(df, probabilities):
    """Create NetworkX DiGraph with probabilities"""
    G = nx.DiGraph()
    
    # Add all unique events as nodes
    for event in df["event"].unique():
        G.add_node(event)
    
    # Add edges with probability as weight
    for (src, dst), prob in probabilities.items():
        G.add_edge(src, dst, weight=prob)
    
    # Add node attributes for frequency
    node_freq = df["event"].value_counts().to_dict()
    nx.set_node_attributes(G, node_freq, "frequency")
    
    return G

# -----------------------------
# Step 5: Visualize
# -----------------------------
def visualize_graph(G, probabilities, output_path="dag_visualization.png"):
    """Create and save DAG visualization"""
    plt.figure(figsize=(14, 10))
    
    # Use hierarchical layout for DAG
    try:
        pos = nx.multipartite_layout(G, subset_key="layer") 
    except:
        pos = nx.spring_layout(G, seed=42, k=2)
    
    # Node sizes based on frequency
    node_sizes = [G.nodes[n].get("frequency", 100) * 10 for n in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color="lightblue", alpha=0.8)
    
    # Draw edges with thickness based on probability
    edge_widths = [probabilities.get((u, v), 0.1) * 10 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          edge_color="gray", alpha=0.6, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add edge labels for top probabilities
    top_edges = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:20]
    edge_labels = {(src, dst): f"{prob:.2f}" for (src, dst), prob in top_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Network Event DAG\n(Edge thickness = probability)", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Visualization saved to {output_path}")
    plt.close()  # Close figure to free memory

# -----------------------------
# Step 6: Save outputs
# -----------------------------
def save_outputs(G, probabilities, edges_df, total_from):
    """Save all DAG outputs"""
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Save edges as CSV
    edges_df.to_csv(f"{OUTPUT_DIR}/dag_edges.csv", index=False)
    
    # 2. Save graph model
    model_data = {
        'graph': G,
        'probabilities': probabilities,
        'total_from': total_from,
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges()
    }
    
    with open(f"{OUTPUT_DIR}/dag_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    # 3. Save statistics
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_dag': nx.is_directed_acyclic_graph(G)
    }
    
    pd.DataFrame([stats]).to_csv(f"{OUTPUT_DIR}/dag_stats.csv", index=False)
    
    print(f"💾 All outputs saved to {OUTPUT_DIR}/")

# -----------------------------
# Main function
# -----------------------------
def main():
    print("🚀 Building DAG from event sequences...")
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Build DAG
    edges = build_dag(df)
    
    # Step 3: Compute probabilities
    probabilities, total_from = compute_probabilities(edges)
    
    # Step 4: Create graph
    G = create_graph(df, probabilities)
    
    # Step 5: Create edges DataFrame
    edges_df = pd.DataFrame([
        {"source": src, "target": dst, "count": edges[(src, dst)], "probability": prob}
        for (src, dst), prob in probabilities.items()
    ])
    
    # Step 6: Print statistics
    print("\n📈 DAG Statistics:")
    print(f"• Unique events (nodes): {G.number_of_nodes()}")
    print(f"• Unique transitions (edges): {G.number_of_edges()}")
    print(f"• Graph density: {nx.density(G):.4f}")
    print(f"• Is DAG: {nx.is_directed_acyclic_graph(G)}")
    
    # Show top transitions
    print("\n🏆 Top 5 most common transitions:")
    top_transitions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    for (src, dst), prob in top_transitions:
        print(f"  {src} → {dst}: {prob:.3f} ({edges[(src, dst)]} occurrences)")
    
    # Step 7: Save outputs
    save_outputs(G, probabilities, edges_df, total_from)
    
    # Step 8: Visualize
    visualize_graph(G, probabilities, f"{OUTPUT_DIR}/dag_visualization.png")
    
    print("\n✅ DAG construction complete!")

if __name__ == "__main__":
    main()