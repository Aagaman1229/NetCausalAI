import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import pickle
import os
import json
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
CSV_FILE = "data/processed/all_sessions_detailed.csv"  # Use detailed version
OUTPUT_DIR = "data/processed"
MIN_PROB_THRESHOLD = 0.01

# -----------------------------
# Step 1: Load processed CSV
# -----------------------------
def load_data():
    """Load and validate session data"""
    if not os.path.exists(CSV_FILE):
        # Fallback to lightweight version if detailed doesn't exist
        fallback = "data/processed/all_sessions.csv"
        print(f"⚠️ Detailed CSV not found, using {fallback}")
        if os.path.exists(fallback):
            df = pd.read_csv(fallback)
        else:
            raise FileNotFoundError(f"No session file found")
    else:
        df = pd.read_csv(CSV_FILE)
    
    if df.empty:
        raise ValueError("❌ CSV file is empty!")
    
    print(f"✅ Loaded {len(df)} events from {len(df['session_id'].unique())} sessions")
    return df.sort_values(by=["session_id", "timestamp"])

# -----------------------------
# Step 2: Build DAG with transition metrics
# -----------------------------
def build_dag_with_metrics(df):
    """
    Build DAG and compute ALL transition metrics needed for RCA
    """
    # Count transitions
    transition_counts = defaultdict(int)
    transition_sessions = defaultdict(set)  # Track which sessions have this transition
    transition_timestamps = defaultdict(list)  # Track when transitions occur
    
    # Per-session transition data
    session_transitions = {}
    
    for session_id, group in df.groupby("session_id"):
        events = group["event"].tolist()
        timestamps = group["timestamp"].tolist()
        
        session_trans = []
        
        for i in range(len(events) - 1):
            src, dst = events[i], events[i + 1]
            if src != dst:  # Skip self-loops
                trans_key = (src, dst)
                
                # Global metrics
                transition_counts[trans_key] += 1
                transition_sessions[trans_key].add(session_id)
                transition_timestamps[trans_key].append(timestamps[i])
                
                # Per-session metrics
                session_trans.append({
                    'position': i,
                    'src': src,
                    'dst': dst,
                    'timestamp': timestamps[i],
                    'next_timestamp': timestamps[i+1]
                })
        
        session_transitions[session_id] = session_trans
    
    print(f"✅ Found {len(transition_counts)} unique transitions")
    
    return {
        'transition_counts': transition_counts,
        'transition_sessions': transition_sessions,
        'transition_timestamps': transition_timestamps,
        'session_transitions': session_transitions
    }

# -----------------------------
# Step 3: Compute probabilities with confidence
# -----------------------------
def compute_probabilities_with_confidence(transition_data):
    """
    Compute transition probabilities AND confidence metrics
    """
    counts = transition_data['transition_counts']
    sessions = transition_data['transition_sessions']
    
    # Total transitions from each source
    total_from = defaultdict(int)
    for (src, dst), count in counts.items():
        total_from[src] += count
    
    # Probabilities and confidence metrics
    probabilities = {}
    confidence_scores = {}
    
    for (src, dst), count in counts.items():
        prob = count / total_from[src]
        probabilities[(src, dst)] = prob
        
        # Confidence based on number of sessions and count
        num_sessions = len(sessions[(src, dst)])
        confidence = min(1.0, (count / 10) * (num_sessions / 5))  # Heuristic
        confidence_scores[(src, dst)] = confidence
    
    return {
        'probabilities': probabilities,
        'total_from': total_from,
        'confidence_scores': confidence_scores
    }

# -----------------------------
# Step 4: Create enhanced graph
# -----------------------------
def create_enhanced_graph(df, probabilities, confidence_scores):
    """
    Create NetworkX DiGraph with ALL node/edge attributes
    """
    G = nx.DiGraph()
    
    # Add nodes with attributes
    event_counts = df["event"].value_counts().to_dict()
    event_sessions = df.groupby("event")["session_id"].nunique().to_dict()
    
    for event in df["event"].unique():
        G.add_node(event, 
                  frequency=event_counts.get(event, 0),
                  sessions=event_sessions.get(event, 0),
                  type=event.split('_')[0] if '_' in event else event)
    
    # Add edges with all attributes
    for (src, dst), prob in probabilities.items():
        G.add_edge(src, dst, 
                  weight=prob,
                  probability=prob,
                  confidence=confidence_scores.get((src, dst), 0),
                  count=probabilities.get((src, dst), 0))
    
    return G

# -----------------------------
# Step 5: Compute session features for clustering
# -----------------------------
def compute_session_features(df, transition_data, prob_data):
    """
    Compute ALL features needed for RCA and clustering
    """
    print("📊 Computing session features for RCA and clustering...")
    
    features = []
    probabilities = prob_data['probabilities']
    min_prob = 0.001
    session_transitions = transition_data['session_transitions']
    
    for session_id, group in df.groupby("session_id"):
        events = group["event"].tolist()
        
        if len(events) < 2:
            continue
        
        # Get session-level aggregates from the first row
        first_row = group.iloc[0]
        
        # ============ TRANSITION-LEVEL METRICS ============
        transition_probs = []
        unseen_count = 0
        rare_count = 0
        
        for i in range(len(events) - 1):
            prob = probabilities.get((events[i], events[i+1]), min_prob)
            transition_probs.append(prob)
            
            if prob == min_prob:
                unseen_count += 1
            elif prob < 0.01:
                rare_count += 1
        
        # Find first unseen transition
        first_unseen_idx = -1
        for i in range(len(events) - 1):
            if probabilities.get((events[i], events[i+1]), min_prob) == min_prob:
                first_unseen_idx = i
                break
        
        # Cumulative log scores at percentiles
        log_probs = [np.log(p) for p in transition_probs]
        cum_log = np.cumsum(log_probs)
        
        if len(cum_log) > 0:
            log_at_25 = cum_log[int(len(cum_log) * 0.25)] if len(cum_log) > 0 else 0
            log_at_50 = cum_log[int(len(cum_log) * 0.5)] if len(cum_log) > 0 else 0
            log_at_75 = cum_log[int(len(cum_log) * 0.75)] if len(cum_log) > 0 else 0
            log_at_100 = cum_log[-1] if len(cum_log) > 0 else 0
        else:
            log_at_25 = log_at_50 = log_at_75 = log_at_100 = 0
        
        # ============ AGGREGATE SESSION FEATURES ============
        # Duration and bytes
        duration = group['session_duration'].iloc[0] if 'session_duration' in group.columns else 0
        bytes_total = group['session_bytes_total'].iloc[0] if 'session_bytes_total' in group.columns else group['length'].sum()
        packet_rate = group['session_packet_rate'].iloc[0] if 'session_packet_rate' in group.columns else (len(group) / duration if duration > 0 else 0)
        
        # Repetition rate
        unique_events = len(set(events))
        repetition_rate = 1 - (unique_events / len(events)) if len(events) > 0 else 0
        
        # Entropy of transition probabilities
        probs_array = np.array(transition_probs)
        probs_array = probs_array[probs_array > 0]
        entropy = -np.sum(probs_array * np.log(probs_array)) if len(probs_array) > 0 else 0
        
        # ============ NETWORK CONTEXT ============
        # Get destination info
        dst_ips = group['dst_ip'].unique() if 'dst_ip' in group.columns else []
        dst_ports = group['dst_port'].unique() if 'dst_port' in group.columns else []
        
        # Known services (common ports)
        known_ports = {80, 443, 22, 21, 25, 53, 3306, 5432, 27017, 6379}
        is_known_service = any(p in known_ports for p in dst_ports if pd.notna(p))
        
        # ============ DERIVED FEATURES ============
        # Burstiness (from session aggregates)
        burstiness = group['session_max_packets_per_sec'].iloc[0] if 'session_max_packets_per_sec' in group.columns else 1
        
        # Downstream diversity
        downstream_diversity = group['session_unique_dst_ips'].iloc[0] if 'session_unique_dst_ips' in group.columns else len(dst_ips)
        port_diversity = group['session_unique_dst_ports'].iloc[0] if 'session_unique_dst_ports' in group.columns else len(dst_ports)
        
        # Ratio of large transfers
        large_transfer_count = sum(1 for e in events if e == 'LARGE_TRANSFER')
        ratio_large = large_transfer_count / len(events) if len(events) > 0 else 0
        
        # Build feature vector
        feature_row = {
            # Session ID
            'session_id': session_id,
            
            # ===== TRANSITION METRICS =====
            'avg_transition_prob': np.mean(transition_probs) if transition_probs else 0,
            'min_transition_prob': np.min(transition_probs) if transition_probs else 0,
            'std_transition_prob': np.std(transition_probs) if transition_probs else 0,
            'number_unseen_transitions': unseen_count,
            'number_rare_transitions': rare_count,
            'index_first_unseen_normalized': first_unseen_idx / len(events) if first_unseen_idx >= 0 else -1,
            'cumulative_log_25': log_at_25,
            'cumulative_log_50': log_at_50,
            'cumulative_log_75': log_at_75,
            'cumulative_log_100': log_at_100,
            
            # ===== SESSION AGGREGATES =====
            'event_count': len(events),
            'duration_seconds': duration,
            'bytes_transferred': bytes_total,
            'avg_event_rate': packet_rate,
            'repetition_rate': repetition_rate,
            'unique_events_count': unique_events,
            'entropy_of_transition_probs': entropy,
            
            # ===== NETWORK CONTEXT =====
            'src_ip': first_row.get('src_ip', 'unknown'),
            'dst_ip': first_row.get('dst_ip', 'unknown'),
            'dst_port': first_row.get('dst_port', 0),
            'protocol': first_row.get('protocol', 0),
            'is_known_service': is_known_service,
            
            # ===== DERIVED FEATURES =====
            'burstiness': burstiness,
            'downstream_diversity_ips': downstream_diversity,
            'downstream_diversity_ports': port_diversity,
            'ratio_large_transfer_events': ratio_large,
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)

# -----------------------------
# Step 6: Save ALL outputs
# -----------------------------
def save_all_outputs(G, prob_data, transition_data, session_features):
    """Save complete outputs for RCA and clustering"""
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Save graph model
    model_data = {
        'graph': G,
        'probabilities': prob_data['probabilities'],
        'confidence_scores': prob_data['confidence_scores'],
        'total_from': prob_data['total_from'],
        'transition_counts': transition_data['transition_counts'],
        'transition_sessions': transition_data['transition_sessions'],
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges()
    }
    
    with open(f"{OUTPUT_DIR}/dag_model_complete.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    # 2. Save transition probabilities as CSV
    edges_df = pd.DataFrame([
        {
            'source': src, 
            'target': dst, 
            'probability': prob,
            'confidence': prob_data['confidence_scores'].get((src, dst), 0),
            'count': transition_data['transition_counts'].get((src, dst), 0),
            'sessions': len(transition_data['transition_sessions'].get((src, dst), set()))
        }
        for (src, dst), prob in prob_data['probabilities'].items()
    ])
    edges_df.to_csv(f"{OUTPUT_DIR}/dag_edges_complete.csv", index=False)
    
    # 3. Save session features for clustering
    session_features.to_csv(f"{OUTPUT_DIR}/session_features_for_clustering.csv", index=False)
    
    # 4. Save feature summary
    summary = {
        'num_sessions': len(session_features),
        'num_features': len(session_features.columns),
        'feature_names': list(session_features.columns),
        'feature_types': session_features.dtypes.astype(str).to_dict()
    }
    
    with open(f"{OUTPUT_DIR}/feature_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"💾 All outputs saved to {OUTPUT_DIR}/")
    print(f"   • dag_model_complete.pkl - Full model with transition data")
    print(f"   • dag_edges_complete.csv - Edge probabilities with confidence")
    print(f"   • session_features_for_clustering.csv - {len(session_features)} sessions with {len(session_features.columns)} features")
    print(f"   • feature_summary.json - Feature metadata")


import matplotlib.pyplot as plt

def visualize_minimal(G, output_path="dag.png"):
    """Ultra simple - shows exactly what you need"""
    plt.figure(figsize=(14, 10))
    
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Nodes - size = frequency
    sizes = [G.nodes[n].get('frequency', 100) / 5 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='lightblue')
    
    # Edges - width = probability
    edges = G.edges(data=True)
    widths = [d['probability'] * 15 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray',
                          arrows=True, arrowsize=15)
    
    # Edge labels - probability + count
    labels = {(u, v): f"{d['probability']:.0%}\n({d.get('count',0)})" 
              for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    
    # Node labels - name + count
    node_labels = {n: f"{n}\n({G.nodes[n].get('frequency',0)})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)
    
    plt.title("Event Transitions - Probability% (count)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

# -----------------------------
# Main function
# -----------------------------
def main():
    print("🚀 Building complete DAG with RCA and clustering features...")
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Build DAG with metrics
    transition_data = build_dag_with_metrics(df)
    
    # Step 3: Compute probabilities with confidence
    prob_data = compute_probabilities_with_confidence(transition_data)
    
    # Step 4: Create graph
    G = create_enhanced_graph(df, prob_data['probabilities'], prob_data['confidence_scores'])
    
    visualize_minimal(G, output_path=f"{OUTPUT_DIR}/dag_graph_complete.png")
    # Step 5: Compute session features for clustering
    session_features = compute_session_features(df, transition_data, prob_data)
    
    # Step 6: Print statistics
    print("\n📈 DAG Statistics:")
    print(f"• Unique events: {G.number_of_nodes()}")
    print(f"• Unique transitions: {G.number_of_edges()}")
    print(f"• Sessions with features: {len(session_features)}")
    print(f"• Features per session: {len(session_features.columns)}")
    
    # Step 7: Save all outputs
    save_all_outputs(G, prob_data, transition_data, session_features)
    
    print("\n✅ Complete! Ready for RCA and clustering!")

if __name__ == "__main__":
    main()