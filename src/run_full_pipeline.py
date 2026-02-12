"""
Complete NetCausalAI Pipeline:
1. Parse PCAPs → Detailed sessions
2. Build DAG + Features → Session features
3. Anomaly Scoring → 0-100 scores
4. Root Cause Analysis → Why each anomaly
5. Behavioral Clustering → Behavior families
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

def run_complete_analysis():
    """Execute full pipeline: Detection → RCA → Clustering"""
    
    print("\n" + "="*60)
    print("NETCAUSALAI - COMPLETE ANALYSIS PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Parse PCAPs
    print("[STEP 1/5] Parsing PCAPs...")
    from src.casual.parse_pcap import main as parse_main
    parse_main()
    
    # Step 2: Build DAG + Features
    print("\n[STEP 2/5] Building DAG and session features...")
    from src.casual.build_dag import main as build_main
    build_main()
    
    # Step 3: Anomaly Scoring
    print("\n[STEP 3/5] Calculating anomaly scores...")
    from src.casual.anomaly_scoring import run_anomaly_scoring_with_rca
    run_anomaly_scoring_with_rca(
        sessions_csv="data/processed/all_sessions_detailed.csv",
        dag_model="data/processed/dag_model_complete.pkl",
        generate_rca=False  # We'll do RCA separately with full features
    )
    
    # Step 4: Root Cause Analysis
    print("\n[STEP 4/5] Generating RCA explanations...")
    from src.casual.root_cause_analysis import run_rca_pipeline
    rca_results = run_rca_pipeline(
        sessions_csv="data/processed/all_sessions_detailed.csv",
        scores_csv="data/processed/anomaly_scores_with_features.csv",
        dag_model="data/processed/dag_model_complete.pkl",
        limit=200  # Explain top 200 anomalies
    )
    
    # Step 5: Behavioral Clustering
    print("\n[STEP 5/5] Clustering behavior families...")
    from src.casual.behavioral_clustering import run_clustering_pipeline
    clustering_results = run_clustering_pipeline(
        rca_csv="data/processed/rca_explanations.csv",
        output_dir="data/processed",
        min_cluster_size=5,  # Campaign threshold
        visualize=True
    )
    
    print("\n" + "="*60)
    print("✅ COMPLETE PIPELINE FINISHED")
    print("="*60)
    print("\n📁 Output Files:")
    print("  • data/processed/all_sessions_detailed.csv - Raw session data")
    print("  • data/processed/session_features_for_clustering.csv - 25+ features per session")
    print("  • data/processed/dag_model_complete.pkl - Full Markov model")
    print("  • data/processed/anomaly_scores_with_features.csv - 0-100 scores")
    print("  • data/processed/rca_explanations.csv - Root cause analysis")
    print("  • data/processed/rca_summary.txt - Human-readable RCA summary")
    print("  • data/processed/clustering_results.csv - Sessions with cluster IDs")
    print("  • data/processed/cluster_interpretations.json - Behavior family profiles")
    print("  • data/processed/clustering_summary.txt - Human-readable cluster summary")
    print("  • data/processed/cluster_visualization.png - 2D cluster visualization")
    print("  • data/processed/clusterer_model.pkl - Trained HDBSCAN model")
    print("  • data/processed/active_campaigns.csv - Ongoing campaigns (>5 sessions)")

if __name__ == "__main__":
    run_complete_analysis()