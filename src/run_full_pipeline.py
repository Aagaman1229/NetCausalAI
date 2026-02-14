"""
Complete NetCausalAI Pipeline:
1. Parse PCAPs → Detailed sessions (src/ingestion/pcap_to_sessions.py)
2. Build DAG + Features → Session features (src/casual/build_dag.py)
3. Anomaly Scoring → 0-100 scores (src/casual/anamoly_scoring.py)
4. Root Cause Analysis → Why each anomaly (src/casual/root_cause_analysis.py)
5. Behavioral Clustering → Behavior families (src/casual/behavioral_clustering.py)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add project root to path so Python can find the 'src' package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_file_exists(filepath, timeout=30):
    """Wait for file to exist with timeout"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(filepath):
            return True
        time.sleep(1)
    return False

def run_complete_analysis():
    """Execute full pipeline: Detection → RCA → Clustering"""
    
    print("\n" + "="*60)
    print("NETCAUSALAI - COMPLETE ANALYSIS PIPELINE")
    print("="*60 + "\n")
    
    processed_dir = os.path.join(PROJECT_ROOT, "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Step 1: Parse PCAPs - USE SUBPROCESS to ensure it runs properly
    print("[STEP 1/5] Parsing PCAPs...")
    
    # Check if PCAP files exist in raw directory
    raw_path = os.path.join(PROJECT_ROOT, "data/raw")
    pcap_files = [f for f in os.listdir(raw_path) if f.endswith('.pcap')] if os.path.exists(raw_path) else []
    
    if not pcap_files:
        print("   ⚠️ No PCAP files found in data/raw/")
        print("   Please add PCAP files to data/raw/ directory")
        return
    
    print(f"   Found {len(pcap_files)} PCAP files to process")
    
    # Run pcap_to_sessions.py as subprocess
    result = subprocess.run(
        ["python", "src/ingestion/pcap_to_sessions.py"],
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"   ❌ PCAP parsing failed")
        return
    
    # Verify the detailed sessions file was created
    detailed_file = os.path.join(processed_dir, "all_sessions_detailed.csv")
    if not check_file_exists(detailed_file):
        print(f"   ❌ Expected output file not found: {detailed_file}")
        print("   Please check pcap_to_sessions.py for errors")
        return
    
    print(f"   ✅ PCAP parsing completed - found {detailed_file}")
    
    # Step 2: Build DAG + Features
    print("\n[STEP 2/5] Building DAG and session features...")
    try:
        from src.casual.build_dag import main as build_main
        build_main()
    except Exception as e:
        print(f"   ❌ Error in build_dag: {e}")
        return
    
    # Verify DAG outputs
    dag_model = os.path.join(processed_dir, "dag_model_complete.pkl")
    if not os.path.exists(dag_model):
        print(f"   ❌ DAG model not created: {dag_model}")
        return
    print(f"   ✅ DAG built successfully")
    
    # Step 3: Anomaly Scoring
    print("\n[STEP 3/5] Calculating anomaly scores...")
    try:
        from src.casual.anamoly_scoring import run_anomaly_scoring_with_rca
        
        results = run_anomaly_scoring_with_rca(
            sessions_csv=detailed_file,
            dag_model=dag_model,
            output_dir=processed_dir,
            anomaly_percentile=5.0,
            generate_rca=False  # We'll do RCA separately
        )
        
        if not results.get('success', False):
            print("   ❌ Anomaly scoring failed")
            return
    except Exception as e:
        print(f"   ❌ Error in anomaly scoring: {e}")
        return
    
    print(f"   ✅ Anomaly scoring completed")
    
    # Step 4: Root Cause Analysis
    print("\n[STEP 4/5] Generating RCA explanations...")
    try:
        from src.casual.root_cause_analysis import run_rca_pipeline
        
        scores_file = os.path.join(processed_dir, "anomaly_scores_with_features.csv")
        
        rca_results = run_rca_pipeline(
            sessions_csv=detailed_file,
            scores_csv=scores_file,
            dag_model=dag_model,
            output_dir=processed_dir,
            limit=200  # Explain top 200 anomalies
        )
        
        if not rca_results.get('success', False):
            print("   ❌ RCA failed")
            return
    except Exception as e:
        print(f"   ❌ Error in RCA: {e}")
        return
    
    print(f"   ✅ RCA completed")
    
    # Step 5: Behavioral Clustering
    print("\n[STEP 5/5] Clustering behavior families...")
    try:
        from src.casual.behavioral_clustering import run_clustering_pipeline
        
        rca_file = os.path.join(processed_dir, "rca_explanations.csv")
        
        if not os.path.exists(rca_file):
            print(f"   ❌ RCA file not found: {rca_file}")
            return
        
        clustering_results = run_clustering_pipeline(
            rca_csv=rca_file,
            output_dir=processed_dir,
            min_cluster_size=5,  # Campaign threshold
            visualize=True
        )
        
        if not clustering_results.get('success', False):
            print("   ❌ Clustering failed")
            return
    except Exception as e:
        print(f"   ❌ Error in clustering: {e}")
        return
    
    print("\n" + "="*60)
    print("✅ COMPLETE PIPELINE FINISHED")
    print("="*60)
    
    # List all output files
    print("\n📁 Output Files Generated:")
    output_files = [
        "all_sessions_detailed.csv",
        "all_sessions.csv",
        "session_features_for_clustering.csv",
        "dag_model_complete.pkl",
        "dag_edges_complete.csv",
        "feature_summary.json",
        "anomaly_scores_with_features.csv",
        "rca_explanations.csv",
        "rca_summary.txt",
        "clustering_results.csv",
        "cluster_interpretations.json",
        "clustering_summary.txt",
        "cluster_visualization.png",
        "clusterer_model.pkl",
        "active_campaigns.csv"
    ]
    
    for file in output_files:
        file_path = os.path.join(processed_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  • {file:<35} ({size:.1f} KB)")
        else:
            print(f"  • {file:<35} (❌ missing)")

if __name__ == "__main__":
    run_complete_analysis()