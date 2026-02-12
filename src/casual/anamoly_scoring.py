"""
Markov Chain Anomaly Scoring with Root Cause Analysis
Converts log probabilities to 0-100 anomaly scores + pinpoints exact causes
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

class MarkovAnomalyScorer:
    """
    Markov Chain-based anomaly scoring with RCA capabilities
    """
    
    def __init__(self, 
                 dag_model_path: str = "data/processed/dag_model_complete.pkl",
                 session_features_path: str = "data/processed/session_features_for_clustering.csv",
                 min_prob: float = 0.001,
                 anomaly_percentile: float = 5.0):
        """
        Initialize the anomaly scorer with RCA support
        """
        self.dag_model_path = dag_model_path
        self.session_features_path = session_features_path
        self.min_prob = min_prob
        self.anomaly_percentile = anomaly_percentile
        
        # Load DAG model
        self._load_dag_model()
        
        # Load session features if available
        self._load_session_features()
        
        print(f"✅ MarkovAnomalyScorer initialized with RCA")
        print(f"   • DAG nodes: {len(self.graph_nodes)}")
        print(f"   • DAG transitions: {len(self.transition_probs)}")
        print(f"   • Session features: {len(self.session_features) if hasattr(self, 'session_features') else 0}")
        print(f"   • Anomaly threshold: Bottom {anomaly_percentile}%")
    
    def _load_dag_model(self):
        """Load the DAG model from pickle file"""
        try:
            with open(self.dag_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.transition_probs = model_data['probabilities']
            self.confidence_scores = model_data.get('confidence_scores', {})
            self.transition_counts = model_data.get('transition_counts', {})
            self.graph = model_data['graph']
            self.graph_nodes = list(self.graph.nodes())
            
        except FileNotFoundError:
            # Fallback to old model
            fallback = "data/processed/dag_model.pkl"
            print(f"⚠️ Complete model not found, trying {fallback}")
            with open(fallback, 'rb') as f:
                model_data = pickle.load(f)
            self.transition_probs = model_data['probabilities']
            self.graph = model_data['graph']
            self.graph_nodes = list(self.graph.nodes())
            self.confidence_scores = {}
            self.transition_counts = {}
    
    def _load_session_features(self):
        """Load pre-computed session features"""
        try:
            self.session_features = pd.read_csv(self.session_features_path)
            print(f"   Loaded {len(self.session_features)} pre-computed session features")
        except:
            self.session_features = pd.DataFrame()
            print(f"   No pre-computed session features found")
    
    def _get_transition_prob(self, src: str, dst: str) -> float:
        """Get probability of transition src -> dst"""
        return self.transition_probs.get((src, dst), self.min_prob)
    
    # ============ NEW: ROOT CAUSE ANALYSIS METHODS ============
    
    def explain_session(self, session_id: str, events: List[str] = None) -> Dict[str, Any]:
        """
        Perform Root Cause Analysis for a specific session
        
        Args:
            session_id: Session identifier
            events: Optional list of events (if not provided, will try to load from features)
            
        Returns:
            Dictionary with RCA explanation
        """
        # Get events if not provided
        if events is None and hasattr(self, 'session_features'):
            # Try to extract from session features
            session_row = self.session_features[self.session_features['session_id'] == session_id]
            if len(session_row) > 0 and 'events' in session_row.columns:
                events = eval(session_row.iloc[0]['events'])
        
        if events is None or len(events) < 2:
            return {"error": "Cannot explain: insufficient events"}
        
        # Step 1: Calculate per-transition log probabilities
        transitions = []
        log_probs = []
        
        for i in range(len(events) - 1):
            src, dst = events[i], events[i+1]
            prob = self._get_transition_prob(src, dst)
            log_prob = np.log(prob)
            
            transitions.append({
                'position': i,
                'from': src,
                'to': dst,
                'probability': prob,
                'log_probability': log_prob,
                'is_unseen': prob == self.min_prob,
                'is_rare': prob < 0.01 and prob != self.min_prob,
                'confidence': self.confidence_scores.get((src, dst), 0)
            })
            log_probs.append(log_prob)
        
        # Step 2: Find most anomalous transitions
        log_probs_array = np.array(log_probs)
        total_abs_sum = np.sum(np.abs(log_probs_array))
        
        for trans in transitions:
            # Calculate contribution percentage
            trans['contribution'] = np.abs(trans['log_probability']) / total_abs_sum if total_abs_sum > 0 else 0
        
        # Sort by contribution (most negative log prob = most anomalous)
        sorted_transitions = sorted(transitions, key=lambda x: x['log_probability'])[:5]
        
        # Step 3: Find divergence point
        cumulative_log = np.cumsum(log_probs)
        
        # Find where cumulative score drops significantly
        divergence_idx = 0
        if len(cumulative_log) > 1:
            # Simple heuristic: first drop below median of first half
            first_half = cumulative_log[:len(cumulative_log)//2]
            if len(first_half) > 0:
                median_first_half = np.median(first_half)
                for i, val in enumerate(cumulative_log):
                    if val < median_first_half - 1.0:  # 1 nat drop
                        divergence_idx = i
                        break
        
        # Step 4: Generate explanation
        primary_cause = sorted_transitions[0] if sorted_transitions else None
        
        # Determine anomaly driver
        unseen_count = sum(1 for t in transitions if t['is_unseen'])
        rare_count = sum(1 for t in transitions if t['is_rare'])
        
        if unseen_count > 0:
            driver = "unseen_transition"
            driver_description = f"Sequence contains {unseen_count} transition(s) never seen in training"
        elif rare_count > 2:
            driver = "rare_transitions"
            driver_description = f"Sequence contains {rare_count} rare transition(s) (probability < 0.01)"
        elif np.mean(log_probs) < -5:
            driver = "sustained_low_probability"
            driver_description = "Consistently low transition probabilities throughout session"
        else:
            driver = "single_anomalous_transition"
            driver_description = f"Highly anomalous transition at position {primary_cause['position'] if primary_cause else 0}"
        
        # Step 5: Generate hypothesis
        hypothesis = self._generate_hypothesis(events, transitions, primary_cause, driver)
        
        explanation = {
            'session_id': session_id,
            'event_count': len(events),
            'anomaly_score': self._calculate_log_score(events),  # Raw log score
            
            # RCA - WHAT caused it
            'primary_cause': {
                'position': primary_cause['position'] if primary_cause else None,
                'from_event': primary_cause['from'] if primary_cause else None,
                'to_event': primary_cause['to'] if primary_cause else None,
                'probability': primary_cause['probability'] if primary_cause else None,
                'contribution': primary_cause['contribution'] if primary_cause else None,
                'is_unseen': primary_cause['is_unseen'] if primary_cause else False,
                'is_rare': primary_cause['is_rare'] if primary_cause else False
            },
            
            # RCA - WHEN it started
            'divergence': {
                'start_position': divergence_idx,
                'start_event_index': divergence_idx + 1,  # +1 because transitions
                'cumulative_score_at_start': cumulative_log[divergence_idx] if divergence_idx < len(cumulative_log) else 0,
                'cumulative_scores': cumulative_log.tolist()
            },
            
            # RCA - TYPE of anomaly
            'anomaly_driver': driver,
            'driver_description': driver_description,
            'unseen_transitions_count': unseen_count,
            'rare_transitions_count': rare_count,
            'avg_log_probability': float(np.mean(log_probs)),
            'min_log_probability': float(np.min(log_probs)),
            
            # Hypothesis
            'hypothesis': hypothesis,
            
            # Top contributors
            'top_anomalous_transitions': [
                {
                    'position': t['position'],
                    'from': t['from'],
                    'to': t['to'],
                    'probability': t['probability'],
                    'contribution': t['contribution']
                }
                for t in sorted_transitions[:3]
            ],
            
            # Full transition details (for debugging)
            'all_transitions': transitions
        }
        
        return explanation
    
    def _generate_hypothesis(self, events: List[str], transitions: List[Dict], 
                           primary_cause: Dict, driver: str) -> str:
        """Generate natural language hypothesis about the anomaly"""
        
        # Check for specific patterns
        if len(events) > 100 and events.count('LARGE_TRANSFER') > len(events) * 0.5:
            return "Likely data exfiltration: Very long session with >50% LARGE_TRANSFER events"
        
        if 'UDP_PACKET' in events and events.count('UDP_PACKET') > 100:
            return "Possible UDP flood attack: High volume of UDP packets"
        
        if 'TCP_SYN' in events and 'TCP_HANDSHAKE' not in events[:10]:
            return "Potential SYN scan or incomplete handshake: SYNs without completions"
        
        if primary_cause and primary_cause.get('is_unseen'):
            return f"Novel behavior: Transition {primary_cause['from']}→{primary_cause['to']} never seen in training"
        
        if primary_cause and primary_cause.get('probability', 1) < 0.001:
            return f"Highly unusual transition at step {primary_cause['position']}: {primary_cause['from']}→{primary_cause['to']} (probability={primary_cause['probability']:.6f})"
        
        if driver == "sustained_low_probability":
            return "Session consistently uses unusual event sequences throughout"
        
        return "Unusual session pattern detected - investigate further"
    
    def explain_top_anomalies(self, scores_df: pd.DataFrame, 
                            sessions_df: pd.DataFrame = None,
                            n: int = 10) -> pd.DataFrame:
        """
        Generate RCA explanations for top N anomalies
        
        Args:
            scores_df: DataFrame with anomaly scores
            sessions_df: DataFrame with raw events
            n: Number of top anomalies to explain
            
        Returns:
            DataFrame with RCA explanations
        """
        print(f"\n🔍 Generating RCA explanations for top {n} anomalies...")
        
        # Get top anomalies
        top_anomalies = scores_df[scores_df['is_anomaly']].head(n)
        
        explanations = []
        
        for idx, row in top_anomalies.iterrows():
            session_id = row['session_id']
            
            # Get events for this session
            events = None
            if sessions_df is not None:
                session_events = sessions_df[sessions_df['session_id'] == session_id]
                if len(session_events) > 0:
                    events = session_events['event'].tolist()
            
            # Get explanation
            explanation = self.explain_session(session_id, events)
            
            explanations.append({
                'session_id': session_id,
                'anomaly_score': row.get('anomaly_score_0_100', 0),
                'severity': row.get('severity', 'UNKNOWN'),
                'primary_cause_position': explanation.get('primary_cause', {}).get('position'),
                'primary_cause_from': explanation.get('primary_cause', {}).get('from_event'),
                'primary_cause_to': explanation.get('primary_cause', {}).get('to_event'),
                'primary_cause_probability': explanation.get('primary_cause', {}).get('probability'),
                'divergence_start': explanation.get('divergence', {}).get('start_position'),
                'anomaly_driver': explanation.get('anomaly_driver'),
                'unseen_transitions': explanation.get('unseen_transitions_count'),
                'rare_transitions': explanation.get('rare_transitions_count'),
                'hypothesis': explanation.get('hypothesis'),
                'avg_log_prob': explanation.get('avg_log_probability')
            })
        
        rca_df = pd.DataFrame(explanations)
        print(f"✅ Generated explanations for {len(rca_df)} sessions")
        
        return rca_df
    
    # ============ EXISTING METHODS (UPDATED) ============
    
    def _calculate_log_score(self, events: List[str]) -> float:
        """Calculate raw log probability score"""
        if len(events) < 2:
            return 0.0
        
        log_score = 0.0
        for i in range(len(events) - 1):
            prob = self._get_transition_prob(events[i], events[i+1])
            log_score += np.log(prob)
        return log_score
    
    def calculate_session_metrics(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics with RCA-ready features"""
        print("🔍 Calculating anomaly scores with RCA features...")
        
        results = []
        log_scores = []
        session_data = []
        
        for session_id, group in sessions_df.groupby('session_id'):
            events = group['event'].tolist()
            
            if len(events) < 2:
                continue
            
            # Calculate raw log score
            log_score = self._calculate_log_score(events)
            log_scores.append(log_score)
            
            # Get session features if available
            session_features = {}
            if hasattr(self, 'session_features') and len(self.session_features) > 0:
                feat_row = self.session_features[self.session_features['session_id'] == session_id]
                if len(feat_row) > 0:
                    session_features = feat_row.iloc[0].to_dict()
            
            session_data.append({
                'session_id': session_id,
                'events': events,
                'log_score': log_score,
                'event_count': len(events),
                'timestamp_start': group['timestamp'].min(),
                'timestamp_end': group['timestamp'].max(),
                **session_features  # Include pre-computed features
            })
        
        # Normalization
        min_log = min(log_scores) if log_scores else 0.0
        max_log = max(log_scores) if log_scores else 0.0
        
        for session in session_data:
            anomaly_score = self._normalize_to_0_100(
                session['log_score'], min_log, max_log
            )
            
            results.append({
                'session_id': session['session_id'],
                'event_count': session['event_count'],
                'log_score': session['log_score'],
                'anomaly_score_0_100': anomaly_score,
                'duration': session['timestamp_end'] - session['timestamp_start'],
                
                # Include key RCA features
                'avg_transition_prob': session.get('avg_transition_prob', 0),
                'min_transition_prob': session.get('min_transition_prob', 0),
                'number_unseen_transitions': session.get('number_unseen_transitions', 0),
                'number_rare_transitions': session.get('number_rare_transitions', 0),
                'repetition_rate': session.get('repetition_rate', 0),
                'entropy': session.get('entropy_of_transition_probs', 0),
                'bytes_transferred': session.get('bytes_transferred', 0),
                'ratio_large_transfer': session.get('ratio_large_transfer_events', 0),
                'burstiness': session.get('burstiness', 0),
                
                # Raw events for RCA
                'events': str(session['events'])
            })
        
        return pd.DataFrame(results)
    
    def _normalize_to_0_100(self, log_score: float, min_log: float, max_log: float) -> float:
        """Normalize log score to 0-100 scale"""
        if min_log == max_log:
            return 50.0
        normalized = 100 * (log_score - max_log) / (min_log - max_log)
        return max(0.0, min(100.0, normalized))
    
    def detect_anomalies(self, scores_df: pd.DataFrame, method: str = 'percentile', 
                        custom_threshold: Optional[float] = None) -> pd.DataFrame:
        """Detect anomalies with severity classification"""
        scores_df = scores_df.copy()
        
        if method == 'percentile':
            anomaly_threshold = np.percentile(
                scores_df['anomaly_score_0_100'], 
                100 - self.anomaly_percentile
            )
            scores_df['is_anomaly'] = scores_df['anomaly_score_0_100'] >= anomaly_threshold
            scores_df['anomaly_threshold'] = anomaly_threshold
        elif method == 'threshold' and custom_threshold is not None:
            scores_df['is_anomaly'] = scores_df['anomaly_score_0_100'] >= custom_threshold
            scores_df['anomaly_threshold'] = custom_threshold
        
        # Severity levels
        scores_df['severity'] = pd.cut(
            scores_df['anomaly_score_0_100'],
            bins=[0, 70, 85, 95, 100],
            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            include_lowest=True
        )
        
        scores_df['confidence'] = np.clip(
            (scores_df['anomaly_score_0_100'] - scores_df['anomaly_threshold']) / 
            (100 - scores_df['anomaly_threshold']), 0, 1
        )
        
        return scores_df.sort_values('anomaly_score_0_100', ascending=False)
    
    def save_rca_results(self, rca_df: pd.DataFrame, output_dir: str = "data/processed"):
        """Save RCA explanations to file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save RCA explanations
        rca_path = f"{output_dir}/rca_explanations.csv"
        rca_df.to_csv(rca_path, index=False)
        print(f"💾 Saved RCA explanations to: {rca_path}")
        
        # Save as JSON for dashboard
        rca_json = rca_df.to_dict(orient='records')
        json_path = f"{output_dir}/rca_explanations.json"
        with open(json_path, 'w') as f:
            json.dump(rca_json, f, indent=2, default=str)
        print(f"💾 Saved RCA JSON to: {json_path}")


# -------------------------------------------------------------------
# Main execution function
# -------------------------------------------------------------------

def run_anomaly_scoring_with_rca(
    sessions_csv: str = "data/processed/all_sessions_detailed.csv",
    dag_model: str = "data/processed/dag_model_complete.pkl",
    output_dir: str = "data/processed",
    anomaly_percentile: float = 5.0,
    generate_rca: bool = True
) -> Dict:
    """
    Main function to run anomaly scoring with RCA
    """
    print("=" * 60)
    print("NETCAUSALAI - ANOMALY SCORING WITH ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    try:
        # 1. Initialize scorer
        print("\n[1/5] Initializing Markov anomaly scorer with RCA...")
        scorer = MarkovAnomalyScorer(
            dag_model_path=dag_model,
            anomaly_percentile=anomaly_percentile
        )
        
        # 2. Load sessions data
        print(f"\n[2/5] Loading sessions...")
        if not Path(sessions_csv).exists():
            # Fallback to lightweight
            sessions_csv = "data/processed/all_sessions.csv"
        
        sessions_df = pd.read_csv(sessions_csv)
        print(f"   • Loaded {len(sessions_df)} events")
        print(f"   • Unique sessions: {sessions_df['session_id'].nunique()}")
        
        # 3. Calculate anomaly scores
        print(f"\n[3/5] Calculating 0-100 anomaly scores...")
        scores_df = scorer.calculate_session_metrics(sessions_df)
        
        # 4. Detect anomalies
        print(f"\n[4/5] Detecting anomalies...")
        anomalies_df = scorer.detect_anomalies(scores_df, method='percentile')
        
        # 5. Generate RCA explanations
        if generate_rca:
            print(f"\n[5/5] Generating Root Cause Analysis...")
            rca_df = scorer.explain_top_anomalies(anomalies_df, sessions_df, n=20)
            scorer.save_rca_results(rca_df, output_dir)
        
        # Save main results
        anomalies_df.to_csv(f"{output_dir}/anomaly_scores_with_features.csv", index=False)
        
        # Summary
        n_anomalies = anomalies_df['is_anomaly'].sum()
        print(f"\n✅ Complete! Detected {n_anomalies} anomalies")
        print(f"   • RCA generated for top 20 anomalies")
        print(f"   • Full results: {output_dir}/anomaly_scores_with_features.csv")
        print(f"   • RCA explanations: {output_dir}/rca_explanations.csv")
        
        return {
            'scorer': scorer,
            'scores_df': scores_df,
            'anomalies_df': anomalies_df,
            'rca_df': rca_df if generate_rca else None,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NetCausalAI Anomaly Scoring with RCA')
    parser.add_argument('--sessions', type=str, default='data/processed/all_sessions_detailed.csv')
    parser.add_argument('--dag-model', type=str, default='data/processed/dag_model_complete.pkl')
    parser.add_argument('--percentile', type=float, default=5.0)
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--no-rca', action='store_true', help='Skip RCA generation')
    
    args = parser.parse_args()
    
    results = run_anomaly_scoring_with_rca(
        sessions_csv=args.sessions,
        dag_model=args.dag_model,
        output_dir=args.output_dir,
        anomaly_percentile=args.percentile,
        generate_rca=not args.no_rca
    )
    
    exit(0 if results.get('success', False) else 1)