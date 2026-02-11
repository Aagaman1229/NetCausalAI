# src/casual/anomaly_scoring.py
"""
Markov Chain Anomaly Scoring for NetCausalAI
Converts log probabilities to 0-100 anomaly scores for human interpretation
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class MarkovAnomalyScorer:
    """
    Markov Chain-based anomaly scoring with 0-100 normalized scores
    """
    
    def __init__(self, 
                 dag_model_path: str = "data/processed/dag_model.pkl",
                 min_prob: float = 0.001,
                 anomaly_percentile: float = 5.0):
        """
        Initialize the anomaly scorer
        
        Args:
            dag_model_path: Path to saved DAG model
            min_prob: Minimum probability for unseen transitions
            anomaly_percentile: Bottom percentile to flag as anomalies
        """
        self.dag_model_path = dag_model_path
        self.min_prob = min_prob
        self.anomaly_percentile = anomaly_percentile
        
        # Load DAG model
        self._load_dag_model()
        
        print(f"✅ MarkovAnomalyScorer initialized")
        print(f"   • DAG nodes: {len(self.graph_nodes)}")
        print(f"   • DAG transitions: {len(self.transition_probs)}")
        print(f"   • Anomaly threshold: Bottom {anomaly_percentile}%")
    
    def _load_dag_model(self):
        """Load the DAG model from pickle file"""
        try:
            with open(self.dag_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.transition_probs = model_data['probabilities']
            self.graph = model_data['graph']
            self.graph_nodes = list(self.graph.nodes())
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"DAG model not found at {self.dag_model_path}. "
                f"Please run build_dag.py first."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load DAG model: {str(e)}")
    
    def _get_transition_prob(self, src: str, dst: str) -> float:
        """
        Get probability of transition src -> dst
        Returns min_prob for unseen transitions
        """
        return self.transition_probs.get((src, dst), self.min_prob)
    
    def _calculate_log_score(self, events: List[str]) -> float:
        """
        Calculate raw log probability score
        
        Args:
            events: List of event strings
            
        Returns:
            Log probability of sequence
        """
        if len(events) < 2:
            return 0.0
        
        log_score = 0.0
        for i in range(len(events) - 1):
            prob = self._get_transition_prob(events[i], events[i+1])
            log_score += np.log(prob)
        return log_score
    
    def _normalize_to_0_100(self, log_score: float, 
                           min_log: float, max_log: float) -> float:
        """
        Normalize log score to 0-100 scale
        
        Args:
            log_score: Raw log probability score
            min_log: Minimum log score in dataset
            max_log: Maximum log score in dataset
            
        Returns:
            Normalized score 0-100 (0 = most normal, 100 = most anomalous)
        """
        # Handle edge cases
        if min_log == max_log:
            return 50.0  # Middle value
        
        # Invert: low log score = high anomaly score
        # Scale to 0-100 range
        normalized = 100 * (log_score - max_log) / (min_log - max_log)
        
        # Clip to 0-100 range
        return max(0.0, min(100.0, normalized))
    
    def calculate_session_metrics(self, 
                                  sessions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics for all sessions with normalized 0-100 scores
        
        Args:
            sessions_df: DataFrame with columns ['session_id', 'event', 'timestamp']
            
        Returns:
            DataFrame with anomaly scores and metrics
        """
        print("🔍 Calculating anomaly scores for all sessions...")
        
        results = []
        total_sessions = sessions_df['session_id'].nunique()
        
        # First pass: calculate raw log scores
        log_scores = []
        session_data = []
        
        for session_id, group in sessions_df.groupby('session_id'):
            events = group['event'].tolist()
            
            # Skip very short sessions
            if len(events) < 2:
                continue
            
            # Calculate raw log score
            log_score = self._calculate_log_score(events)
            log_scores.append(log_score)
            
            session_data.append({
                'session_id': session_id,
                'events': events,
                'log_score': log_score,
                'event_count': len(events),
                'timestamp_start': group['timestamp'].min(),
                'timestamp_end': group['timestamp'].max()
            })
        
        # Calculate min/max for normalization
        min_log = min(log_scores) if log_scores else 0.0
        max_log = max(log_scores) if log_scores else 0.0
        
        print(f"   Log score range: [{min_log:.2f}, {max_log:.2f}]")
        
        # Second pass: calculate normalized scores and additional metrics
        for idx, session in enumerate(session_data):
            # Calculate normalized 0-100 score
            anomaly_score = self._normalize_to_0_100(
                session['log_score'], min_log, max_log
            )
            
            # Calculate additional metrics
            metrics = self._calculate_additional_metrics(session['events'])
            
            results.append({
                'session_id': session['session_id'],
                'event_count': session['event_count'],
                'log_score': session['log_score'],
                'anomaly_score_0_100': anomaly_score,
                **metrics,
                'events': str(session['events']),
                'duration': session['timestamp_end'] - session['timestamp_start']
            })
            
            # Progress indicator
            if (idx + 1) % 100 == 0 or (idx + 1) == len(session_data):
                print(f"   Processed {idx + 1}/{len(session_data)} sessions...", end='\r')
        
        print(f"\n✅ Scored {len(results)} sessions")
        
        return pd.DataFrame(results)
    
    def _calculate_additional_metrics(self, events: List[str]) -> Dict:
        """
        Calculate additional session metrics
        
        Args:
            events: List of events
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'avg_transition_prob': 0.0,
            'min_transition_prob': 1.0,
            'rare_transitions': 0,
            'unseen_transitions': 0,
            'entropy': 0.0,
            'unique_events': len(set(events)),
            'repetition_rate': 0.0
        }
        
        if len(events) < 2:
            return metrics
        
        # Calculate repetition rate
        if len(events) > 0:
            metrics['repetition_rate'] = 1 - (metrics['unique_events'] / len(events))
        
        transition_probs = []
        for i in range(len(events) - 1):
            prob = self._get_transition_prob(events[i], events[i+1])
            transition_probs.append(prob)
            
            if prob < 0.01:  # Rare transition
                metrics['rare_transitions'] += 1
            
            if prob == self.min_prob:  # Unseen transition
                metrics['unseen_transitions'] += 1
        
        if transition_probs:
            metrics['avg_transition_prob'] = np.mean(transition_probs)
            metrics['min_transition_prob'] = np.min(transition_probs)
            
            # Calculate entropy (uncertainty)
            probs_array = np.array(transition_probs)
            probs_array = probs_array[probs_array > 0]  # Remove zeros
            if len(probs_array) > 0:
                metrics['entropy'] = -np.sum(probs_array * np.log(probs_array))
        
        return metrics
    
    def detect_anomalies(self, 
                         scores_df: pd.DataFrame,
                         method: str = 'percentile',
                         custom_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Detect anomalies using 0-100 scores
        
        Args:
            scores_df: DataFrame with scores from calculate_session_metrics()
            method: 'percentile' or 'threshold'
            custom_threshold: Custom threshold value (if method='threshold')
            
        Returns:
            DataFrame with anomaly flags
        """
        print("\n🚨 Detecting anomalies...")
        
        scores_df = scores_df.copy()
        
        if method == 'percentile':
            # Use percentile-based threshold on 0-100 scores
            anomaly_threshold = np.percentile(
                scores_df['anomaly_score_0_100'], 
                100 - self.anomaly_percentile
            )
            scores_df['is_anomaly'] = scores_df['anomaly_score_0_100'] >= anomaly_threshold
            scores_df['anomaly_threshold'] = anomaly_threshold
            
            print(f"   Method: Percentile (top {self.anomaly_percentile}% of anomaly scores)")
            print(f"   Threshold (0-100): {anomaly_threshold:.1f}")
            
        elif method == 'threshold' and custom_threshold is not None:
            # Use custom threshold
            scores_df['is_anomaly'] = scores_df['anomaly_score_0_100'] >= custom_threshold
            scores_df['anomaly_threshold'] = custom_threshold
            
            print(f"   Method: Custom threshold")
            print(f"   Threshold (0-100): {custom_threshold:.1f}")
        
        else:
            raise ValueError("Invalid method or threshold")
        
        # Add severity levels based on anomaly score
        scores_df['severity'] = pd.cut(
            scores_df['anomaly_score_0_100'],
            bins=[0, 70, 85, 95, 100],
            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            include_lowest=True
        )
        
        # Calculate confidence score (how far above threshold)
        scores_df['confidence'] = np.clip(
            (scores_df['anomaly_score_0_100'] - scores_df['anomaly_threshold']) / 
            (100 - scores_df['anomaly_threshold']), 0, 1
        )
        
        # Sort by anomaly score (most anomalous first)
        scores_df = scores_df.sort_values('anomaly_score_0_100', ascending=False)
        
        # Print summary
        n_anomalies = scores_df['is_anomaly'].sum()
        n_total = len(scores_df)
        
        print(f"   Anomalies detected: {n_anomalies}/{n_total} ({n_anomalies/n_total*100:.1f}%)")
        
        # Print severity breakdown
        if n_anomalies > 0:
            print(f"   Severity breakdown:")
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = scores_df[scores_df['is_anomaly'] & (scores_df['severity'] == severity)].shape[0]
                if count > 0:
                    print(f"     • {severity}: {count} sessions")
        
        return scores_df
    
    def categorize_anomalies(self, anomalies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize anomalies into attack types
        
        Args:
            anomalies_df: DataFrame with anomaly sessions
            
        Returns:
            DataFrame with added categories
        """
        print("\n🔍 Categorizing anomalies...")
        
        anomalies_df = anomalies_df.copy()
        categories = []
        
        for _, row in anomalies_df.iterrows():
            events = eval(row['events']) if isinstance(row['events'], str) else row['events']
            
            # Check for common attack patterns
            if len(events) >= 10:
                # Check for flood patterns
                first_10 = events[:10]
                if all(e == 'UDP_PACKET' for e in first_10):
                    category = "UDP_FLOOD"
                elif all(e == 'TCP_SYN' for e in first_10):
                    category = "SYN_FLOOD"
                elif all(e == 'LARGE_TRANSFER' for e in first_10):
                    category = "DATA_EXFILTRATION"
                elif len(set(events)) <= 2:  # Very repetitive
                    category = "REPETITIVE_ATTACK"
                elif row['event_count'] > 1000:  # Extremely long
                    category = "LONG_SESSION_ATTACK"
                else:
                    category = "UNUSUAL_PATTERN"
            else:
                # Short sequences
                if 'UDP_PACKET' in events and events.count('UDP_PACKET') > len(events) * 0.8:
                    category = "UDP_FLOOD"
                elif 'LARGE_TRANSFER' in events and events.count('LARGE_TRANSFER') > len(events) * 0.8:
                    category = "DATA_EXFILTRATION"
                else:
                    category = "SHORT_SUSPICIOUS"
            
            categories.append(category)
        
        anomalies_df['attack_category'] = categories
        return anomalies_df
    
    def analyze_anomaly_patterns(self, 
                                 anomalies_df: pd.DataFrame,
                                 top_n: int = 10) -> Dict:
        """
        Analyze patterns in detected anomalies
        
        Args:
            anomalies_df: DataFrame with anomaly sessions
            top_n: Number of top patterns to return
            
        Returns:
            Dictionary with pattern analysis
        """
        print("\n📊 Analyzing anomaly patterns...")
        
        if len(anomalies_df) == 0:
            return {"error": "No anomalies to analyze"}
        
        patterns = {
            'most_common_events': defaultdict(int),
            'most_common_transitions': defaultdict(int),
            'attack_categories': defaultdict(int),
            'severity_counts': defaultdict(int)
        }
        
        for _, row in anomalies_df.iterrows():
            events = eval(row['events']) if isinstance(row['events'], str) else row['events']
            
            # Count events
            for event in events:
                patterns['most_common_events'][event] += 1
            
            # Count transitions
            for i in range(len(events) - 1):
                transition = f"{events[i]}→{events[i+1]}"
                patterns['most_common_transitions'][transition] += 1
            
            # Track categories and severity
            patterns['attack_categories'][row.get('attack_category', 'UNKNOWN')] += 1
            patterns['severity_counts'][str(row.get('severity', 'UNKNOWN'))] += 1
        
        # Convert to sorted lists
        analysis = {
            'total_anomalies': len(anomalies_df),
            'attack_category_distribution': dict(patterns['attack_categories']),
            'severity_distribution': dict(patterns['severity_counts']),
            'top_events': sorted(
                patterns['most_common_events'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n],
            'top_transitions': sorted(
                patterns['most_common_transitions'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n],
            'summary_stats': {
                'avg_anomaly_score': anomalies_df['anomaly_score_0_100'].mean(),
                'avg_event_count': anomalies_df['event_count'].mean(),
                'avg_rare_transitions': anomalies_df['rare_transitions'].mean(),
                'critical_count': (anomalies_df['severity'] == 'CRITICAL').sum()
            }
        }
        
        return analysis
    
    def save_results(self,
                     scores_df: pd.DataFrame,
                     analysis: Dict,
                     output_dir: str = "data/processed") -> None:
        """
        Save all results to files
        
        Args:
            scores_df: DataFrame with scores and anomaly flags
            analysis: Dictionary with pattern analysis
            output_dir: Directory to save results
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save scores (CSV)
        scores_path = f"{output_dir}/anomaly_scores_0_100.csv"
        scores_df.to_csv(scores_path, index=False)
        print(f"💾 Saved anomaly scores to: {scores_path}")
        
        # Save analysis (JSON)
        analysis_path = f"{output_dir}/anomaly_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"💾 Saved pattern analysis to: {analysis_path}")
        
        # Save summary (TXT)
        summary = self._create_summary(scores_df, analysis)
        summary_path = f"{output_dir}/anomaly_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"💾 Saved summary to: {summary_path}")
        
        # Save top anomalies separately (CSV)
        top_anomalies = scores_df[scores_df['is_anomaly']].head(50)
        top_path = f"{output_dir}/top_anomalies.csv"
        top_anomalies.to_csv(top_path, index=False)
        print(f"💾 Saved top 50 anomalies to: {top_path}")
    
    def _create_summary(self, scores_df: pd.DataFrame, analysis: Dict) -> str:
        """Create a human-readable summary"""
        n_anomalies = scores_df['is_anomaly'].sum()
        n_total = len(scores_df)
        threshold = scores_df['anomaly_threshold'].iloc[0]
        
        summary_lines = [
            "=" * 60,
            "NETCAUSALAI - ANOMALY DETECTION SUMMARY (0-100 SCORES)",
            "=" * 60,
            f"\n📊 OVERVIEW:",
            f"   • Total sessions analyzed: {n_total}",
            f"   • Anomalies detected: {n_anomalies}",
            f"   • Anomaly rate: {n_anomalies/n_total*100:.1f}%",
            f"   • Detection method: Top {self.anomaly_percentile}% of anomaly scores",
            f"   • Threshold (0-100 scale): {threshold:.1f}",
            f"\n📈 SCORE STATISTICS:",
            f"   • Minimum anomaly score: {scores_df['anomaly_score_0_100'].min():.1f}",
            f"   • Maximum anomaly score: {scores_df['anomaly_score_0_100'].max():.1f}",
            f"   • Average anomaly score: {scores_df['anomaly_score_0_100'].mean():.1f}",
            f"   • Median anomaly score: {scores_df['anomaly_score_0_100'].median():.1f}",
        ]
        
        # Add severity breakdown
        if 'severity' in scores_df.columns and n_anomalies > 0:
            anomalies = scores_df[scores_df['is_anomaly']]
            summary_lines.append(f"\n⚠️  SEVERITY BREAKDOWN:")
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = (anomalies['severity'] == severity).sum()
                if count > 0:
                    percentage = count / n_anomalies * 100
                    summary_lines.append(f"   • {severity}: {count} sessions ({percentage:.1f}%)")
        
        # Add top anomalies
        if n_anomalies > 0:
            summary_lines.append(f"\n🔴 TOP 5 CRITICAL ANOMALIES:")
            top_critical = scores_df[
                (scores_df['is_anomaly']) & 
                (scores_df['severity'] == 'CRITICAL')
            ].head(5)
            
            for idx, (_, row) in enumerate(top_critical.iterrows(), 1):
                events = eval(row['events']) if isinstance(row['events'], str) else row['events']
                category = row.get('attack_category', 'Unknown')
                summary_lines.append(f"\n   {idx}. Session {row['session_id'][:40]}...")
                summary_lines.append(f"      • Anomaly Score: {row['anomaly_score_0_100']:.1f}/100")
                summary_lines.append(f"      • Severity: {row.get('severity', 'Unknown')}")
                summary_lines.append(f"      • Category: {category}")
                summary_lines.append(f"      • Events: {len(events)} total")
                summary_lines.append(f"      • Pattern: {' → '.join(events[:3])}{'...' if len(events) > 3 else ''}")
        
        # Add attack category breakdown
        if 'attack_categories' in analysis:
            summary_lines.append(f"\n🔍 ATTACK CATEGORY ANALYSIS:")
            for category, count in analysis['attack_categories'].items():
                summary_lines.append(f"   • {category}: {count} sessions")
        
        summary_lines.append(f"\n📁 OUTPUT FILES:")
        summary_lines.append(f"   • anomaly_scores_0_100.csv - All session scores")
        summary_lines.append(f"   • top_anomalies.csv - Top 50 anomalies")
        summary_lines.append(f"   • anomaly_analysis.json - Detailed analysis")
        summary_lines.append(f"   • anomaly_summary.txt - This summary")
        
        summary_lines.append(f"\n" + "=" * 60)
        
        return "\n".join(summary_lines)


# -------------------------------------------------------------------
# Main execution function
# -------------------------------------------------------------------

def run_anomaly_scoring_pipeline(
    sessions_csv: str = "data/processed/all_sessions.csv",
    dag_model: str = "data/processed/dag_model.pkl",
    output_dir: str = "data/processed",
    anomaly_percentile: float = 5.0,
    save_results: bool = True
) -> Dict:
    """
    Main function to run anomaly scoring pipeline
    
    Args:
        sessions_csv: Path to sessions CSV
        dag_model: Path to DAG model pickle
        output_dir: Directory to save results
        anomaly_percentile: Bottom percentile for anomalies
        save_results: Whether to save results to files
        
    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("NETCAUSALAI - ANOMALY SCORING PIPELINE (0-100 SCORES)")
    print("=" * 60)
    
    try:
        # 1. Initialize scorer
        print("\n[1/5] Initializing Markov anomaly scorer...")
        scorer = MarkovAnomalyScorer(
            dag_model_path=dag_model,
            anomaly_percentile=anomaly_percentile
        )
        
        # 2. Load sessions data
        print(f"\n[2/5] Loading sessions from {sessions_csv}...")
        if not Path(sessions_csv).exists():
            raise FileNotFoundError(f"Sessions CSV not found: {sessions_csv}")
        
        sessions_df = pd.read_csv(sessions_csv)
        print(f"   • Loaded {len(sessions_df)} events")
        print(f"   • Unique sessions: {sessions_df['session_id'].nunique()}")
        
        # 3. Calculate anomaly scores (0-100 scale)
        print(f"\n[3/5] Calculating 0-100 anomaly scores...")
        scores_df = scorer.calculate_session_metrics(sessions_df)
        
        # 4. Detect anomalies
        print(f"\n[4/5] Detecting anomalies...")
        anomalies_df = scorer.detect_anomalies(scores_df, method='percentile')
        
        # Categorize anomalies
        anomalies_df = scorer.categorize_anomalies(anomalies_df)
        
        # 5. Analyze patterns
        anomaly_sessions = anomalies_df[anomalies_df['is_anomaly']]
        pattern_analysis = scorer.analyze_anomaly_patterns(anomaly_sessions)
        
        # 6. Save results
        if save_results:
            scorer.save_results(anomalies_df, pattern_analysis, output_dir)
        
        # 7. Print summary
        summary = scorer._create_summary(anomalies_df, pattern_analysis)
        print("\n" + summary)
        
        # Return results
        return {
            'scorer': scorer,
            'scores_df': scores_df,
            'anomalies_df': anomalies_df,
            'pattern_analysis': pattern_analysis,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# -------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NetCausalAI Anomaly Scoring (0-100 Scale)')
    parser.add_argument('--sessions', type=str, default='data/processed/all_sessions.csv',
                       help='Path to sessions CSV file')
    parser.add_argument('--dag-model', type=str, default='data/processed/dag_model.pkl',
                       help='Path to DAG model pickle file')
    parser.add_argument('--percentile', type=float, default=5.0,
                       help='Anomaly percentile threshold (default: 5.0)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for results')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results to files')
    
    args = parser.parse_args()
    
    # Run the pipeline
    results = run_anomaly_scoring_pipeline(
        sessions_csv=args.sessions,
        dag_model=args.dag_model,
        output_dir=args.output_dir,
        anomaly_percentile=args.percentile,
        save_results=not args.no_save
    )
    
    # Exit with appropriate code
    exit(0 if results.get('success', False) else 1)