"""
Root Cause Analysis (RCA) for NetCausalAI
Transforms anomaly detection into structured, human-readable explanations

RCA answers:
1. WHEN did the session deviate from normal?
2. WHAT transitions caused the anomaly?
3. WHY is it abnormal? (type classification)
4. WHICH behavior dominates?
5. HOW extreme is it compared to baseline?
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# RCA SIGNAL DEFINITIONS - Core building blocks of explanations
# ============================================================================

class AnomalyDriver(Enum):
    """Classification of WHAT type of anomaly this is"""
    UNSEEN_TRANSITION = "unseen_transition"
    RARE_TRANSITION = "rare_transition"
    REPETITIVE_PATTERN = "repetitive_pattern"
    LENGTH_EXTREME = "length_extreme"
    ENTROPY_ABNORMAL = "entropy_abnormal"
    DOMINANCE_EXTREME = "dominance_extreme"
    BURST_BEHAVIOR = "burst_behavior"
    SUSTAINED_LOW_PROB = "sustained_low_probability"
    SINGLE_POINT_FAILURE = "single_point_failure"
    MIXED = "mixed_anomaly"


class SeverityLevel(Enum):
    """Operational severity for triage"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TransitionAnomaly:
    """A single suspicious transition within a session"""
    position: int
    from_event: str
    to_event: str
    probability: float
    log_probability: float
    contribution: float
    is_unseen: bool
    is_rare: bool
    confidence: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DeviationPoint:
    """When the session started behaving abnormally"""
    index: int
    event_index: int
    cumulative_score_at_point: float
    percent_through_session: float
    event_at_point: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class RootCauseExplanation:
    """Complete RCA output for a single session"""
    # Identification
    session_id: str
    anomaly_score_0_100: float
    severity: str
    
    # WHEN it deviated
    deviation_start: DeviationPoint
    
    # WHAT caused it
    primary_transition: TransitionAnomaly
    top_anomalous_transitions: List[TransitionAnomaly]
    
    # WHY (classification)
    drivers: List[str]
    driver_descriptions: List[str]
    root_causes: List[str]
    
    # DOMINANCE analysis
    dominant_event: str
    dominance_ratio: float
    event_distribution: Dict[str, int]
    
    # CONTEXT metrics
    event_count: int
    unseen_count: int
    rare_count: int
    repetition_rate: float
    entropy: float
    length_factor: Optional[float] = None
    bytes_transferred: Optional[float] = None
    
    # Human-readable story
    narrative: str = ""
    
    # Cluster hint (for RCA → clustering pipeline)
    cluster_hint: str = ""
    
    def to_dict(self):
        """Convert to serializable dict"""
        result = {
            'session_id': self.session_id,
            'anomaly_score_0_100': self.anomaly_score_0_100,
            'severity': self.severity,
            'deviation_start': self.deviation_start.to_dict() if self.deviation_start else None,
            'primary_transition': self.primary_transition.to_dict() if self.primary_transition else None,
            'top_anomalous_transitions': [t.to_dict() for t in self.top_anomalous_transitions[:3]],
            'drivers': self.drivers,
            'driver_descriptions': self.driver_descriptions,
            'root_causes': self.root_causes,
            'dominant_event': self.dominant_event,
            'dominance_ratio': self.dominance_ratio,
            'event_distribution': self.event_distribution,
            'event_count': self.event_count,
            'unseen_count': self.unseen_count,
            'rare_count': self.rare_count,
            'repetition_rate': self.repetition_rate,
            'entropy': self.entropy,
            'length_factor': self.length_factor,
            'bytes_transferred': self.bytes_transferred,
            'narrative': self.narrative,
            'cluster_hint': self.cluster_hint
        }
        return result


# ============================================================================
# RCA ENGINE - Core logic for explaining anomalies
# ============================================================================

class RootCauseAnalyzer:
    """
    Root Cause Analysis engine for network session anomalies
    
    Transforms anomaly scores into structured, human-readable explanations
    """
    
    def __init__(self, 
                 dag_model_path: str = "data/processed/dag_model_complete.pkl",
                 baseline_stats_path: Optional[str] = None,
                 min_prob: float = 0.001,
                 rare_threshold: float = 0.01,
                 dominance_threshold: float = 0.7,
                 repetition_threshold: float = 0.8,
                 entropy_low_threshold: float = 1.0,
                 entropy_high_threshold: float = 3.0):
        """
        Initialize RCA engine with learned model and thresholds
        """
        
        # Thresholds for RCA decisions
        self.rare_threshold = rare_threshold
        self.dominance_threshold = dominance_threshold
        self.repetition_threshold = repetition_threshold
        self.entropy_low_threshold = entropy_low_threshold
        self.entropy_high_threshold = entropy_high_threshold
        self.min_prob = min_prob
        
        # Load the Markov model
        self._load_model(dag_model_path)
        
        # Load or compute baseline statistics
        self.baseline_stats = self._load_or_compute_baseline(baseline_stats_path)
        
        print(f"✅ RootCauseAnalyzer initialized")
        print(f"   • Model: {Path(dag_model_path).name}")
        print(f"   • Transitions: {len(self.transition_probs)}")
        print(f"   • Baseline sessions: {self.baseline_stats.get('num_sessions', 0)}")
    
    def _load_model(self, path: str):
        """Load the trained Markov model"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.transition_probs = model_data['probabilities']
            self.confidence_scores = model_data.get('confidence_scores', {})
            self.transition_counts = model_data.get('transition_counts', {})
            self.transition_sessions = model_data.get('transition_sessions', {})
            self.graph = model_data.get('graph', None)
            
        except FileNotFoundError:
            # Fallback to old format
            fallback = "data/processed/dag_model.pkl"
            print(f"⚠️ Model not found at {path}, trying {fallback}")
            with open(fallback, 'rb') as f:
                model_data = pickle.load(f)
            self.transition_probs = model_data['probabilities']
            self.confidence_scores = {}
            self.transition_counts = {}
            self.transition_sessions = {}
            self.graph = None
    
    def _load_or_compute_baseline(self, path: Optional[str]) -> Dict:
        """Load or compute baseline session statistics"""
        
        # Try to load existing baseline
        if path and Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
        
        # Otherwise return default baseline
        # In production, you'd compute this from training data
        return {
            'num_sessions': 1000,
            'mean_session_length': 50,
            'std_session_length': 30,
            'mean_entropy': 2.1,
            'std_entropy': 0.8,
            'percentile_95_length': 120,
            'percentile_99_length': 200
        }
    
    # ========================================================================
    # STEP 1: FIND WHEN DEVIATION BEGINS
    # ========================================================================
    
    def find_deviation_point(self, events: List[str], 
                            cumulative_log: np.ndarray) -> DeviationPoint:
        """
        Identify the exact point where session behavior became anomalous
        
        Method: Find first significant drop in cumulative log probability
        compared to early-session behavior
        """
        if len(events) < 3 or len(cumulative_log) < 2:
            return DeviationPoint(
                index=0,
                event_index=1,
                cumulative_score_at_point=0,
                percent_through_session=0,
                event_at_point=events[0] if events else "UNKNOWN"
            )
        
        # Use first 20% of session as baseline "normal" behavior
        baseline_end = max(1, int(len(cumulative_log) * 0.2))
        if baseline_end >= len(cumulative_log):
            baseline_end = len(cumulative_log) - 1
        
        baseline_mean = np.mean(cumulative_log[:baseline_end])
        baseline_std = max(0.1, np.std(cumulative_log[:baseline_end]))
        
        # Find first point where cumulative score drops significantly
        deviation_idx = 0
        for i in range(baseline_end, len(cumulative_log)):
            # 2 standard deviation drop = significant
            if cumulative_log[i] < baseline_mean - (2 * baseline_std):
                deviation_idx = i
                break
        
        # If no sharp drop, find the minimum point
        if deviation_idx == 0:
            deviation_idx = np.argmin(cumulative_log)
        
        return DeviationPoint(
            index=deviation_idx,
            event_index=deviation_idx + 1,  # +1 because transitions vs events
            cumulative_score_at_point=float(cumulative_log[deviation_idx]),
            percent_through_session=deviation_idx / len(cumulative_log),
            event_at_point=events[deviation_idx + 1] if deviation_idx + 1 < len(events) else events[-1]
        )
    
    # ========================================================================
    # STEP 2: TRANSITION-LEVEL RCA (CORE)
    # ========================================================================
    
    def analyze_transitions(self, events: List[str]) -> Tuple[List[TransitionAnomaly], float, List[float]]:
        """
        Deep analysis of every transition in the session
        
        Returns:
            - List of TransitionAnomaly objects
            - Average log probability
            - Cumulative log probabilities
        """
        if len(events) < 2:
            return [], 0.0, []
        
        transitions = []
        log_probs = []
        
        for i in range(len(events) - 1):
            src, dst = events[i], events[i+1]
            prob = self.transition_probs.get((src, dst), self.min_prob)
            log_prob = np.log(prob)
            log_probs.append(log_prob)
            
            transitions.append(TransitionAnomaly(
                position=i,
                from_event=src,
                to_event=dst,
                probability=prob,
                log_probability=log_prob,
                contribution=0.0,  # Will compute after normalization
                is_unseen=prob == self.min_prob,
                is_rare=prob < self.rare_threshold and prob != self.min_prob,
                confidence=self.confidence_scores.get((src, dst), 0.0)
            ))
        
        # Calculate contribution percentages
        total_abs_sum = np.sum(np.abs(log_probs))
        if total_abs_sum > 0:
            for t in transitions:
                t.contribution = np.abs(t.log_probability) / total_abs_sum
        
        # Sort by anomaly contribution
        transitions.sort(key=lambda x: x.contribution, reverse=True)
        
        return transitions, float(np.mean(log_probs)), log_probs
    
    # ========================================================================
    # STEP 3: EVENT DOMINANCE ANALYSIS
    # ========================================================================
    
    def analyze_event_dominance(self, events: List[str]) -> Tuple[str, float, Dict[str, int]]:
        """
        Identify if session is dominated by a single event type
        
        Returns:
            - Dominant event name
            - Dominance ratio (0-1)
            - Full event distribution
        """
        if not events:
            return "UNKNOWN", 0.0, {}
        
        # Count events
        event_counts = {}
        for e in events:
            event_counts[e] = event_counts.get(e, 0) + 1
        
        # Find most common
        dominant = max(event_counts.items(), key=lambda x: x[1])
        dominance_ratio = dominant[1] / len(events)
        
        return dominant[0], dominance_ratio, event_counts
    
    # ========================================================================
    # STEP 4: LENGTH-BASED RCA
    # ========================================================================
    
    def analyze_length_anomaly(self, event_count: int) -> Tuple[bool, float, str]:
        """
        Determine if session is abnormally long
        
        Returns:
            - Is length anomaly?
            - Length factor (how many times longer than baseline)
            - Description
        """
        baseline_mean = self.baseline_stats.get('mean_session_length', 50)
        baseline_std = self.baseline_stats.get('std_session_length', 30)
        p95 = self.baseline_stats.get('percentile_95_length', 120)
        
        length_factor = event_count / baseline_mean if baseline_mean > 0 else 1.0
        
        if event_count > p95:
            return True, length_factor, f"session length ({event_count}) exceeds 95th percentile ({p95})"
        elif event_count > baseline_mean + (3 * baseline_std):
            return True, length_factor, f"session length ({event_count}) is >3σ above baseline"
        else:
            return False, length_factor, "normal length"
    
    # ========================================================================
    # STEP 5: ENTROPY & PREDICTABILITY
    # ========================================================================
    
    def analyze_entropy(self, events: List[str]) -> Tuple[float, str, str]:
        """
        Calculate entropy and interpret predictability
        
        Returns:
            - Entropy value
            - Interpretation (low/normal/high)
            - Behavioral description
        """
        if len(events) < 2:
            return 0.0, "insufficient", "too few events"
        
        # Calculate transition probabilities for this session
        trans_counts = defaultdict(int)
        for i in range(len(events) - 1):
            trans_counts[(events[i], events[i+1])] += 1
        
        total_trans = len(events) - 1
        
        # Calculate entropy
        entropy = 0.0
        for count in trans_counts.values():
            p = count / total_trans
            entropy -= p * np.log2(p)
        
        # Interpret
        if entropy < self.entropy_low_threshold:
            interpretation = "low"
            description = "highly repetitive, scripted behavior"
        elif entropy > self.entropy_high_threshold:
            interpretation = "high"
            description = "chaotic, scanning/fuzzing-like behavior"
        else:
            interpretation = "normal"
            description = "diverse but structured interaction"
        
        return entropy, interpretation, description
    
    # ========================================================================
    # STEP 6: ROOT CAUSE CLASSIFICATION
    # ========================================================================
    
    def classify_root_causes(self, 
                            events: List[str],
                            transitions: List[TransitionAnomaly],
                            deviation: DeviationPoint,
                            dominant_event: str,
                            dominance_ratio: float,
                            entropy_val: float,
                            entropy_interpretation: str,
                            is_length_anomaly: bool,
                            repetition_rate: float) -> Tuple[List[str], List[str], List[str]]:
        """
        Combine all signals to determine root causes
        
        Returns:
            - Drivers (categorical)
            - Driver descriptions (detailed)
            - Root causes (human-readable)
        """
        drivers = []
        driver_descriptions = []
        root_causes = []
        
        # ----- Signal 1: Unseen transitions -----
        unseen_count = sum(1 for t in transitions if t.is_unseen)
        if unseen_count > 0:
            drivers.append(AnomalyDriver.UNSEEN_TRANSITION.value)
            driver_descriptions.append(f"{unseen_count} transition(s) never observed in training")
            
            # Find the first unseen transition
            first_unseen = next((t for t in transitions if t.is_unseen), None)
            if first_unseen:
                root_causes.append(
                    f"never-seen behavior: {first_unseen.from_event} → {first_unseen.to_event} "
                    f"at position {first_unseen.position}"
                )
            else:
                root_causes.append(f"{unseen_count} unseen transition patterns")
        
        # ----- Signal 2: Rare transitions -----
        rare_count = sum(1 for t in transitions if t.is_rare)
        if rare_count > 0:
            drivers.append(AnomalyDriver.RARE_TRANSITION.value)
            driver_descriptions.append(f"{rare_count} statistically rare transition(s) (p < {self.rare_threshold})")
            
            most_rare = next((t for t in transitions if t.is_rare), None)
            if most_rare and not root_causes:
                root_causes.append(
                    f"unusual transition: {most_rare.from_event} → {most_rare.to_event} "
                    f"(probability={most_rare.probability:.4f})"
                )
        
        # ----- Signal 3: Repetitive pattern -----
        if repetition_rate > self.repetition_threshold:
            drivers.append(AnomalyDriver.REPETITIVE_PATTERN.value)
            driver_descriptions.append(f"highly repetitive (repetition rate={repetition_rate:.2f})")
            root_causes.append(f"extreme repetition of {dominant_event} events")
        
        # ----- Signal 4: Length anomaly -----
        if is_length_anomaly:
            drivers.append(AnomalyDriver.LENGTH_EXTREME.value)
            driver_descriptions.append(f"abnormally long session ({len(events)} events)")
            root_causes.append(f"session duration far above baseline")
        
        # ----- Signal 5: Entropy abnormal -----
        if entropy_interpretation == "low":
            drivers.append(AnomalyDriver.ENTROPY_ABNORMAL.value)
            driver_descriptions.append(f"low entropy ({entropy_val:.2f}) - scripted behavior")
            root_causes.append(f"automated/scripted behavior pattern")
        elif entropy_interpretation == "high":
            drivers.append(AnomalyDriver.ENTROPY_ABNORMAL.value)
            driver_descriptions.append(f"high entropy ({entropy_val:.2f}) - chaotic")
            root_causes.append(f"random/scanning behavior pattern")
        
        # ----- Signal 6: Dominance extreme -----
        if dominance_ratio > self.dominance_threshold:
            drivers.append(AnomalyDriver.DOMINANCE_EXTREME.value)
            driver_descriptions.append(f"dominated by {dominant_event} ({dominance_ratio:.1%})")
            if not any("repetition" in rc for rc in root_causes):
                root_causes.append(f"flood-like behavior: {dominant_event} dominates traffic")
        
        # ----- Signal 7: Sustained low probability -----
        avg_log = np.mean([t.log_probability for t in transitions]) if transitions else 0
        if avg_log < -5 and len(transitions) > 10:
            drivers.append(AnomalyDriver.SUSTAINED_LOW_PROB.value)
            driver_descriptions.append(f"consistently low probability transitions throughout")
            root_causes.append(f"sustained unusual behavior from start to end")
        
        # ----- Signal 8: Single point failure -----
        if len(transitions) > 0 and transitions[0].contribution > 0.5:
            drivers.append(AnomalyDriver.SINGLE_POINT_FAILURE.value)
            driver_descriptions.append(f"single transition causes >50% of anomaly")
            if not root_causes:
                t = transitions[0]
                root_causes.append(
                    f"critical anomalous transition at step {t.position}: "
                    f"{t.from_event}→{t.to_event}"
                )
        
        # Default case
        if not drivers:
            drivers.append(AnomalyDriver.MIXED.value)
            driver_descriptions.append("multiple subtle anomalies")
            root_causes.append("combination of unusual behaviors")
        
        return drivers, driver_descriptions, root_causes
    
    # ========================================================================
    # STEP 7: NARRATIVE GENERATION (Human-readable story)
    # ========================================================================
    
    def generate_narrative(self,
                          session_id: str,
                          events: List[str],
                          deviation: DeviationPoint,
                          dominant_event: str,
                          dominance_ratio: float,
                          drivers: List[str],
                          root_causes: List[str],
                          entropy_interpretation: str,
                          is_length_anomaly: bool) -> str:
        """
        Transform technical RCA into a human-readable story
        
        This is what makes the project sellable to SOC analysts
        """
        
        # Start with normal behavior description
        if deviation.index > 5:
            normal_part = f"behaved normally for {deviation.event_index} events"
        else:
            normal_part = "behaved abnormally from the start"
        
        # Describe the deviation
        if deviation.index < len(events) - 1:
            deviation_desc = f"then at event {deviation.event_index} ({deviation.event_at_point}), "
        else:
            deviation_desc = ""
        
        # Describe the dominant behavior
        if dominance_ratio > self.dominance_threshold:
            dominance_desc = f"dominated by {dominant_event} ({dominance_ratio:.0%} of traffic), "
        else:
            dominance_desc = ""
        
        # Describe the root causes
        cause_desc = " and ".join(root_causes[:2]).lower()
        if len(root_causes) > 2:
            cause_desc += f" plus {len(root_causes)-2} other anomalies"
        
        # Entropy description
        if entropy_interpretation == "low":
            entropy_desc = "highly repetitive, scripted"
        elif entropy_interpretation == "high":
            entropy_desc = "chaotic, scanning-like"
        else:
            entropy_desc = "unusual"
        
        # Length description
        if is_length_anomaly:
            length_desc = f"abnormally long ({len(events)} events), "
        else:
            length_desc = ""
        
        # Construct the full narrative
        narrative = (
            f"Session {session_id[:8]}... {normal_part}. "
            f"{deviation_desc}{dominance_desc}resulting in {entropy_desc} behavior "
            f"characterized by {cause_desc}. "
            f"{length_desc}Overall anomaly score: high confidence."
        )
        
        return narrative
    
    # ========================================================================
    # STEP 8: CLUSTER HINT (Bridge to clustering)
    # ========================================================================
    
    def generate_cluster_hint(self,
                             dominant_event: str,
                             dominance_ratio: float,
                             drivers: List[str],
                             entropy_val: float,
                             is_length_anomaly: bool,
                             unseen_count: int) -> str:
        """
        Generate a hint for clustering algorithms
        
        This creates a natural grouping key based on RCA output
        """
        
        if unseen_count > 0:
            return "novel_behavior"
        
        if dominance_ratio > 0.9 and dominant_event in ['TCP_SYN', 'UDP_PACKET']:
            return "flood_attack"
        
        if dominance_ratio > 0.8 and dominant_event == 'LARGE_TRANSFER':
            return "data_exfiltration"
        
        if is_length_anomaly and dominance_ratio > 0.7:
            return "long_session_attack"
        
        if 'rare_transition' in drivers and entropy_val < 1.5:
            return "scripted_attack"
        
        if entropy_val > 3.0:
            return "scanning_behavior"
        
        return "unusual_pattern"
    
    # ========================================================================
    # MAIN RCA FUNCTION
    # ========================================================================
    
    def explain_session(self,
                       session_id: str,
                       events: List[str],
                       anomaly_score: float = 0.0,
                       severity: str = "unknown",
                       repetition_rate: Optional[float] = None,
                       bytes_transferred: Optional[float] = None) -> RootCauseExplanation:
        """
        Complete Root Cause Analysis for a single session
        
        This is the main entry point for RCA
        """
        
        if len(events) < 2:
            return None
        
        # ----- STEP 1: Analyze transitions -----
        transitions, avg_log_prob, log_probs = self.analyze_transitions(events)
        cumulative_log = np.cumsum(log_probs) if log_probs else np.array([])
        
        # ----- STEP 2: Find deviation point -----
        deviation = self.find_deviation_point(events, cumulative_log)
        
        # ----- STEP 3: Analyze event dominance -----
        dominant_event, dominance_ratio, event_dist = self.analyze_event_dominance(events)
        
        # ----- STEP 4: Calculate repetition rate (if not provided) -----
        if repetition_rate is None:
            unique_events = len(set(events))
            repetition_rate = 1 - (unique_events / len(events)) if len(events) > 0 else 0
        
        # ----- STEP 5: Analyze entropy -----
        entropy_val, entropy_interp, entropy_desc = self.analyze_entropy(events)
        
        # ----- STEP 6: Analyze length anomaly -----
        is_length_anomaly, length_factor, length_desc = self.analyze_length_anomaly(len(events))
        
        # ----- STEP 7: Count unseen/rare -----
        unseen_count = sum(1 for t in transitions if t.is_unseen)
        rare_count = sum(1 for t in transitions if t.is_rare)
        
        # ----- STEP 8: Classify root causes -----
        drivers, driver_descriptions, root_causes = self.classify_root_causes(
            events, transitions, deviation, dominant_event, dominance_ratio,
            entropy_val, entropy_interp, is_length_anomaly, repetition_rate
        )
        
        # ----- STEP 9: Generate narrative -----
        narrative = self.generate_narrative(
            session_id, events, deviation, dominant_event, dominance_ratio,
            drivers, root_causes, entropy_interp, is_length_anomaly
        )
        
        # ----- STEP 10: Generate cluster hint -----
        cluster_hint = self.generate_cluster_hint(
            dominant_event, dominance_ratio, drivers, entropy_val,
            is_length_anomaly, unseen_count
        )
        
        # ----- Build complete explanation -----
        explanation = RootCauseExplanation(
            session_id=session_id,
            anomaly_score_0_100=anomaly_score,
            severity=severity,
            deviation_start=deviation,
            primary_transition=transitions[0] if transitions else None,
            top_anomalous_transitions=transitions[:5],
            drivers=drivers,
            driver_descriptions=driver_descriptions,
            root_causes=root_causes[:5],  # Top 5 causes
            dominant_event=dominant_event,
            dominance_ratio=dominance_ratio,
            event_distribution=event_dist,
            event_count=len(events),
            unseen_count=unseen_count,
            rare_count=rare_count,
            repetition_rate=repetition_rate,
            entropy=entropy_val,
            length_factor=length_factor if is_length_anomaly else None,
            bytes_transferred=bytes_transferred,
            narrative=narrative,
            cluster_hint=cluster_hint
        )
        
        return explanation
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    def explain_anomalies(self,
                         sessions_df: pd.DataFrame,
                         scores_df: pd.DataFrame,
                         limit: int = 100) -> pd.DataFrame:
        """
        Generate RCA explanations for multiple anomalous sessions
        
        Args:
            sessions_df: DataFrame with 'session_id' and 'event' columns
            scores_df: DataFrame with anomaly scores and severity
            limit: Maximum number of anomalies to explain
            
        Returns:
            DataFrame with complete RCA explanations
        """
        print(f"🔍 Generating RCA explanations for top {limit} anomalies...")
        
        # Get top anomalies by score
        top_anomalies = scores_df[scores_df['is_anomaly']].head(limit)
        
        explanations = []
        
        for idx, row in top_anomalies.iterrows():
            session_id = row['session_id']
            
            # Get events for this session
            session_events = sessions_df[sessions_df['session_id'] == session_id]
            events = session_events['event'].tolist()
            
            if len(events) < 2:
                continue
            
            # Calculate repetition rate
            unique_events = len(set(events))
            repetition_rate = 1 - (unique_events / len(events))
            
            # Get bytes if available
            bytes_transferred = None
            if 'length' in session_events.columns:
                bytes_transferred = session_events['length'].sum()
            
            # Generate RCA
            explanation = self.explain_session(
                session_id=session_id,
                events=events,
                anomaly_score=row.get('anomaly_score_0_100', 0),
                severity=row.get('severity', 'unknown'),
                repetition_rate=repetition_rate,
                bytes_transferred=bytes_transferred
            )
            
            if explanation:
                explanations.append(explanation.to_dict())
            
            # Progress
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1} sessions...")
        
        print(f"✅ Generated RCA for {len(explanations)} sessions")
        return pd.DataFrame(explanations)
    
    # ========================================================================
    # OUTPUT & SAVING
    # ========================================================================
    
    def save_explanations(self,
                         explanations_df: pd.DataFrame,
                         output_dir: str = "data/processed") -> Dict[str, str]:
        """
        Save RCA explanations in multiple formats
        
        Args:
            explanations_df: DataFrame with RCA explanations
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with paths to saved files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # 1. CSV format (for analysis)
        csv_path = f"{output_dir}/rca_explanations.csv"
        explanations_df.to_csv(csv_path, index=False)
        paths['csv'] = csv_path
        
        # 2. JSON format (for dashboard/API)
        json_path = f"{output_dir}/rca_explanations.json"
        explanations_df.to_json(json_path, orient='records', indent=2)
        paths['json'] = json_path
        
        # 3. Human-readable summary (for SOC analysts)
        summary_path = f"{output_dir}/rca_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary_report(explanations_df))
        paths['summary'] = summary_path
        
        # 4. Cluster hints (for clustering pipeline)
        if 'cluster_hint' in explanations_df.columns:
            cluster_path = f"{output_dir}/rca_cluster_hints.csv"
            explanations_df[['session_id', 'cluster_hint', 'anomaly_score_0_100', 'severity']].to_csv(cluster_path, index=False)
            paths['cluster_hints'] = cluster_path
        
        print(f"💾 RCA outputs saved:")
        for k, v in paths.items():
            print(f"   • {k}: {v}")
        
        return paths
    
    def _generate_summary_report(self, df: pd.DataFrame) -> str:
        """Generate human-readable summary of RCA findings"""
        
        lines = []
        lines.append("=" * 80)
        lines.append("NETCAUSALAI - ROOT CAUSE ANALYSIS SUMMARY")
        lines.append("=" * 80)
        lines.append(f"\nAnalyzed: {len(df)} anomalous sessions")
        lines.append(f"Generated: {pd.Timestamp.now()}")
        lines.append("\n" + "-" * 80)
        
        # Top 10 most severe
        lines.append("\n🔴 TOP 10 MOST SEVERE ANOMALIES (with RCA):\n")
        
        for idx, row in df.head(10).iterrows():
            lines.append(f"[{idx+1}] Session: {row['session_id'][:40]}...")
            lines.append(f"      Score: {row['anomaly_score_0_100']:.1f}/100 | Severity: {row['severity']}")
            lines.append(f"      Deviation: Started at event {row.get('deviation_start', {}).get('event_index', '?')}")
            
            # Root causes
            if 'root_causes' in row and row['root_causes']:
                causes = row['root_causes']
                if isinstance(causes, str):
                    import ast
                    try:
                        causes = ast.literal_eval(causes)
                    except:
                        causes = [causes]
                
                for cause in causes[:3]:
                    lines.append(f"      • {cause}")
            
            # Narrative
            if 'narrative' in row:
                lines.append(f"      📖 {row['narrative']}")
            
            lines.append("")
        
        # Cluster hint distribution
        if 'cluster_hint' in df.columns:
            lines.append("\n" + "-" * 80)
            lines.append("\n📊 ANOMALY FAMILIES (RCA Cluster Hints):\n")
            cluster_dist = df['cluster_hint'].value_counts()
            for cluster, count in cluster_dist.items():
                pct = count / len(df) * 100
                lines.append(f"   • {cluster}: {count} sessions ({pct:.1f}%)")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)


# ============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# ============================================================================

def run_rca_pipeline(
    sessions_csv: str = "data/processed/all_sessions_detailed.csv",
    scores_csv: str = "data/processed/anomaly_scores_with_features.csv",
    dag_model: str = "data/processed/dag_model_complete.pkl",
    output_dir: str = "data/processed",
    limit: int = 100
) -> Dict:
    """
    Complete RCA pipeline - integrate with existing anomaly detection
    
    Args:
        sessions_csv: Path to detailed sessions CSV
        scores_csv: Path to anomaly scores CSV
        dag_model: Path to DAG model
        output_dir: Output directory
        limit: Max number of anomalies to explain
        
    Returns:
        Dictionary with results and paths
    """
    
    print("=" * 60)
    print("NETCAUSALAI - ROOT CAUSE ANALYSIS PIPELINE")
    print("=" * 60)
    
    try:
        # 1. Load data
        print("\n[1/4] Loading session data and anomaly scores...")
        
        if not Path(sessions_csv).exists():
            sessions_csv = "data/processed/all_sessions.csv"
        
        sessions_df = pd.read_csv(sessions_csv)
        scores_df = pd.read_csv(scores_csv)
        
        print(f"   • Sessions: {sessions_df['session_id'].nunique()}")
        print(f"   • Events: {len(sessions_df)}")
        print(f"   • Anomaly scores: {len(scores_df)}")
        
        # 2. Initialize RCA engine
        print("\n[2/4] Initializing Root Cause Analyzer...")
        rca = RootCauseAnalyzer(dag_model_path=dag_model)
        
        # 3. Generate explanations
        print("\n[3/4] Generating RCA explanations...")
        explanations_df = rca.explain_anomalies(
            sessions_df=sessions_df,
            scores_df=scores_df,
            limit=limit
        )
        
        # 4. Save outputs
        print("\n[4/4] Saving RCA outputs...")
        paths = rca.save_explanations(explanations_df, output_dir)
        
        # 5. Summary
        print("\n" + "=" * 60)
        print("✅ RCA PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\n📊 RCA Statistics:")
        print(f"   • Anomalies explained: {len(explanations_df)}")
        print(f"   • RCA features per session: {len(explanations_df.columns)}")
        print(f"   • Output files: {len(paths)}")
        
        return {
            'success': True,
            'explanations_df': explanations_df,
            'rca_engine': rca,
            'paths': paths
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NetCausalAI Root Cause Analysis')
    parser.add_argument('--sessions', type=str, default='data/processed/all_sessions_detailed.csv')
    parser.add_argument('--scores', type=str, default='data/processed/anomaly_scores_with_features.csv')
    parser.add_argument('--dag-model', type=str, default='data/processed/dag_model_complete.pkl')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--limit', type=int, default=100, help='Max anomalies to explain')
    
    args = parser.parse_args()
    
    results = run_rca_pipeline(
        sessions_csv=args.sessions,
        scores_csv=args.scores,
        dag_model=args.dag_model,
        output_dir=args.output_dir,
        limit=args.limit
    )
    
    exit(0 if results.get('success', False) else 1)