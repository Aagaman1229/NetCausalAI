"""
Behavioral Clustering for NetCausalAI
Groups anomalous sessions into behavior families using HDBSCAN

Core philosophy:
    Detection → finds outliers
    RCA → explains each outlier
    Clustering → finds patterns among outliers

Outputs:
    - Behavior families (clusters) of similar anomalies
    - Campaign detection (multiple sessions = same behavior)
    - Noise identification (one-off anomalies)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Clustering imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import hdbscan

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For UMAP (optional but recommended)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP not installed. Install with: pip install umap-learn")


class BehavioralClusterer:
    """
    Cluster anomalous sessions using RCA feature vectors
    
    Input: RCA explanations (one per anomalous session)
    Output: Behavior families + interpretable cluster descriptions
    """
    
    def __init__(self,
                 min_cluster_size: int = 5,
                 min_samples: int = 3,
                 cluster_selection_epsilon: float = 0.5,
                 metric: str = 'euclidean',
                 normalize: bool = True,
                 scale_method: str = 'robust'):
        """
        Initialize HDBSCAN clusterer with security-appropriate defaults
        
        Args:
            min_cluster_size: Minimum sessions to form a cluster (campaign detection)
            min_samples: Density threshold (lower = more conservative)
            cluster_selection_epsilon: Merge nearby clusters (0 = pure HDBSCAN)
            metric: Distance metric
            normalize: Whether to normalize features
            scale_method: 'robust' (better) or 'standard'
        """
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.normalize = normalize
        self.scale_method = scale_method
        
        # Will be set during fit
        self.clusterer = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
        print(f"✅ BehavioralClusterer initialized")
        print(f"   • HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        print(f"   • Normalization: {scale_method if normalize else 'None'}")
    
    # ========================================================================
    # STEP 1: Feature Selection (from RCA output)
    # ========================================================================
    
    def select_features(self, rca_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select RCA-derived features for clustering
        
        Core behavioral features that capture the 'shape' of an anomaly
        """
        
        # Core numeric features (MUST exist in RCA dataframe)
        core_features = [
            'entropy',              # Predictability vs randomness
            'dominance_ratio',      # Flood vs balanced behavior
            'event_count',          # Session length
            'unseen_count',         # Novelty
            'rare_count',          # Statistical rarity
            'repetition_rate',     # Loop/script behavior
            'anomaly_score_0_100'  # Severity
        ]
        
        # Derived features (compute if available)
        derived_features = []
        
        # length_factor (if exists)
        if 'length_factor' in rca_df.columns:
            derived_features.append('length_factor')
        
        # Add percent_unseen and percent_rare
        if 'unseen_count' in rca_df.columns and 'event_count' in rca_df.columns:
            rca_df = rca_df.copy()
            rca_df['percent_unseen'] = rca_df['unseen_count'] / (rca_df['event_count'] - 1)
            rca_df['percent_unseen'] = rca_df['percent_unseen'].fillna(0).clip(0, 1)
            derived_features.append('percent_unseen')
        
        if 'rare_count' in rca_df.columns and 'event_count' in rca_df.columns:
            rca_df = rca_df.copy()
            rca_df['percent_rare'] = rca_df['rare_count'] / (rca_df['event_count'] - 1)
            rca_df['percent_rare'] = rca_df['percent_rare'].fillna(0).clip(0, 1)
            derived_features.append('percent_rare')
        
        # Optional: Add driver-derived features
        if 'drivers' in rca_df.columns:
            # We'll handle this separately in cluster interpretation
            pass
        
        # Combine all available features
        available_features = []
        for feat in core_features + derived_features:
            if feat in rca_df.columns:
                available_features.append(feat)
            else:
                print(f"   ⚠️ Feature '{feat}' not found, skipping")
        
        # Handle missing values
        feature_df = rca_df[available_features].copy()
        
        # Replace inf/-inf with NaN, then fill NaN with median
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        for col in feature_df.columns:
            if feature_df[col].isna().any():
                median_val = feature_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                feature_df[col] = feature_df[col].fillna(median_val)
        
        # Clip extreme values
        for col in feature_df.columns:
            if col in ['event_count', 'unseen_count', 'rare_count']:
                # Log transform for count features
                feature_df[col] = np.log1p(feature_df[col])
        
        print(f"\n📊 Selected {len(available_features)} features for clustering:")
        for feat in available_features:
            print(f"   • {feat}")
        
        return feature_df, available_features
    
    # ========================================================================
    # STEP 2: Normalization (CRITICAL for proper clustering)
    # ========================================================================
    
    def normalize_features(self, feature_df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Normalize features to same scale
        
        Security data has wildly different scales:
            entropy: 0-4
            event_count: 10-10000
            dominance_ratio: 0-1
        
        Without normalization: clustering is dominated by magnitude
        """
        
        if fit:
            if self.scale_method == 'robust':
                self.scaler = RobustScaler(quantile_range=(5, 95))
            else:
                self.scaler = StandardScaler()
            
            normalized = self.scaler.fit_transform(feature_df)
        else:
            normalized = self.scaler.transform(feature_df)
        
        return normalized
    
    # ========================================================================
    # STEP 3: Dimensionality Reduction (for visualization only)
    # ========================================================================
    
    def reduce_dimensions(self, normalized_features: np.ndarray, 
                         method: str = 'umap',
                         n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensions for visualization
        
        This is ONLY for visualization, not for clustering!
        HDBSCAN works in original feature space.
        """
        
        if method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=n_components,
                random_state=42,
                metric='euclidean'
            )
            reduced = reducer.fit_transform(normalized_features)
            self.reducer = reducer
            print(f"   • UMAP reduction: {normalized_features.shape[1]}→{n_components} dims")
            return reduced
        
        elif method == 'pca':
            pca = PCA(n_components=n_components, random_state=42)
            reduced = pca.fit_transform(normalized_features)
            print(f"   • PCA reduction: {normalized_features.shape[1]}→{n_components} dims")
            print(f"   • Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
            self.reducer = pca
            return reduced
        
        else:
            print(f"   • No reduction (using original features)")
            return normalized_features
    
    # ========================================================================
    # STEP 4: HDBSCAN Clustering
    # ========================================================================
    
    def fit_clusters(self, normalized_features: np.ndarray) -> np.ndarray:
        """
        Run HDBSCAN clustering on normalized features
        
        HDBSCAN is perfect for security because:
            1. Doesn't force every point into a cluster
            2. Finds clusters of varying density
            3. Noise label (-1) = unique anomalies (very interesting!)
        """
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            core_dist_n_jobs=-1,
            prediction_data=True,  # Enable predict for new data
            gen_min_span_tree=True
        )
        
        cluster_labels = self.clusterer.fit_predict(normalized_features)
        self.is_fitted = True
        
        return cluster_labels
    
    # ========================================================================
    # STEP 5: Fit & Transform (Complete Pipeline)
    # ========================================================================
    
    def fit_transform(self, rca_df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete clustering pipeline:
            1. Select features from RCA dataframe
            2. Normalize features
            3. Run HDBSCAN
            4. Attach labels back to original dataframe
        
        Args:
            rca_df: DataFrame with RCA explanations (one row per anomalous session)
            
        Returns:
            DataFrame with cluster labels and metadata
        """
        
        print("\n" + "="*60)
        print("🔬 BEHAVIORAL CLUSTERING PIPELINE")
        print("="*60)
        
        # STEP 1: Feature selection
        print("\n[1/4] Selecting RCA features...")
        feature_df, self.feature_names = self.select_features(rca_df)
        print(f"   • Feature matrix: {feature_df.shape}")
        
        # STEP 2: Normalization
        print("\n[2/4] Normalizing features...")
        normalized = self.normalize_features(feature_df, fit=True)
        
        # STEP 3: HDBSCAN clustering
        print("\n[3/4] Running HDBSCAN clustering...")
        cluster_labels = self.fit_clusters(normalized)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = sum(cluster_labels == -1)
        
        print(f"\n📊 Clustering results:")
        print(f"   • Clusters found: {n_clusters}")
        print(f"   • Sessions in clusters: {len(cluster_labels) - n_noise}")
        print(f"   • Noise sessions (unique): {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
        
        # STEP 4: Attach results to dataframe
        print("\n[4/4] Attaching cluster labels...")
        result_df = rca_df.copy()
        result_df['cluster_id'] = cluster_labels
        
        # Add probability scores (how confidently a point belongs to its cluster)
        if hasattr(self.clusterer, 'probabilities_'):
            result_df['cluster_probability'] = self.clusterer.probabilities_
        
        # Add outlier scores (higher = more anomalous within dataset)
        if hasattr(self.clusterer, 'outlier_scores_'):
            result_df['outlier_score'] = self.clusterer.outlier_scores_
        
        # Generate cluster names
        result_df['cluster_name'] = result_df['cluster_id'].apply(
            lambda x: 'NOISE' if x == -1 else f'Family_{x}'
        )
        
        # Generate persistence scores (cluster stability)
        if hasattr(self.clusterer, 'cluster_persistence_'):
            persistence_map = {i: p for i, p in enumerate(self.clusterer.cluster_persistence_)}
            result_df['cluster_persistence'] = result_df['cluster_id'].map(persistence_map)
        
        print(f"\n✅ Clustering complete!")
        print(f"   • Output: {len(result_df)} sessions with cluster labels")
        
        return result_df
    
    # ========================================================================
    # STEP 6: Predict new sessions (using trained model)
    # ========================================================================
    
    def predict(self, rca_df: pd.DataFrame) -> np.ndarray:
        """
        Predict clusters for new sessions using trained HDBSCAN model
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_transform first!")
        
        # Feature selection (using same features)
        feature_df = rca_df[self.feature_names].copy()
        
        # Handle missing values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        for col in feature_df.columns:
            if feature_df[col].isna().any():
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Log transform count features
        for col in feature_df.columns:
            if col in ['event_count', 'unseen_count', 'rare_count']:
                feature_df[col] = np.log1p(feature_df[col])
        
        # Normalize
        normalized = self.normalize_features(feature_df, fit=False)
        
        # Predict using HDBSCAN's approximate prediction
        labels, _ = hdbscan.approximate_predict(self.clusterer, normalized)
        
        return labels
    
    # ========================================================================
    # STEP 7: Cluster Interpretation (CRITICAL for SOC value)
    # ========================================================================
    
    def interpret_clusters(self, clustered_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Generate human-readable interpretations for each cluster
        
        This is what makes clustering useful for SOC analysts
        """
        
        interpretations = {}
        
        # Get all cluster IDs (excluding -1 for separate handling)
        cluster_ids = sorted([c for c in clustered_df['cluster_id'].unique() if c != -1])
        
        for cluster_id in cluster_ids:
            cluster_data = clustered_df[clustered_df['cluster_id'] == cluster_id]
            
            # ===== Statistical profile =====
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(clustered_df) * 100,
                
                # Central tendencies
                'mean_entropy': cluster_data['entropy'].mean(),
                'mean_dominance': cluster_data['dominance_ratio'].mean(),
                'mean_repetition': cluster_data['repetition_rate'].mean(),
                'mean_unseen': cluster_data['unseen_count'].mean(),
                'mean_rare': cluster_data['rare_count'].mean(),
                'mean_score': cluster_data['anomaly_score_0_100'].mean(),
                'mean_length': cluster_data['event_count'].mean(),
                
                # Variability
                'std_entropy': cluster_data['entropy'].std(),
                'std_dominance': cluster_data['dominance_ratio'].std(),
            }
            
            # ===== Dominant patterns =====
            # Most common driver
            if 'drivers' in cluster_data.columns:
                all_drivers = []
                for drivers in cluster_data['drivers'].dropna():
                    if isinstance(drivers, str):
                        try:
                            import ast
                            driver_list = ast.literal_eval(drivers)
                            all_drivers.extend(driver_list)
                        except:
                            all_drivers.append(drivers)
                
                if all_drivers:
                    driver_counts = Counter(all_drivers)
                    stats['top_driver'] = driver_counts.most_common(1)[0][0]
                    stats['driver_distribution'] = dict(driver_counts.most_common(3))
            
            # Most common dominant event
            if 'dominant_event' in cluster_data.columns:
                event_counts = cluster_data['dominant_event'].value_counts()
                stats['top_event'] = event_counts.index[0] if len(event_counts) > 0 else 'unknown'
                stats['event_distribution'] = event_counts.head(3).to_dict()
            
            # Most common cluster hint
            if 'cluster_hint' in cluster_data.columns:
                hint_counts = cluster_data['cluster_hint'].value_counts()
                stats['top_hint'] = hint_counts.index[0] if len(hint_counts) > 0 else 'unknown'
            
            # ===== Generate human-readable label =====
            label = self._generate_cluster_label(stats, cluster_data)
            stats['human_label'] = label
            
            # ===== Generate description =====
            description = self._generate_cluster_description(stats, cluster_id, cluster_data)
            stats['description'] = description
            
            # ===== Find prototype session (most representative) =====
            stats['prototype_session'] = self._find_prototype(cluster_data)
            
            # ===== Example sessions (top 3 most anomalous) =====
            stats['example_sessions'] = cluster_data.nlargest(3, 'anomaly_score_0_100')[
                ['session_id', 'anomaly_score_0_100', 'severity', 'cluster_hint'] + 
                (['narrative'] if 'narrative' in cluster_data.columns else [])
            ].to_dict('records')
            
            interpretations[int(cluster_id)] = stats
        
        # Special handling for noise points
        if -1 in clustered_df['cluster_id'].values:
            noise_data = clustered_df[clustered_df['cluster_id'] == -1]
            interpretations[-1] = {
                'size': len(noise_data),
                'percentage': len(noise_data) / len(clustered_df) * 100,
                'human_label': 'UNIQUE_ANOMALIES',
                'description': 'One-of-a-kind anomalies that don\'t fit any behavior family. Each requires individual investigation.',
                'example_sessions': noise_data.nlargest(3, 'anomaly_score_0_100')[
                    ['session_id', 'anomaly_score_0_100', 'severity', 'cluster_hint']
                ].to_dict('records')
            }
        
        return interpretations
    
    def _generate_cluster_label(self, stats: Dict, cluster_data: pd.DataFrame) -> str:
        """Generate a human-readable label for the cluster"""
        
        # High dominance + low entropy + high repetition = FLOOD
        if stats['mean_dominance'] > 0.8 and stats['mean_entropy'] < 1.5:
            if stats.get('top_event') in ['TCP_SYN', 'UDP_PACKET']:
                return f"FLOOD_ATTACK"
            elif stats.get('top_event') == 'LARGE_TRANSFER':
                return f"DATA_EXFILTRATION"
            else:
                return f"REPETITIVE_FLOOD"
        
        # High unseen + high rarity = NOVEL ATTACK
        if stats['mean_unseen'] > 2 or stats['mean_rare'] > 5:
            return f"NOVEL_BEHAVIOR"
        
        # High entropy + low dominance = SCANNING
        if stats['mean_entropy'] > 3.0 and stats['mean_dominance'] < 0.3:
            return f"SCANNING_BEHAVIOR"
        
        # Long sessions
        if stats['mean_length'] > 500:
            return f"LONG_SESSION_CAMPAIGN"
        
        # Default: Use top driver
        if 'top_driver' in stats:
            driver_name = stats['top_driver'].upper().replace('_', ' ')
            return f"{driver_name}_CAMPAIGN"
        
        return f"BEHAVIOR_FAMILY_{int(stats.get('size', 0))}"
    
    def _generate_cluster_description(self, stats: Dict, cluster_id: int, 
                                    cluster_data: pd.DataFrame) -> str:
        """Generate a detailed description of the cluster"""
        
        parts = []
        
        # Size
        parts.append(f"Cluster {cluster_id}: {stats['size']} sessions ({stats['percentage']:.1f}%)")
        
        # Behavior summary
        behavior_parts = []
        
        if stats['mean_dominance'] > 0.7:
            behavior_parts.append(f"dominated by {stats.get('top_event', 'single event')}")
        if stats['mean_entropy'] < 1.5:
            behavior_parts.append("highly repetitive")
        if stats['mean_entropy'] > 3.0:
            behavior_parts.append("chaotic/scanning")
        if stats['mean_unseen'] > 1:
            behavior_parts.append("novel transitions")
        if stats['mean_length'] > 1000:
            behavior_parts.append("extremely long")
        
        if behavior_parts:
            parts.append("sessions are " + ", ".join(behavior_parts))
        
        # Anomaly severity
        if stats['mean_score'] > 95:
            parts.append("CRITICAL severity")
        elif stats['mean_score'] > 85:
            parts.append("HIGH severity")
        elif stats['mean_score'] > 70:
            parts.append("MEDIUM severity")
        
        # Campaign detection
        if stats['size'] > 10:
            parts.append(f"likely an ongoing campaign")
        
        return ". ".join(parts)
    
    def _find_prototype(self, cluster_data: pd.DataFrame) -> Dict:
        """
        Find the most representative session in the cluster
        
        Prototype = session closest to cluster centroid
        """
        if len(cluster_data) == 0:
            return {}
        
        # Get numeric features for this cluster
        feature_cols = [f for f in self.feature_names if f in cluster_data.columns]
        
        if not feature_cols:
            return cluster_data.iloc[0][['session_id', 'anomaly_score_0_100']].to_dict()
        
        # Calculate centroid
        centroid = cluster_data[feature_cols].mean()
        
        # Find session closest to centroid
        distances = []
        for idx, row in cluster_data[feature_cols].iterrows():
            dist = np.linalg.norm(row - centroid)
            distances.append(dist)
        
        prototype_idx = np.argmin(distances)
        prototype = cluster_data.iloc[prototype_idx]
        
        return {
            'session_id': prototype.get('session_id', 'unknown'),
            'anomaly_score': prototype.get('anomaly_score_0_100', 0),
            'severity': prototype.get('severity', 'unknown'),
            'narrative': prototype.get('narrative', '')[:200] + '...' if prototype.get('narrative') else ''
        }
    
    # ========================================================================
    # STEP 8: Visualization
    # ========================================================================
    
    def visualize_clusters(self, clustered_df: pd.DataFrame, 
                          reduced_features: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None):
        """
        Create visualization of clusters (2D projection)
        """
        
        # Get reduced dimensions for visualization
        if reduced_features is None:
            # Get normalized features
            feature_df, _ = self.select_features(clustered_df)
            normalized = self.normalize_features(feature_df, fit=False)
            reduced = self.reduce_dimensions(normalized, method='umap')
        else:
            reduced = reduced_features
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ===== Plot 1: Clusters =====
        ax1 = axes[0]
        
        # Get unique clusters
        clusters = sorted(clustered_df['cluster_id'].unique())
        
        # Color map (with -1 as gray)
        colors = plt.cm.tab20(np.linspace(0, 1, len([c for c in clusters if c != -1])))
        
        # Plot noise first
        noise_mask = clustered_df['cluster_id'] == -1
        if noise_mask.any():
            ax1.scatter(reduced[noise_mask, 0], reduced[noise_mask, 1],
                       c='lightgray', s=30, alpha=0.6, label=f'Noise ({sum(noise_mask)})')
        
        # Plot clusters
        color_idx = 0
        for cluster_id in clusters:
            if cluster_id == -1:
                continue
            
            mask = clustered_df['cluster_id'] == cluster_id
            ax1.scatter(reduced[mask, 0], reduced[mask, 1],
                       c=[colors[color_idx]], s=50, alpha=0.7,
                       label=f'Family {cluster_id} ({sum(mask)})')
            color_idx += 1
        
        ax1.set_title('Behavior Families (HDBSCAN)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # ===== Plot 2: Anomaly Score =====
        ax2 = axes[1]
        
        scatter = ax2.scatter(reduced[:, 0], reduced[:, 1],
                            c=clustered_df['anomaly_score_0_100'],
                            cmap='RdYlGn_r', s=50, alpha=0.7,
                            vmin=70, vmax=100)
        
        ax2.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=ax2, label='Anomaly Score (0-100)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # STEP 9: Export Results
    # ========================================================================
    
    def export_results(self, clustered_df: pd.DataFrame,
                      interpretations: Dict,
                      output_dir: str = "data/processed") -> Dict[str, str]:
        """
        Export clustering results in multiple formats
        """
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # 1. Full clustering results
        results_path = f"{output_dir}/clustering_results.csv"
        clustered_df.to_csv(results_path, index=False)
        paths['full_results'] = results_path
        
        # 2. Cluster interpretations (JSON)
        interpretations_path = f"{output_dir}/cluster_interpretations.json"
        with open(interpretations_path, 'w') as f:
            # Convert numpy types to Python types
            clean_interpretations = json.loads(
                json.dumps(interpretations, default=lambda x: float(x) if isinstance(x, (np.floating, float)) else 
                          str(x) if isinstance(x, np.integer) else x)
            )
            json.dump(clean_interpretations, f, indent=2)
        paths['interpretations'] = interpretations_path
        
        # 3. Human-readable summary
        summary_path = f"{output_dir}/clustering_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary_report(clustered_df, interpretations))
        paths['summary'] = summary_path
        
        # 4. Campaign report (clusters with >5 sessions)
        campaigns = clustered_df[clustered_df['cluster_id'] != -1].groupby('cluster_id').filter(lambda x: len(x) >= 5)
        if len(campaigns) > 0:
            campaign_path = f"{output_dir}/active_campaigns.csv"
            campaigns[['session_id', 'cluster_id', 'cluster_name', 'anomaly_score_0_100', 
                      'severity', 'timestamp' if 'timestamp' in campaigns.columns else 'session_id']].to_csv(campaign_path, index=False)
            paths['campaigns'] = campaign_path
        
        # 5. Save model
        model_path = f"{output_dir}/clusterer_model.pkl"
        with open(model_path, 'wb') as f:
            # Save only the clusterer and scaler, not the full dataframe
            model_data = {
                'clusterer': self.clusterer,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples
            }
            pickle.dump(model_data, f)
        paths['model'] = model_path
        
        print(f"\n💾 Clustering outputs saved:")
        for k, v in paths.items():
            print(f"   • {k}: {v}")
        
        return paths
    
    def _generate_summary_report(self, clustered_df: pd.DataFrame, 
                                interpretations: Dict) -> str:
        """Generate a human-readable summary of clustering results"""
        
        lines = []
        lines.append("=" * 80)
        lines.append("NETCAUSALAI - BEHAVIORAL CLUSTERING SUMMARY")
        lines.append("=" * 80)
        lines.append(f"\nAnalysis Timestamp: {pd.Timestamp.now()}")
        
        # Overview
        n_total = len(clustered_df)
        n_clusters = len([c for c in interpretations.keys() if c != -1])
        n_noise = interpretations.get(-1, {}).get('size', 0)
        
        lines.append(f"\n📊 OVERVIEW:")
        lines.append(f"   • Anomalous sessions analyzed: {n_total}")
        lines.append(f"   • Behavior families found: {n_clusters}")
        lines.append(f"   • Unique anomalies (noise): {n_noise} ({n_noise/n_total*100:.1f}%)")
        lines.append(f"   • Sessions grouped into families: {n_total - n_noise} ({100 - n_noise/n_total*100:.1f}%)")
        
        # Cluster summaries
        lines.append(f"\n🔬 BEHAVIOR FAMILIES:")
        
        for cluster_id, stats in interpretations.items():
            if cluster_id == -1:
                continue
                
            lines.append(f"\n  [Family {cluster_id}] {stats.get('human_label', 'UNKNOWN')}")
            lines.append(f"  • Sessions: {stats['size']} ({stats['percentage']:.1f}%)")
            lines.append(f"  • Description: {stats.get('description', 'No description')}")
            
            # Characteristics
            chars = []
            if stats['mean_dominance'] > 0.7:
                chars.append(f"dominance={stats['mean_dominance']:.1%}")
            if stats['mean_entropy']:
                chars.append(f"entropy={stats['mean_entropy']:.2f}")
            if stats['mean_unseen']:
                chars.append(f"unseen={stats['mean_unseen']:.1f}")
            
            if chars:
                lines.append(f"  • Characteristics: {', '.join(chars)}")
            
            # Top events
            if 'event_distribution' in stats:
                events = list(stats['event_distribution'].items())[:3]
                events_str = ', '.join([f"{e} ({c})" for e, c in events])
                lines.append(f"  • Top events: {events_str}")
            
            # Prototype
            if 'prototype_session' in stats and stats['prototype_session']:
                proto = stats['prototype_session']
                lines.append(f"  • Prototype: {proto.get('session_id', '')[:30]}... (score={proto.get('anomaly_score', 0):.0f})")
        
        # Noise section
        if -1 in interpretations:
            noise = interpretations[-1]
            lines.append(f"\n🎯 UNIQUE ANOMALIES (NOISE):")
            lines.append(f"  • {noise['size']} sessions with no similar behavior")
            lines.append(f"  • Each requires individual investigation")
            lines.append(f"  • May represent novel attacks or rare events")
        
        # Recommendations
        lines.append(f"\n🎯 OPERATIONAL RECOMMENDATIONS:")
        
        # Find critical campaigns
        critical_clusters = []
        for cluster_id, stats in interpretations.items():
            if cluster_id != -1 and stats.get('mean_score', 0) > 90 and stats['size'] >= 5:
                critical_clusters.append(f"Family {cluster_id} ({stats.get('human_label', 'Unknown')})")
        
        if critical_clusters:
            lines.append(f"  • CRITICAL CAMPAIGNS DETECTED: {', '.join(critical_clusters)}")
            lines.append(f"  • Investigate immediately - these are active, severe behavior patterns")
        else:
            lines.append(f"  • No critical campaigns detected")
        
        lines.append(f"  • Focus on top 3 families for immediate triage")
        lines.append(f"  • Review unique anomalies for novel attack patterns")
        
        lines.append(f"\n" + "=" * 80)
        
        return "\n".join(lines)


# ========================================================================
# MAIN PIPELINE
# ========================================================================

def run_clustering_pipeline(
    rca_csv: str = "data/processed/rca_explanations.csv",
    output_dir: str = "data/processed",
    min_cluster_size: int = 5,
    visualize: bool = True
) -> Dict:
    """
    Complete behavioral clustering pipeline
    
    Args:
        rca_csv: Path to RCA explanations CSV
        output_dir: Output directory
        min_cluster_size: Minimum sessions to form a cluster
        visualize: Whether to generate visualizations
    
    Returns:
        Dictionary with results and paths
    """
    
    print("=" * 60)
    print("NETCAUSALAI - BEHAVIORAL CLUSTERING PIPELINE")
    print("=" * 60)
    
    try:
        # 1. Load RCA data
        print("\n[1/5] Loading RCA explanations...")
        if not Path(rca_csv).exists():
            # Try alternative paths
            alt_paths = [
                "data/processed/rca_explanations.csv",
                "../data/processed/rca_explanations.csv",
                "../../data/processed/rca_explanations.csv"
            ]
            for alt in alt_paths:
                if Path(alt).exists():
                    rca_csv = alt
                    break
        
        rca_df = pd.read_csv(rca_csv)
        print(f"   • Loaded {len(rca_df)} RCA explanations")
        print(f"   • Columns: {list(rca_df.columns)}")
        
        # 2. Initialize clusterer
        print("\n[2/5] Initializing HDBSCAN clusterer...")
        clusterer = BehavioralClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=max(3, min_cluster_size // 2),
            cluster_selection_epsilon=0.5,
            metric='euclidean',
            normalize=True,
            scale_method='robust'
        )
        
        # 3. Fit clusters
        print("\n[3/5] Running HDBSCAN clustering...")
        clustered_df = clusterer.fit_transform(rca_df)
        
        # 4. Interpret clusters
        print("\n[4/5] Interpreting behavior families...")
        interpretations = clusterer.interpret_clusters(clustered_df)
        
        # 5. Export results
        print("\n[5/5] Exporting results...")
        paths = clusterer.export_results(clustered_df, interpretations, output_dir)
        
        # 6. Visualize (optional)
        if visualize:
            print("\n🎨 Generating visualizations...")
            
            # Get normalized features for visualization
            feature_df, _ = clusterer.select_features(clustered_df)
            normalized = clusterer.normalize_features(feature_df, fit=False)
            
            # UMAP reduction
            if UMAP_AVAILABLE:
                reduced = clusterer.reduce_dimensions(normalized, method='umap')
            else:
                reduced = clusterer.reduce_dimensions(normalized, method='pca')
            
            # Create visualization
            viz_path = f"{output_dir}/cluster_visualization.png"
            clusterer.visualize_clusters(clustered_df, reduced, save_path=viz_path)
            paths['visualization'] = viz_path
        
        # 7. Summary
        print("\n" + "=" * 60)
        print("✅ CLUSTERING PIPELINE COMPLETE")
        print("=" * 60)
        
        n_clusters = len([c for c in interpretations.keys() if c != -1])
        n_noise = interpretations.get(-1, {}).get('size', 0)
        
        print(f"\n📊 Final Results:")
        print(f"   • Behavior families: {n_clusters}")
        print(f"   • Sessions in families: {len(clustered_df) - n_noise}")
        print(f"   • Unique anomalies: {n_noise}")
        print(f"\n📁 Outputs saved to: {output_dir}")
        
        return {
            'success': True,
            'clusterer': clusterer,
            'clustered_df': clustered_df,
            'interpretations': interpretations,
            'paths': paths
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NetCausalAI Behavioral Clustering')
    parser.add_argument('--rca-csv', type=str, default='data/processed/rca_explanations.csv')
    parser.add_argument('--output-dir', type=str, default='data/processed')
    parser.add_argument('--min-cluster-size', type=int, default=5, 
                       help='Minimum sessions to form a cluster (campaign threshold)')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    results = run_clustering_pipeline(
        rca_csv=args.rca_csv,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        visualize=not args.no_viz
    )
    
    exit(0 if results.get('success', False) else 1)