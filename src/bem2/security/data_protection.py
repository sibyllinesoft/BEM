"""
Data Protection and Poisoning Detection for VC0 Training Pipeline

Provides comprehensive protection against data poisoning attacks and
training data integrity validation for safety-critical ML systems.

Security Features:
- Statistical anomaly detection in training data
- Data poisoning pattern recognition
- Differential privacy mechanisms
- Secure data validation and sanitization
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import hashlib
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class DataValidationResult:
    """Result from data validation check"""
    is_valid: bool
    anomaly_score: float
    issues: List[str]
    statistics: Dict[str, Any]
    validation_time: float


class DataPoisoningDetector:
    """
    Comprehensive data poisoning detection system.
    
    Detects potential poisoning attacks in training data using
    statistical analysis, clustering, and pattern recognition.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Baseline statistics for comparison
        self.baseline_stats: Optional[Dict[str, Any]] = None
        self.baseline_established = False
        
        # Anomaly detection parameters
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.0)  # Standard deviations
        self.clustering_eps = self.config.get('clustering_eps', 0.5)
        self.min_samples = self.config.get('min_samples', 5)
        
        # Detection history
        self.detection_history: List[Dict[str, Any]] = []
        
        # Known poisoning patterns (would be updated based on threat intelligence)
        self.poisoning_patterns = self._initialize_poisoning_patterns()
        
        # Differential privacy parameters
        self.dp_epsilon = self.config.get('dp_epsilon', 1.0)
        self.dp_delta = self.config.get('dp_delta', 1e-5)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default data protection configuration"""
        return {
            'anomaly_threshold': 2.0,      # Standard deviations for anomaly detection
            'clustering_eps': 0.5,         # DBSCAN epsilon parameter
            'min_samples': 5,              # Minimum samples for cluster
            'statistical_tests': True,     # Enable statistical testing
            'clustering_analysis': True,   # Enable clustering-based detection
            'pattern_matching': True,      # Enable known pattern detection
            'differential_privacy': False, # Enable DP (impacts utility)
            'dp_epsilon': 1.0,             # DP privacy budget
            'dp_delta': 1e-5,              # DP delta parameter
            'validation_batch_size': 1000, # Batch size for validation
            'max_history_size': 10000      # Maximum detection history
        }
    
    def _initialize_poisoning_patterns(self) -> List[Dict[str, Any]]:
        """Initialize known poisoning attack patterns"""
        return [
            {
                'name': 'label_flipping',
                'description': 'Systematic label manipulation',
                'detection_method': 'statistical',
                'indicators': ['unusual_label_distribution', 'class_imbalance']
            },
            {
                'name': 'backdoor_trigger',
                'description': 'Backdoor trigger patterns in inputs',
                'detection_method': 'pattern_matching',
                'indicators': ['repeated_patterns', 'unusual_features']
            },
            {
                'name': 'gradient_ascent',
                'description': 'Data crafted to maximize loss',
                'detection_method': 'clustering',
                'indicators': ['outlier_clusters', 'loss_anomalies']
            },
            {
                'name': 'distribution_shift',
                'description': 'Systematic shift in data distribution',
                'detection_method': 'statistical',
                'indicators': ['distribution_divergence', 'feature_drift']
            }
        ]
    
    def establish_baseline(self, clean_data: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        Establish baseline statistics from known clean data.
        
        Args:
            clean_data: Known clean training data
            
        Returns:
            Baseline statistics dictionary
        """
        logger.info("Establishing data baseline from clean data")
        
        if isinstance(clean_data, torch.Tensor):
            data = clean_data.cpu().numpy()
        else:
            data = clean_data
        
        # Flatten data for statistical analysis
        if data.ndim > 2:
            data_flat = data.reshape(data.shape[0], -1)
        else:
            data_flat = data
        
        self.baseline_stats = {
            # Basic statistics
            'mean': np.mean(data_flat, axis=0),
            'std': np.std(data_flat, axis=0),
            'median': np.median(data_flat, axis=0),
            'min': np.min(data_flat, axis=0),
            'max': np.max(data_flat, axis=0),
            
            # Distribution properties
            'skewness': stats.skew(data_flat, axis=0),
            'kurtosis': stats.kurtosis(data_flat, axis=0),
            
            # Data shape and type info
            'shape': data.shape,
            'dtype': str(data.dtype),
            'sample_count': data.shape[0],
            
            # Advanced statistics
            'covariance_matrix': np.cov(data_flat.T),
            'principal_components': self._compute_pca_components(data_flat),
            
            # Timestamp
            'established_at': datetime.now()
        }
        
        self.baseline_established = True
        logger.info(f"Baseline established with {data.shape[0]} samples")
        
        return self.baseline_stats
    
    def _compute_pca_components(self, data: np.ndarray, n_components: int = 10) -> np.ndarray:
        """Compute principal components for baseline"""
        try:
            from sklearn.decomposition import PCA
            
            # Limit components to avoid memory issues
            n_components = min(n_components, data.shape[1], data.shape[0] - 1)
            
            if n_components <= 0:
                return np.array([])
            
            pca = PCA(n_components=n_components)
            pca.fit(data)
            return pca.components_
            
        except ImportError:
            logger.warning("sklearn not available for PCA computation")
            return np.array([])
        except Exception as e:
            logger.warning(f"PCA computation failed: {e}")
            return np.array([])
    
    def validate_training_data(self, data: Union[torch.Tensor, np.ndarray],
                              labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
                              batch_id: Optional[str] = None) -> DataValidationResult:
        """
        Validate training data for potential poisoning.
        
        Args:
            data: Training data to validate
            labels: Optional training labels
            batch_id: Optional batch identifier for tracking
            
        Returns:
            DataValidationResult with validation outcome
        """
        start_time = datetime.now()
        
        if not self.baseline_established:
            raise ValueError("Baseline must be established before validation")
        
        # Convert to numpy if needed
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data.copy()
        
        if labels is not None and isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels
        
        # Run validation checks
        validation_issues = []
        anomaly_scores = []
        
        # 1. Statistical validation
        if self.config.get('statistical_tests', True):
            stat_issues, stat_score = self._statistical_validation(data_np, labels_np)
            validation_issues.extend(stat_issues)
            anomaly_scores.append(stat_score)
        
        # 2. Clustering analysis
        if self.config.get('clustering_analysis', True):
            cluster_issues, cluster_score = self._clustering_validation(data_np)
            validation_issues.extend(cluster_issues)
            anomaly_scores.append(cluster_score)
        
        # 3. Pattern matching
        if self.config.get('pattern_matching', True):
            pattern_issues, pattern_score = self._pattern_validation(data_np, labels_np)
            validation_issues.extend(pattern_issues)
            anomaly_scores.append(pattern_score)
        
        # Calculate overall anomaly score
        overall_score = np.mean(anomaly_scores) if anomaly_scores else 0.0
        is_valid = overall_score < self.anomaly_threshold and len(validation_issues) == 0
        
        # Compute batch statistics
        batch_stats = self._compute_batch_statistics(data_np, labels_np)
        
        # Calculate validation time
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = DataValidationResult(
            is_valid=is_valid,
            anomaly_score=overall_score,
            issues=validation_issues,
            statistics=batch_stats,
            validation_time=validation_time
        )
        
        # Store in history
        self._store_validation_result(result, batch_id, data_np.shape)
        
        # Log results
        if not is_valid:
            logger.warning(f"Data validation failed with {len(validation_issues)} issues, "
                         f"anomaly score: {overall_score:.3f}")
        else:
            logger.debug(f"Data validation passed, anomaly score: {overall_score:.3f}")
        
        return result
    
    def _statistical_validation(self, data: np.ndarray, 
                               labels: Optional[np.ndarray]) -> Tuple[List[str], float]:
        """Perform statistical validation against baseline"""
        issues = []
        anomaly_scores = []
        
        # Flatten data for comparison
        if data.ndim > 2:
            data_flat = data.reshape(data.shape[0], -1)
        else:
            data_flat = data
        
        # Compare with baseline statistics
        batch_mean = np.mean(data_flat, axis=0)
        batch_std = np.std(data_flat, axis=0)
        
        baseline_mean = self.baseline_stats['mean']
        baseline_std = self.baseline_stats['std']
        
        # Z-score test for mean difference
        mean_z_scores = np.abs(batch_mean - baseline_mean) / (baseline_std + 1e-8)
        mean_anomaly = np.mean(mean_z_scores)
        
        if mean_anomaly > self.anomaly_threshold:
            issues.append(f"Mean deviation anomaly: {mean_anomaly:.3f}")
        
        anomaly_scores.append(mean_anomaly)
        
        # Standard deviation comparison
        std_ratio = np.mean(batch_std / (baseline_std + 1e-8))
        if std_ratio > 2.0 or std_ratio < 0.5:
            issues.append(f"Standard deviation anomaly: {std_ratio:.3f}")
            anomaly_scores.append(abs(std_ratio - 1.0))
        
        # Distribution tests
        try:
            # Kolmogorov-Smirnov test for distribution similarity
            for i in range(min(5, data_flat.shape[1])):  # Test first 5 dimensions
                baseline_sample = np.random.multivariate_normal(
                    baseline_mean[:len(baseline_mean)//10], 
                    np.eye(len(baseline_mean)//10) * np.mean(baseline_std[:len(baseline_mean)//10]),
                    size=min(1000, data_flat.shape[0])
                )[:, 0] if len(baseline_mean) > 10 else baseline_mean[i]
                
                if isinstance(baseline_sample, (int, float)):
                    baseline_sample = np.full(min(1000, data_flat.shape[0]), baseline_sample)
                
                ks_statistic, p_value = stats.ks_2samp(
                    data_flat[:, i], 
                    baseline_sample[:len(data_flat[:, i])]
                )
                
                if p_value < 0.01:  # Significant difference
                    issues.append(f"Distribution shift in dimension {i}: p={p_value:.4f}")
                    anomaly_scores.append(ks_statistic)
                    
        except Exception as e:
            logger.debug(f"Statistical test error: {e}")
        
        # Label distribution validation (if labels provided)
        if labels is not None:
            label_dist = np.bincount(labels.astype(int))
            expected_uniformity = len(label_dist) / np.sum(label_dist)
            actual_uniformity = np.std(label_dist / np.sum(label_dist))
            
            if actual_uniformity > 0.3:  # High class imbalance
                issues.append(f"Unusual label distribution: std={actual_uniformity:.3f}")
                anomaly_scores.append(actual_uniformity)
        
        overall_score = np.mean(anomaly_scores) if anomaly_scores else 0.0
        return issues, overall_score
    
    def _clustering_validation(self, data: np.ndarray) -> Tuple[List[str], float]:
        """Perform clustering-based validation"""
        issues = []
        
        try:
            # Flatten and standardize data
            if data.ndim > 2:
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            
            # Sample data if too large
            if data_flat.shape[0] > 5000:
                indices = np.random.choice(data_flat.shape[0], 5000, replace=False)
                data_sample = data_flat[indices]
            else:
                data_sample = data_flat
            
            # Standardize features
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_sample)
            
            # Apply DBSCAN clustering
            clusterer = DBSCAN(eps=self.clustering_eps, min_samples=self.min_samples)
            cluster_labels = clusterer.fit_predict(data_scaled)
            
            # Analyze clustering results
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_outliers = list(cluster_labels).count(-1)
            outlier_ratio = n_outliers / len(cluster_labels)
            
            # Check for anomalies
            if outlier_ratio > 0.1:  # More than 10% outliers
                issues.append(f"High outlier ratio: {outlier_ratio:.3f}")
            
            if n_clusters > len(data_sample) / 10:  # Too many small clusters
                issues.append(f"Excessive clustering: {n_clusters} clusters")
            
            return issues, outlier_ratio
            
        except Exception as e:
            logger.warning(f"Clustering validation failed: {e}")
            return [], 0.0
    
    def _pattern_validation(self, data: np.ndarray, 
                           labels: Optional[np.ndarray]) -> Tuple[List[str], float]:
        """Perform pattern-based validation for known attack signatures"""
        issues = []
        pattern_scores = []
        
        # Check for repeated patterns (potential backdoor triggers)
        if data.ndim > 1:
            # Look for exactly repeated samples
            unique_samples = len(np.unique(data.reshape(data.shape[0], -1), axis=0))
            repetition_ratio = 1.0 - (unique_samples / data.shape[0])
            
            if repetition_ratio > 0.05:  # More than 5% repeated samples
                issues.append(f"High sample repetition: {repetition_ratio:.3f}")
                pattern_scores.append(repetition_ratio)
        
        # Check for systematic patterns in data
        if data.ndim == 4:  # Image-like data
            # Check for small repeated patterns (potential triggers)
            patch_size = min(8, data.shape[-1] // 4)
            if patch_size > 0:
                patches = self._extract_patches(data, patch_size)
                unique_patches = len(np.unique(patches.reshape(patches.shape[0], -1), axis=0))
                patch_repetition = 1.0 - (unique_patches / patches.shape[0])
                
                if patch_repetition > 0.1:
                    issues.append(f"Repeated patch patterns: {patch_repetition:.3f}")
                    pattern_scores.append(patch_repetition)
        
        # Check label consistency patterns
        if labels is not None:
            # Look for systematic label flipping patterns
            if hasattr(self, 'expected_label_patterns'):
                # This would compare against expected patterns
                pass
        
        overall_score = np.mean(pattern_scores) if pattern_scores else 0.0
        return issues, overall_score
    
    def _extract_patches(self, data: np.ndarray, patch_size: int) -> np.ndarray:
        """Extract patches from image-like data"""
        patches = []
        
        for i in range(0, data.shape[-2] - patch_size + 1, patch_size):
            for j in range(0, data.shape[-1] - patch_size + 1, patch_size):
                patch = data[:, :, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        
        return np.array(patches)
    
    def _compute_batch_statistics(self, data: np.ndarray, 
                                 labels: Optional[np.ndarray]) -> Dict[str, Any]:
        """Compute comprehensive statistics for batch"""
        if data.ndim > 2:
            data_flat = data.reshape(data.shape[0], -1)
        else:
            data_flat = data
        
        stats_dict = {
            'batch_size': data.shape[0],
            'feature_dimensions': data_flat.shape[1],
            'mean': np.mean(data_flat).item(),
            'std': np.std(data_flat).item(),
            'min': np.min(data_flat).item(),
            'max': np.max(data_flat).item(),
            'nan_count': np.isnan(data_flat).sum().item(),
            'inf_count': np.isinf(data_flat).sum().item(),
            'zero_ratio': (data_flat == 0).sum().item() / data_flat.size,
            'unique_samples': len(np.unique(data_flat, axis=0))
        }
        
        if labels is not None:
            stats_dict.update({
                'label_classes': len(np.unique(labels)),
                'label_distribution': np.bincount(labels.astype(int)).tolist(),
                'label_imbalance': np.std(np.bincount(labels.astype(int)))
            })
        
        return stats_dict
    
    def _store_validation_result(self, result: DataValidationResult, 
                                batch_id: Optional[str], data_shape: tuple):
        """Store validation result in history"""
        record = {
            'timestamp': datetime.now(),
            'batch_id': batch_id,
            'data_shape': data_shape,
            'is_valid': result.is_valid,
            'anomaly_score': result.anomaly_score,
            'issue_count': len(result.issues),
            'issues': result.issues,
            'validation_time': result.validation_time
        }
        
        self.detection_history.append(record)
        
        # Maintain history size limit
        max_size = self.config.get('max_history_size', 10000)
        if len(self.detection_history) > max_size:
            self.detection_history = self.detection_history[-max_size:]
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        if not self.detection_history:
            return {'message': 'No validation history available'}
        
        recent_validations = self.detection_history[-100:]  # Last 100 validations
        
        return {
            'total_validations': len(self.detection_history),
            'recent_validations': len(recent_validations),
            'poisoning_detection_rate': sum(1 for r in recent_validations if not r['is_valid']) / len(recent_validations),
            'average_anomaly_score': np.mean([r['anomaly_score'] for r in recent_validations]),
            'average_validation_time': np.mean([r['validation_time'] for r in recent_validations]),
            'baseline_established': self.baseline_established,
            'baseline_timestamp': (
                self.baseline_stats['established_at'].isoformat() 
                if self.baseline_established else None
            ),
            'common_issues': self._get_common_issues(),
            'anomaly_threshold': self.anomaly_threshold
        }
    
    def _get_common_issues(self) -> Dict[str, int]:
        """Get most common validation issues"""
        issue_counts = {}
        
        for record in self.detection_history[-1000:]:  # Last 1000 records
            for issue in record['issues']:
                # Extract issue type (before first colon)
                issue_type = issue.split(':')[0]
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Return top 10 most common issues
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def add_differential_privacy(self, data: torch.Tensor, 
                                sensitivity: float = 1.0) -> torch.Tensor:
        """
        Add differential privacy noise to data.
        
        Args:
            data: Input data tensor
            sensitivity: Sensitivity of the function
            
        Returns:
            Data with DP noise added
        """
        if not self.config.get('differential_privacy', False):
            return data
        
        # Calculate noise scale based on epsilon and sensitivity
        noise_scale = sensitivity / self.dp_epsilon
        
        # Add Gaussian noise (for epsilon-delta DP)
        noise = torch.normal(0, noise_scale, size=data.shape, device=data.device)
        
        return data + noise


# Utility functions
def create_data_protector(baseline_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
                         config: Optional[Dict[str, Any]] = None) -> DataPoisoningDetector:
    """Create data poisoning detector with optional baseline"""
    detector = DataPoisoningDetector(config)
    
    if baseline_data is not None:
        detector.establish_baseline(baseline_data)
    
    return detector


def validate_batch_safety(data: Union[torch.Tensor, np.ndarray],
                         detector: DataPoisoningDetector,
                         labels: Optional[Union[torch.Tensor, np.ndarray]] = None) -> bool:
    """Quick validation check for batch safety"""
    result = detector.validate_training_data(data, labels)
    return result.is_valid