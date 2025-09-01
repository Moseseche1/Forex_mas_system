import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class SelfLearningMarketAnalyzer:
    def __init__(self):
        self.market_memory = []
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.pattern_clusterer = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.learned_patterns = []
        
    async def analyze_market_regime(self, market_data: Dict) -> Dict:
        """Self-learning market regime analysis"""
        # Convert market data to feature vector
        features = self._extract_features(market_data)
        
        # Detect anomalies and novel patterns
        anomalies = self._detect_anomalies(features)
        patterns = self._cluster_patterns(features)
        
        # Update learned patterns
        self._update_learned_patterns(features, patterns)
        
        # Predict regime based on learned patterns
        regime = self._predict_regime(features)
        
        return {
            'regime': regime,
            'anomaly_score': anomalies,
            'pattern_cluster': patterns,
            'confidence': self._calculate_confidence(features),
            'learned_patterns_count': len(self.learned_patterns)
        }
    
    def _extract_features(self, market_data: Dict) -> np.array:
        """Extract features from market data"""
        features = [
            market_data.get('volatility', 0),
            market_data.get('trend_strength', 0),
            market_data.get('volume_ratio', 1),
            market_data.get('spread_ratio', 1),
            market_data.get('rsi', 50),
            market_data.get('price_change', 0),
            market_data.get('time_of_day', 12),
            market_data.get('day_of_week', 3),
        ]
        return np.array(features).reshape(1, -1)
    
    def _detect_anomalies(self, features: np.array) -> float:
        """Detect market anomalies using isolation forest"""
        try:
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Fit and predict anomalies
            self.anomaly_detector.fit(scaled_features)
            anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
            
            return float(anomaly_score)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return 0.0
    
    def _cluster_patterns(self, features: np.array) -> int:
        """Cluster market patterns using DBSCAN"""
        try:
            if len(self.market_memory) > 10:
                # Add current features to memory
                self.market_memory.append(features.flatten())
                
                # Cluster patterns
                clusters = self.pattern_clusterer.fit_predict(self.market_memory)
                return int(clusters[-1]) if clusters.size > 0 else -1
            
            return -1
            
        except Exception as e:
            logger.error(f"Pattern clustering failed: {e}")
            return -1
    
    def _update_learned_patterns(self, features: np.array, cluster: int):
        """Update learned patterns database"""
        if cluster != -1 and cluster not in self.learned_patterns:
            pattern_data = {
                'features': features.flatten().tolist(),
                'cluster': cluster,
                'timestamp': datetime.utcnow(),
                'occurrence_count': 1
            }
            self.learned_patterns.append(pattern_data)
    
    def _predict_regime(self, features: np.array) -> str:
        """Predict market regime based on learned patterns"""
        # Simple regime prediction based on volatility and trend
        volatility = features[0][0]
        trend = features[0][1]
        
        if volatility > 0.02:
            return 'high_volatility'
        elif trend > 0.5:
            return 'strong_trend_bull'
        elif trend < -0.5:
            return 'strong_trend_bear'
        else:
            return 'ranging'
