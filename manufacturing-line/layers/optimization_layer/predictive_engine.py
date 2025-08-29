"""
Week 4: PredictiveEngine - Equipment Failure Prediction and Maintenance Scheduling

This module implements predictive analytics capabilities for manufacturing line control,
including equipment failure prediction, performance anomaly detection, and intelligent
maintenance scheduling.

Key Features:
- Equipment failure prediction with ML models
- Performance anomaly detection in real-time
- Intelligent maintenance scheduling recommendations
- Historical data processing and trend analysis
- Real-time predictive model inference with <200ms target

Author: Claude Code
Date: 2024-08-28
Version: 1.0
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import threading
from datetime import datetime, timedelta
import math
import random

class PredictionModelType(Enum):
    FAILURE_PREDICTION = "failure_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    PERFORMANCE_FORECAST = "performance_forecast"
    MAINTENANCE_OPTIMIZATION = "maintenance_optimization"

class AnomalyType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNUSUAL_PATTERN = "unusual_pattern"
    THRESHOLD_VIOLATION = "threshold_violation"
    TREND_ANOMALY = "trend_anomaly"

@dataclass
class PredictionResult:
    equipment_id: str
    prediction_type: PredictionModelType
    probability: float
    confidence: float
    time_horizon_hours: int
    predicted_failure_time: Optional[datetime]
    contributing_factors: List[str]
    recommendation: str
    computation_time_ms: float

@dataclass
class AnomalyDetection:
    equipment_id: str
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    detected_at: datetime
    affected_metrics: List[str]
    anomaly_score: float
    threshold_violated: bool
    description: str

@dataclass
class MaintenanceRecommendation:
    equipment_id: str
    maintenance_type: str
    priority: int  # 1-5, 5 being highest priority
    recommended_date: datetime
    estimated_duration_hours: float
    estimated_cost: float
    impact_on_production: str
    justification: str

@dataclass
class PerformanceForecast:
    equipment_id: str
    metric_name: str
    forecast_horizon_hours: int
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    trend: str  # "improving", "degrading", "stable"
    forecast_accuracy: float

class PredictiveEngine:
    """Equipment failure prediction and maintenance scheduling engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PredictiveEngine with configuration."""
        self.config = config or {}
        
        # Performance configuration
        self.performance_target_ms = self.config.get('performance_target_ms', 200)
        self.model_inference_timeout_ms = self.config.get('model_inference_timeout_ms', 1000)
        
        # Model parameters
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.5)  # Standard deviations
        self.failure_probability_threshold = self.config.get('failure_probability_threshold', 0.8)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Historical data storage
        self.equipment_history: Dict[str, List[Dict[str, Any]]] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.anomaly_history: Dict[str, List[AnomalyDetection]] = {}
        
        # Model storage (simplified - in production would use actual ML models)
        self.models = {
            PredictionModelType.FAILURE_PREDICTION: self._create_failure_model(),
            PredictionModelType.ANOMALY_DETECTION: self._create_anomaly_model(),
            PredictionModelType.PERFORMANCE_FORECAST: self._create_forecast_model(),
            PredictionModelType.MAINTENANCE_OPTIMIZATION: self._create_maintenance_model()
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'avg_prediction_time_ms': 0,
            'total_predictions': 0,
            'successful_predictions': 0,
            'anomalies_detected': 0,
            'maintenance_recommendations': 0,
            'prediction_accuracy': 0.0
        }
        
        logging.info(f"PredictiveEngine initialized with {self.performance_target_ms}ms target")

    def predict_equipment_failure(self, 
                                 equipment_id: str,
                                 equipment_data: Dict[str, Any],
                                 time_horizon_hours: int = 168) -> PredictionResult:
        """Predict equipment failure probability within specified time horizon."""
        start_time = time.time()
        
        try:
            # Validate inputs
            if not equipment_id or not equipment_data:
                raise ValueError("Equipment ID and data are required")
            
            # Extract features for prediction
            features = self._extract_failure_features(equipment_data)
            
            # Run failure prediction model
            model = self.models[PredictionModelType.FAILURE_PREDICTION]
            probability, confidence = self._run_failure_model(model, features)
            
            # Calculate predicted failure time
            predicted_failure_time = None
            if probability > self.failure_probability_threshold:
                predicted_failure_time = datetime.now() + timedelta(hours=int(time_horizon_hours * (1 - probability)))
            
            # Identify contributing factors
            contributing_factors = self._identify_failure_factors(features, equipment_data)
            
            # Generate recommendation
            recommendation = self._generate_failure_recommendation(probability, confidence, predicted_failure_time)
            
            # Create result
            computation_time = (time.time() - start_time) * 1000
            result = PredictionResult(
                equipment_id=equipment_id,
                prediction_type=PredictionModelType.FAILURE_PREDICTION,
                probability=probability,
                confidence=confidence,
                time_horizon_hours=time_horizon_hours,
                predicted_failure_time=predicted_failure_time,
                contributing_factors=contributing_factors,
                recommendation=recommendation,
                computation_time_ms=computation_time
            )
            
            # Update performance metrics
            self._update_prediction_metrics(computation_time, True)
            
            return result
            
        except Exception as e:
            logging.error(f"Equipment failure prediction failed for {equipment_id}: {e}")
            computation_time = (time.time() - start_time) * 1000
            self._update_prediction_metrics(computation_time, False)
            
            return PredictionResult(
                equipment_id=equipment_id,
                prediction_type=PredictionModelType.FAILURE_PREDICTION,
                probability=0.0,
                confidence=0.0,
                time_horizon_hours=time_horizon_hours,
                predicted_failure_time=None,
                contributing_factors=[],
                recommendation="Unable to generate prediction - insufficient data",
                computation_time_ms=computation_time
            )

    def detect_performance_anomalies(self, 
                                   equipment_id: str,
                                   current_metrics: Dict[str, float]) -> List[AnomalyDetection]:
        """Detect performance anomalies in real-time equipment metrics."""
        start_time = time.time()
        anomalies = []
        
        try:
            # Get baseline metrics for comparison
            baseline = self._get_baseline_metrics(equipment_id)
            if not baseline:
                logging.warning(f"No baseline metrics for {equipment_id} - building baseline")
                self._update_baseline_metrics(equipment_id, current_metrics)
                return []
            
            # Check each metric for anomalies
            for metric_name, current_value in current_metrics.items():
                if metric_name not in baseline:
                    continue
                
                baseline_value = baseline[metric_name]['mean']
                baseline_std = baseline[metric_name]['std']
                
                # Calculate z-score for anomaly detection
                if baseline_std > 0:
                    z_score = abs(current_value - baseline_value) / baseline_std
                    
                    if z_score > self.anomaly_threshold:
                        # Anomaly detected
                        anomaly = AnomalyDetection(
                            equipment_id=equipment_id,
                            anomaly_type=self._classify_anomaly_type(metric_name, current_value, baseline_value),
                            severity=min(1.0, z_score / (self.anomaly_threshold * 2)),
                            detected_at=datetime.now(),
                            affected_metrics=[metric_name],
                            anomaly_score=z_score,
                            threshold_violated=z_score > self.anomaly_threshold,
                            description=f"{metric_name} anomaly: {current_value:.2f} vs baseline {baseline_value:.2f}"
                        )
                        anomalies.append(anomaly)
            
            # Pattern-based anomaly detection
            pattern_anomalies = self._detect_pattern_anomalies(equipment_id, current_metrics)
            anomalies.extend(pattern_anomalies)
            
            # Store detected anomalies
            if anomalies:
                if equipment_id not in self.anomaly_history:
                    self.anomaly_history[equipment_id] = []
                self.anomaly_history[equipment_id].extend(anomalies)
                self.performance_metrics['anomalies_detected'] += len(anomalies)
            
            # Update baseline with current metrics
            self._update_baseline_metrics(equipment_id, current_metrics)
            
            computation_time = (time.time() - start_time) * 1000
            logging.info(f"Anomaly detection for {equipment_id} completed in {computation_time:.2f}ms - {len(anomalies)} anomalies found")
            
            return anomalies
            
        except Exception as e:
            logging.error(f"Anomaly detection failed for {equipment_id}: {e}")
            return []

    def recommend_maintenance_schedule(self,
                                     equipment_status: Dict[str, Dict[str, Any]],
                                     production_schedule: Dict[str, Any]) -> List[MaintenanceRecommendation]:
        """Generate AI-powered maintenance scheduling recommendations."""
        start_time = time.time()
        recommendations = []
        
        try:
            for equipment_id, status_data in equipment_status.items():
                # Get failure prediction for this equipment
                failure_prediction = self.predict_equipment_failure(equipment_id, status_data)
                
                # Get recent anomalies
                recent_anomalies = self._get_recent_anomalies(equipment_id, hours=24)
                
                # Calculate maintenance priority
                priority = self._calculate_maintenance_priority(failure_prediction, recent_anomalies, status_data)
                
                if priority >= 3:  # Only recommend high-priority maintenance
                    # Determine optimal maintenance timing
                    optimal_date = self._find_optimal_maintenance_window(
                        equipment_id, production_schedule, failure_prediction.predicted_failure_time
                    )
                    
                    # Estimate maintenance parameters
                    maintenance_type = self._determine_maintenance_type(failure_prediction, recent_anomalies)
                    duration = self._estimate_maintenance_duration(maintenance_type, status_data)
                    cost = self._estimate_maintenance_cost(maintenance_type, equipment_id)
                    impact = self._assess_production_impact(equipment_id, production_schedule, duration)
                    
                    recommendation = MaintenanceRecommendation(
                        equipment_id=equipment_id,
                        maintenance_type=maintenance_type,
                        priority=priority,
                        recommended_date=optimal_date,
                        estimated_duration_hours=duration,
                        estimated_cost=cost,
                        impact_on_production=impact,
                        justification=self._generate_maintenance_justification(failure_prediction, recent_anomalies)
                    )
                    recommendations.append(recommendation)
            
            # Sort recommendations by priority
            recommendations.sort(key=lambda r: r.priority, reverse=True)
            
            # Update performance metrics
            self.performance_metrics['maintenance_recommendations'] += len(recommendations)
            
            computation_time = (time.time() - start_time) * 1000
            logging.info(f"Maintenance scheduling completed in {computation_time:.2f}ms - {len(recommendations)} recommendations")
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Maintenance scheduling failed: {e}")
            return []

    def generate_performance_forecast(self,
                                    equipment_id: str,
                                    metric_name: str,
                                    forecast_horizon_hours: int = 168) -> PerformanceForecast:
        """Generate performance forecast for specified equipment metric."""
        start_time = time.time()
        
        try:
            # Get historical data
            history = self._get_equipment_history(equipment_id)
            if len(history) < 10:  # Need minimum history for forecasting
                logging.warning(f"Insufficient history for {equipment_id} forecast")
                return self._create_default_forecast(equipment_id, metric_name, forecast_horizon_hours)
            
            # Extract metric values and timestamps
            metric_values = [record.get(metric_name, 0) for record in history if metric_name in record]
            if len(metric_values) < 5:
                return self._create_default_forecast(equipment_id, metric_name, forecast_horizon_hours)
            
            # Simple time series forecasting (in production, would use advanced models)
            forecast_points = forecast_horizon_hours // 4  # Forecast every 4 hours
            predicted_values = []
            confidence_intervals = []
            
            # Calculate trend
            recent_values = metric_values[-10:]
            trend_slope = self._calculate_trend_slope(recent_values)
            
            # Generate forecast
            last_value = metric_values[-1]
            for i in range(forecast_points):
                # Simple linear trend with noise
                predicted_value = last_value + (trend_slope * (i + 1))
                noise_factor = np.std(recent_values) * 0.1 if len(recent_values) > 1 else 0.1
                
                predicted_values.append(predicted_value)
                confidence_intervals.append((
                    predicted_value - 1.96 * noise_factor,
                    predicted_value + 1.96 * noise_factor
                ))
            
            # Determine trend direction
            if trend_slope > 0.1:
                trend_direction = "improving"
            elif trend_slope < -0.1:
                trend_direction = "degrading"
            else:
                trend_direction = "stable"
            
            # Calculate forecast accuracy based on historical performance
            accuracy = self._calculate_forecast_accuracy(metric_values)
            
            computation_time = (time.time() - start_time) * 1000
            
            return PerformanceForecast(
                equipment_id=equipment_id,
                metric_name=metric_name,
                forecast_horizon_hours=forecast_horizon_hours,
                predicted_values=predicted_values,
                confidence_intervals=confidence_intervals,
                trend=trend_direction,
                forecast_accuracy=accuracy
            )
            
        except Exception as e:
            logging.error(f"Performance forecast failed for {equipment_id}.{metric_name}: {e}")
            return self._create_default_forecast(equipment_id, metric_name, forecast_horizon_hours)

    def _create_failure_model(self) -> Dict[str, Any]:
        """Create failure prediction model (simplified)."""
        return {
            'model_type': 'failure_prediction',
            'version': '1.0',
            'features': ['temperature', 'vibration', 'pressure', 'runtime_hours', 'error_rate'],
            'weights': [0.3, 0.25, 0.2, 0.15, 0.1]
        }

    def _create_anomaly_model(self) -> Dict[str, Any]:
        """Create anomaly detection model."""
        return {
            'model_type': 'anomaly_detection',
            'version': '1.0',
            'method': 'statistical',
            'threshold': self.anomaly_threshold
        }

    def _create_forecast_model(self) -> Dict[str, Any]:
        """Create performance forecasting model."""
        return {
            'model_type': 'performance_forecast',
            'version': '1.0',
            'method': 'time_series',
            'horizon_hours': 168
        }

    def _create_maintenance_model(self) -> Dict[str, Any]:
        """Create maintenance optimization model."""
        return {
            'model_type': 'maintenance_optimization',
            'version': '1.0',
            'priority_weights': {
                'failure_probability': 0.4,
                'anomaly_severity': 0.3,
                'production_impact': 0.2,
                'cost_factor': 0.1
            }
        }

    def _extract_failure_features(self, equipment_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for failure prediction model."""
        features = {}
        
        # Temperature features
        features['temperature'] = equipment_data.get('temperature', 25.0)
        features['temperature_variance'] = equipment_data.get('temperature_variance', 1.0)
        
        # Vibration features
        features['vibration'] = equipment_data.get('vibration_level', 0.5)
        features['vibration_frequency'] = equipment_data.get('vibration_frequency', 60.0)
        
        # Pressure features
        features['pressure'] = equipment_data.get('pressure', 101.3)
        features['pressure_drop'] = equipment_data.get('pressure_drop', 0.0)
        
        # Runtime features
        features['runtime_hours'] = equipment_data.get('runtime_hours', 0.0)
        features['cycles_completed'] = equipment_data.get('cycles_completed', 0)
        
        # Error rate features
        features['error_rate'] = equipment_data.get('error_rate', 0.0)
        features['warning_count'] = equipment_data.get('warning_count', 0)
        
        return features

    def _run_failure_model(self, model: Dict[str, Any], features: Dict[str, float]) -> Tuple[float, float]:
        """Run failure prediction model (simplified implementation)."""
        model_features = model.get('features', [])
        weights = model.get('weights', [])
        
        # Normalize features and calculate weighted score
        score = 0.0
        total_weight = 0.0
        
        for i, feature_name in enumerate(model_features):
            if feature_name in features and i < len(weights):
                # Normalize feature value (simplified)
                normalized_value = min(1.0, features[feature_name] / 100.0)
                score += normalized_value * weights[i]
                total_weight += weights[i]
        
        if total_weight > 0:
            probability = score / total_weight
        else:
            probability = 0.0
        
        # Calculate confidence based on feature completeness
        confidence = len([f for f in model_features if f in features]) / len(model_features)
        
        # Add some randomness for realistic simulation
        probability += random.uniform(-0.05, 0.05)
        confidence += random.uniform(-0.05, 0.05)
        
        return max(0.0, min(1.0, probability)), max(0.0, min(1.0, confidence))

    def _identify_failure_factors(self, features: Dict[str, float], equipment_data: Dict[str, Any]) -> List[str]:
        """Identify contributing factors to failure prediction."""
        factors = []
        
        # Check for high-risk conditions
        if features.get('temperature', 0) > 80:
            factors.append("High operating temperature")
        
        if features.get('vibration', 0) > 2.0:
            factors.append("Excessive vibration levels")
        
        if features.get('error_rate', 0) > 0.05:
            factors.append("High error rate")
        
        if features.get('runtime_hours', 0) > 8760:  # More than 1 year
            factors.append("Extended runtime hours")
        
        if features.get('pressure_drop', 0) > 5.0:
            factors.append("Significant pressure drop")
        
        if not factors:
            factors.append("Normal operating conditions")
        
        return factors

    def _generate_failure_recommendation(self, probability: float, confidence: float, predicted_failure_time: Optional[datetime]) -> str:
        """Generate failure prediction recommendation."""
        if probability > 0.8 and confidence > 0.7:
            if predicted_failure_time:
                return f"High failure risk - schedule immediate maintenance before {predicted_failure_time.strftime('%Y-%m-%d %H:%M')}"
            else:
                return "High failure risk - schedule immediate preventive maintenance"
        elif probability > 0.6:
            return "Moderate failure risk - schedule maintenance within 2 weeks"
        elif probability > 0.3:
            return "Low failure risk - continue monitoring, schedule routine maintenance"
        else:
            return "Normal operation - continue standard maintenance schedule"

    def _get_baseline_metrics(self, equipment_id: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Get baseline metrics for equipment."""
        return self.baseline_metrics.get(equipment_id)

    def _update_baseline_metrics(self, equipment_id: str, current_metrics: Dict[str, float]):
        """Update baseline metrics with current data."""
        if equipment_id not in self.baseline_metrics:
            self.baseline_metrics[equipment_id] = {}
        
        if equipment_id not in self.equipment_history:
            self.equipment_history[equipment_id] = []
        
        # Add current metrics to history
        history_entry = current_metrics.copy()
        history_entry['timestamp'] = time.time()
        self.equipment_history[equipment_id].append(history_entry)
        
        # Keep only recent history (last 1000 entries)
        if len(self.equipment_history[equipment_id]) > 1000:
            self.equipment_history[equipment_id] = self.equipment_history[equipment_id][-1000:]
        
        # Update baseline statistics
        for metric_name, value in current_metrics.items():
            if metric_name not in self.baseline_metrics[equipment_id]:
                self.baseline_metrics[equipment_id][metric_name] = {
                    'mean': value,
                    'std': 0.0,
                    'count': 1,
                    'sum': value,
                    'sum_sq': value * value
                }
            else:
                baseline = self.baseline_metrics[equipment_id][metric_name]
                baseline['count'] += 1
                baseline['sum'] += value
                baseline['sum_sq'] += value * value
                baseline['mean'] = baseline['sum'] / baseline['count']
                
                if baseline['count'] > 1:
                    variance = (baseline['sum_sq'] / baseline['count']) - (baseline['mean'] ** 2)
                    baseline['std'] = math.sqrt(max(0, variance))

    def _classify_anomaly_type(self, metric_name: str, current_value: float, baseline_value: float) -> AnomalyType:
        """Classify the type of anomaly detected."""
        if current_value < baseline_value:
            return AnomalyType.PERFORMANCE_DEGRADATION
        elif abs(current_value - baseline_value) / baseline_value > 0.5:
            return AnomalyType.THRESHOLD_VIOLATION
        else:
            return AnomalyType.UNUSUAL_PATTERN

    def _detect_pattern_anomalies(self, equipment_id: str, current_metrics: Dict[str, float]) -> List[AnomalyDetection]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        # Get recent history
        history = self._get_equipment_history(equipment_id)
        if len(history) < 5:
            return anomalies
        
        # Check for trend anomalies
        for metric_name in current_metrics.keys():
            recent_values = [record.get(metric_name, 0) for record in history[-5:] if metric_name in record]
            if len(recent_values) >= 3:
                trend = self._calculate_trend_slope(recent_values)
                if abs(trend) > 2.0:  # Significant trend
                    anomaly = AnomalyDetection(
                        equipment_id=equipment_id,
                        anomaly_type=AnomalyType.TREND_ANOMALY,
                        severity=min(1.0, abs(trend) / 5.0),
                        detected_at=datetime.now(),
                        affected_metrics=[metric_name],
                        anomaly_score=abs(trend),
                        threshold_violated=False,
                        description=f"Significant trend detected in {metric_name}: {trend:.2f}"
                    )
                    anomalies.append(anomaly)
        
        return anomalies

    def _get_recent_anomalies(self, equipment_id: str, hours: int) -> List[AnomalyDetection]:
        """Get recent anomalies for equipment within specified time window."""
        if equipment_id not in self.anomaly_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [anomaly for anomaly in self.anomaly_history[equipment_id] if anomaly.detected_at > cutoff_time]

    def _calculate_maintenance_priority(self, 
                                      failure_prediction: PredictionResult,
                                      recent_anomalies: List[AnomalyDetection],
                                      status_data: Dict[str, Any]) -> int:
        """Calculate maintenance priority (1-5 scale)."""
        priority_score = 1.0
        
        # Failure probability factor
        priority_score += failure_prediction.probability * 3.0
        
        # Anomaly severity factor
        if recent_anomalies:
            avg_severity = sum(a.severity for a in recent_anomalies) / len(recent_anomalies)
            priority_score += avg_severity * 2.0
        
        # Equipment criticality factor
        criticality = status_data.get('criticality', 'medium')
        if criticality == 'high':
            priority_score += 1.0
        elif criticality == 'critical':
            priority_score += 2.0
        
        return min(5, int(priority_score))

    def _find_optimal_maintenance_window(self,
                                       equipment_id: str,
                                       production_schedule: Dict[str, Any],
                                       predicted_failure_time: Optional[datetime]) -> datetime:
        """Find optimal maintenance window considering production schedule."""
        # Default to next maintenance window (simplified logic)
        if predicted_failure_time:
            # Schedule maintenance before predicted failure
            return predicted_failure_time - timedelta(hours=24)
        else:
            # Schedule within next week during low production period
            return datetime.now() + timedelta(days=7)

    def _determine_maintenance_type(self, 
                                  failure_prediction: PredictionResult,
                                  recent_anomalies: List[AnomalyDetection]) -> str:
        """Determine type of maintenance required."""
        if failure_prediction.probability > 0.8:
            return "Emergency Repair"
        elif failure_prediction.probability > 0.6:
            return "Preventive Maintenance"
        elif recent_anomalies:
            return "Diagnostic Check"
        else:
            return "Routine Maintenance"

    def _estimate_maintenance_duration(self, maintenance_type: str, status_data: Dict[str, Any]) -> float:
        """Estimate maintenance duration in hours."""
        base_durations = {
            "Emergency Repair": 8.0,
            "Preventive Maintenance": 4.0,
            "Diagnostic Check": 2.0,
            "Routine Maintenance": 1.0
        }
        return base_durations.get(maintenance_type, 2.0)

    def _estimate_maintenance_cost(self, maintenance_type: str, equipment_id: str) -> float:
        """Estimate maintenance cost."""
        base_costs = {
            "Emergency Repair": 5000.0,
            "Preventive Maintenance": 1500.0,
            "Diagnostic Check": 500.0,
            "Routine Maintenance": 200.0
        }
        return base_costs.get(maintenance_type, 1000.0)

    def _assess_production_impact(self, equipment_id: str, production_schedule: Dict[str, Any], duration_hours: float) -> str:
        """Assess impact on production schedule."""
        if duration_hours > 8:
            return "High - Full day production loss"
        elif duration_hours > 4:
            return "Medium - Half day production impact"
        else:
            return "Low - Minimal production impact"

    def _generate_maintenance_justification(self,
                                          failure_prediction: PredictionResult,
                                          recent_anomalies: List[AnomalyDetection]) -> str:
        """Generate justification for maintenance recommendation."""
        reasons = []
        
        if failure_prediction.probability > 0.6:
            reasons.append(f"High failure probability ({failure_prediction.probability:.1%})")
        
        if recent_anomalies:
            reasons.append(f"{len(recent_anomalies)} performance anomalies detected")
        
        if failure_prediction.contributing_factors:
            reasons.append(f"Risk factors: {', '.join(failure_prediction.contributing_factors[:2])}")
        
        return "; ".join(reasons) if reasons else "Preventive maintenance scheduling"

    def _get_equipment_history(self, equipment_id: str) -> List[Dict[str, Any]]:
        """Get equipment history."""
        return self.equipment_history.get(equipment_id, [])

    def _create_default_forecast(self, equipment_id: str, metric_name: str, forecast_horizon_hours: int) -> PerformanceForecast:
        """Create default forecast when insufficient data is available."""
        forecast_points = forecast_horizon_hours // 4
        return PerformanceForecast(
            equipment_id=equipment_id,
            metric_name=metric_name,
            forecast_horizon_hours=forecast_horizon_hours,
            predicted_values=[50.0] * forecast_points,  # Default stable forecast
            confidence_intervals=[(40.0, 60.0)] * forecast_points,
            trend="stable",
            forecast_accuracy=0.5
        )

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope for time series data."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression slope calculation
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

    def _calculate_forecast_accuracy(self, metric_values: List[float]) -> float:
        """Calculate forecast accuracy based on historical performance."""
        if len(metric_values) < 10:
            return 0.5  # Default accuracy
        
        # Simplified accuracy calculation
        recent_variance = np.var(metric_values[-10:]) if len(metric_values) >= 10 else 1.0
        overall_variance = np.var(metric_values)
        
        if overall_variance == 0:
            return 0.9
        
        stability_ratio = min(1.0, recent_variance / overall_variance)
        return max(0.3, 0.9 - stability_ratio * 0.4)

    def _update_prediction_metrics(self, computation_time: float, success: bool):
        """Update prediction performance metrics."""
        self.performance_metrics['total_predictions'] += 1
        
        if success:
            self.performance_metrics['successful_predictions'] += 1
        
        # Update average computation time
        total_preds = self.performance_metrics['total_predictions']
        current_avg = self.performance_metrics['avg_prediction_time_ms']
        self.performance_metrics['avg_prediction_time_ms'] = (
            (current_avg * (total_preds - 1) + computation_time) / total_preds
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics['baseline_equipments'] = len(self.baseline_metrics)
        metrics['total_anomaly_history'] = sum(len(anomalies) for anomalies in self.anomaly_history.values())
        return metrics

    def clear_prediction_history(self):
        """Clear prediction history to free memory."""
        self.equipment_history.clear()
        self.anomaly_history.clear()
        self.baseline_metrics.clear()
        logging.info("Prediction history cleared")

    def __str__(self) -> str:
        return f"PredictiveEngine(target={self.performance_target_ms}ms, equipments={len(self.baseline_metrics)})"

    def __repr__(self) -> str:
        return self.__str__()