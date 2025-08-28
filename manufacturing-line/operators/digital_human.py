"""Digital human operator implementation with behavioral AI."""

from typing import Dict, Any
import time
import random
from .base_operator import BaseOperator, OperatorCapability, OperatorAction, OperatorStatus


class DigitalHuman(BaseOperator):
    """AI-driven digital human operator for manufacturing line."""
    
    def __init__(self, operator_id: str, config: Dict[str, Any]):
        super().__init__(operator_id, config)
        self.skill_level = config.get('skill_level', 0.8)  # 0.0 - 1.0
        self.attention_level = config.get('attention_level', 0.9)  # 0.0 - 1.0
        self.fatigue_level = 0.0  # Increases over time
        self.learning_rate = 0.01
        
        # Behavioral parameters
        self.reaction_time = config.get('reaction_time', 2.0)  # seconds
        self.error_rate = config.get('base_error_rate', 0.05)
        self.monitoring_stations = config.get('monitoring_stations', [])
        
        # AI decision making
        self.decision_confidence = 0.8
    
    def press_button(self, station: str, parameters: Dict[str, Any]) -> bool:
        """Simulate intelligent button press with human-like behavior."""
        button_id = parameters.get('button', 'unknown')
        
        # Simulate reaction time with variance
        reaction_delay = self.reaction_time * random.uniform(0.8, 1.2)
        time.sleep(reaction_delay)
        
        # Calculate success probability based on skill and fatigue
        success_prob = self.skill_level * (1 - self.fatigue_level) * self.attention_level
        success = random.random() < success_prob
        
        if success:
            print(f"Digital human pressed {button_id} at {station}")
            self.trigger_hook('button_pressed', {
                'station': station,
                'button': button_id,
                'operator': self.operator_id
            })
        else:
            print(f"Failed to press {button_id} at {station} - skill/fatigue issue")
            self.trigger_hook('action_failed', {
                'action': 'button_press',
                'station': station,
                'reason': 'skill_fatigue'
            })
        
        # Learn from experience
        self._update_skill(success)
        return success
    
    def pickup_item(self, station: str, parameters: Dict[str, Any]) -> bool:
        """Pick up items with dexterity simulation."""
        item_type = parameters.get('item', 'unknown')
        item_weight = parameters.get('weight', 0.1)  # kg
        
        # Movement and positioning time
        movement_time = 1.5 + (item_weight * 0.5)
        time.sleep(movement_time)
        
        # Success based on dexterity (skill level affects fine motor control)
        dexterity = self.skill_level * 0.9  # Slightly harder than button press
        success = random.random() < dexterity
        
        if success:
            print(f"Picked up {item_type} at {station}")
            self.trigger_hook('item_picked', {
                'station': station,
                'item': item_type,
                'weight': item_weight
            })
        else:
            print(f"Failed to pick up {item_type} at {station}")
            self.trigger_hook('pickup_failed', {
                'station': station,
                'item': item_type
            })
        
        self._increase_fatigue(0.02)  # Physical tasks increase fatigue
        return success
    
    def visual_inspection(self, station: str, parameters: Dict[str, Any]) -> bool:
        """Perform visual quality inspection with AI decision making."""
        inspection_type = parameters.get('inspection', 'general')
        criteria = parameters.get('criteria', {})
        
        # Inspection takes time based on complexity
        complexity = len(criteria)
        inspection_time = 3.0 + (complexity * 0.5)
        time.sleep(inspection_time)
        
        # Inspection accuracy based on attention and skill
        accuracy = self.attention_level * self.skill_level
        confidence = self.decision_confidence
        
        # Simulate finding defects
        has_defect = random.random() < 0.1  # 10% defect rate
        detected_correctly = random.random() < accuracy
        
        if has_defect and detected_correctly:
            result = 'fail'
            print(f"Defect detected in {inspection_type} at {station}")
        elif not has_defect and detected_correctly:
            result = 'pass'
            print(f"{inspection_type} inspection passed at {station}")
        else:
            # Missed defect or false positive
            result = 'uncertain'
            print(f"Uncertain inspection result for {inspection_type} at {station}")
            confidence *= 0.5
        
        self.trigger_hook('inspection_completed', {
            'station': station,
            'type': inspection_type,
            'result': result,
            'confidence': confidence
        })
        
        self._increase_fatigue(0.01)
        return detected_correctly
    
    def monitor_issue(self, station: str, parameters: Dict[str, Any]) -> bool:
        """Monitor station for issues with pattern recognition."""
        issue_types = parameters.get('issue_types', ['error', 'jam', 'alarm'])
        monitoring_duration = parameters.get('duration', 30.0)
        
        start_time = time.time()
        issues_detected = []
        
        # Continuous monitoring simulation
        while time.time() - start_time < monitoring_duration:
            # Check for issues with attention-based detection rate
            for issue_type in issue_types:
                detection_prob = self.attention_level * 0.8
                if random.random() < detection_prob * 0.1:  # 10% chance per check
                    issues_detected.append({
                        'type': issue_type,
                        'severity': random.choice(['low', 'medium', 'high']),
                        'timestamp': time.time()
                    })
            
            time.sleep(1.0)  # Check every second
        
        if issues_detected:
            print(f"Detected {len(issues_detected)} issues at {station}")
            self.trigger_hook('issues_detected', {
                'station': station,
                'issues': issues_detected
            })
        
        return len(issues_detected) > 0
    
    def _update_skill(self, success: bool):
        """Update skill level based on performance."""
        if success:
            self.skill_level = min(1.0, self.skill_level + self.learning_rate * 0.1)
        else:
            # Learn from failures too, but less
            self.skill_level = min(1.0, self.skill_level + self.learning_rate * 0.05)
    
    def _increase_fatigue(self, amount: float):
        """Increase fatigue level from work."""
        self.fatigue_level = min(0.8, self.fatigue_level + amount)
        
        # Fatigue affects attention
        if self.fatigue_level > 0.5:
            self.attention_level = max(0.3, 0.9 - (self.fatigue_level - 0.5))
    
    def rest(self, duration: float = 600):  # 10 minutes default
        """Rest to reduce fatigue."""
        self.status = OperatorStatus.BREAK
        
        def recover():
            self.fatigue_level = max(0.0, self.fatigue_level - 0.3)
            self.attention_level = min(0.9, self.attention_level + 0.2)
            self.status = OperatorStatus.IDLE
        
        # Schedule recovery
        import threading
        threading.Timer(duration, recover).start()
    
    def adaptive_behavior(self):
        """Adjust behavior based on line conditions."""
        # Increase attention during critical operations
        if self.assigned_station in ['FCT', 'IQC', 'Packaging']:
            self.attention_level = min(1.0, self.attention_level + 0.1)
        
        # Adjust reaction time based on fatigue
        base_reaction = self.config.get('reaction_time', 2.0)
        self.reaction_time = base_reaction * (1 + self.fatigue_level)
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get AI-specific status information."""
        base_status = self.get_status()
        ai_status = {
            'skill_level': self.skill_level,
            'attention_level': self.attention_level,
            'fatigue_level': self.fatigue_level,
            'decision_confidence': self.decision_confidence,
            'current_reaction_time': self.reaction_time
        }
        
        return {**base_status, 'ai_metrics': ai_status}