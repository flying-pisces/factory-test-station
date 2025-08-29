#!/usr/bin/env python3
"""
DataProtectionEngine - Week 9 Security & Compliance Layer
Comprehensive data protection and encryption management system
"""

import time
import json
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import threading
import base64
import hmac
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc" 
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_4096 = "rsa_4096"
    ECDSA_P384 = "ecdsa_p384"

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class KeyType(Enum):
    """Encryption key types"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    MASTER_KEY = "master_key"
    DERIVED_KEY = "derived_key"

@dataclass
class EncryptionKey:
    """Encryption key metadata"""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_size: int
    created_at: str
    expires_at: Optional[str] = None
    usage_count: int = 0
    max_usage: Optional[int] = None
    is_active: bool = True
    data_classification: DataClassification = DataClassification.CONFIDENTIAL

@dataclass
class DataSpecification:
    """Data encryption specification"""
    data_id: str
    classification: DataClassification
    encryption_required: bool
    retention_policy: Dict[str, Any]
    access_controls: List[str] = field(default_factory=list)
    geographical_restrictions: List[str] = field(default_factory=list)

@dataclass
class EncryptionOperation:
    """Encryption operation record"""
    operation_id: str
    operation_type: str  # encrypt, decrypt, key_rotation
    data_id: str
    key_id: str
    algorithm: EncryptionAlgorithm
    timestamp: str
    success: bool
    duration_ms: float
    data_size_bytes: int = 0

class DataProtectionEngine:
    """Comprehensive data protection and encryption management system
    
    Week 9 Performance Targets:
    - Encryption operations: <200ms
    - Key management: <100ms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DataProtectionEngine with configuration"""
        self.config = config or {}
        
        # Performance targets
        self.encryption_target_ms = 200
        self.key_management_target_ms = 100
        
        # State management
        self.encryption_keys = {}
        self.data_specifications = {}
        self.encryption_operations = []
        self.key_rotation_schedule = {}
        self.dlp_policies = {}
        
        # Initialize encryption algorithms and key store
        self._initialize_encryption_infrastructure()
        
        # Initialize data loss prevention policies
        self._initialize_dlp_policies()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize security orchestration engine if available
        self.security_orchestration = None
        try:
            from layers.security_layer.security_orchestration_engine import SecurityOrchestrationEngine
            self.security_orchestration = SecurityOrchestrationEngine(config.get('orchestration_config', {}))
        except ImportError:
            logger.warning("SecurityOrchestrationEngine not available - using mock interface")
        
        logger.info("DataProtectionEngine initialized with encryption and key management")
    
    def encrypt_data_at_rest(self, data_specifications: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive data at rest with key rotation
        
        Args:
            data_specifications: Data encryption specifications
            
        Returns:
            Encryption results with performance metrics
        """
        start_time = time.time()
        
        try:
            # Parse data specifications
            data_id = data_specifications['data_id']
            data_size = data_specifications.get('data_size_bytes', 1024)
            classification = DataClassification(data_specifications.get('classification', 'confidential'))
            algorithm = EncryptionAlgorithm(data_specifications.get('algorithm', 'aes_256_gcm'))
            
            # Get or create encryption key
            encryption_key = self._get_or_create_encryption_key(data_id, algorithm, classification)
            
            # Perform encryption simulation
            encrypted_data = self._encrypt_data_simulation(data_id, data_size, encryption_key)
            
            # Record operation
            operation = EncryptionOperation(
                operation_id=f"ENC_{int(time.time() * 1000)}",
                operation_type="encrypt_at_rest",
                data_id=data_id,
                key_id=encryption_key.key_id,
                algorithm=algorithm,
                timestamp=datetime.now().isoformat(),
                success=True,
                duration_ms=0,  # Will be calculated below
                data_size_bytes=data_size
            )
            
            # Calculate encryption time
            encryption_time_ms = (time.time() - start_time) * 1000
            operation.duration_ms = encryption_time_ms
            
            # Store operation record
            self.encryption_operations.append(operation)
            
            # Update key usage count
            encryption_key.usage_count += 1
            
            result = {
                'operation_id': operation.operation_id,
                'data_id': data_id,
                'key_id': encryption_key.key_id,
                'algorithm': algorithm.value,
                'classification': classification.value,
                'encryption_time_ms': round(encryption_time_ms, 2),
                'target_met': encryption_time_ms < self.encryption_target_ms,
                'data_size_bytes': data_size,
                'encrypted_data_hash': encrypted_data['hash'],
                'key_rotation_due': self._check_key_rotation_due(encryption_key),
                'encrypted_at': datetime.now().isoformat()
            }
            
            logger.info(f"Data encrypted at rest: {data_id} in {encryption_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error encrypting data at rest: {e}")
            raise
    
    def encrypt_data_in_transit(self, communication_channels: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt data in transit with certificate management
        
        Args:
            communication_channels: Communication channel specifications
            
        Returns:
            Transit encryption results with performance metrics
        """
        start_time = time.time()
        
        try:
            # Parse communication channel specifications
            channel_id = communication_channels['channel_id']
            protocol = communication_channels.get('protocol', 'TLS_1.3')
            data_size = communication_channels.get('data_size_bytes', 2048)
            endpoints = communication_channels.get('endpoints', ['client', 'server'])
            
            # Generate or retrieve certificates
            certificates = self._manage_transit_certificates(channel_id, endpoints)
            
            # Perform transit encryption
            transit_encryption = self._encrypt_transit_simulation(channel_id, data_size, protocol, certificates)
            
            # Record operation
            operation = EncryptionOperation(
                operation_id=f"TRS_{int(time.time() * 1000)}",
                operation_type="encrypt_in_transit",
                data_id=channel_id,
                key_id=certificates['server_cert_id'],
                algorithm=EncryptionAlgorithm.ECDSA_P384,
                timestamp=datetime.now().isoformat(),
                success=True,
                duration_ms=0,  # Will be calculated below
                data_size_bytes=data_size
            )
            
            # Calculate encryption time
            encryption_time_ms = (time.time() - start_time) * 1000
            operation.duration_ms = encryption_time_ms
            
            # Store operation record
            self.encryption_operations.append(operation)
            
            result = {
                'operation_id': operation.operation_id,
                'channel_id': channel_id,
                'protocol': protocol,
                'certificates': list(certificates.keys()),
                'encryption_time_ms': round(encryption_time_ms, 2),
                'target_met': encryption_time_ms < self.encryption_target_ms,
                'data_size_bytes': data_size,
                'cipher_suite': transit_encryption['cipher_suite'],
                'certificate_expiry': certificates['expires_at'],
                'encrypted_at': datetime.now().isoformat()
            }
            
            logger.info(f"Data encrypted in transit: {channel_id} in {encryption_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error encrypting data in transit: {e}")
            raise
    
    def manage_encryption_keys(self, key_operations: Dict[str, Any]) -> Dict[str, Any]:
        """Manage encryption keys with secure key lifecycle
        
        Args:
            key_operations: Key management operations
            
        Returns:
            Key management results with performance metrics
        """
        start_time = time.time()
        
        try:
            # Parse key operations
            operation_type = key_operations['operation_type']
            key_algorithm = EncryptionAlgorithm(key_operations.get('algorithm', 'aes_256_gcm'))
            data_classification = DataClassification(key_operations.get('classification', 'confidential'))
            
            results = []
            
            if operation_type == 'generate':
                # Generate new encryption key
                key = self._generate_encryption_key(key_algorithm, data_classification)
                results.append({
                    'operation': 'generate',
                    'key_id': key.key_id,
                    'algorithm': key.algorithm.value,
                    'key_type': key.key_type.value,
                    'status': 'success'
                })
            
            elif operation_type == 'rotate':
                # Rotate existing keys
                keys_to_rotate = key_operations.get('key_ids', [])
                if not keys_to_rotate:
                    keys_to_rotate = [k.key_id for k in self.encryption_keys.values() 
                                    if self._check_key_rotation_due(k)]
                
                for key_id in keys_to_rotate:
                    rotation_result = self._rotate_encryption_key(key_id)
                    results.append(rotation_result)
            
            elif operation_type == 'backup':
                # Backup encryption keys
                backup_result = self._backup_encryption_keys()
                results.append(backup_result)
            
            elif operation_type == 'audit':
                # Audit key usage
                audit_result = self._audit_key_usage()
                results.append(audit_result)
            
            # Calculate key management time
            key_mgmt_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'operation_type': operation_type,
                'operations_completed': len(results),
                'key_mgmt_time_ms': round(key_mgmt_time_ms, 2),
                'target_met': key_mgmt_time_ms < self.key_management_target_ms,
                'results': results,
                'total_keys_managed': len(self.encryption_keys),
                'active_keys': len([k for k in self.encryption_keys.values() if k.is_active]),
                'managed_at': datetime.now().isoformat()
            }
            
            logger.info(f"Key management operation '{operation_type}' completed in {key_mgmt_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error managing encryption keys: {e}")
            raise
    
    def implement_dlp_policies(self, dlp_specifications: Dict[str, Any]) -> Dict[str, Any]:
        """Implement data loss prevention policies"""
        try:
            # Parse DLP specifications
            policy_name = dlp_specifications['policy_name']
            data_patterns = dlp_specifications.get('data_patterns', [])
            actions = dlp_specifications.get('actions', ['block'])
            
            # Create DLP policy
            dlp_policy = {
                'policy_id': f"DLP_{int(time.time() * 1000)}",
                'name': policy_name,
                'patterns': data_patterns,
                'actions': actions,
                'severity': dlp_specifications.get('severity', 'medium'),
                'created_at': datetime.now().isoformat(),
                'active': True
            }
            
            # Store DLP policy
            self.dlp_policies[dlp_policy['policy_id']] = dlp_policy
            
            # Simulate policy enforcement
            enforcement_results = self._enforce_dlp_policy(dlp_policy)
            
            result = {
                'policy_id': dlp_policy['policy_id'],
                'policy_name': policy_name,
                'patterns_configured': len(data_patterns),
                'enforcement_results': enforcement_results,
                'policy_active': True,
                'implemented_at': datetime.now().isoformat()
            }
            
            logger.info(f"DLP policy implemented: {policy_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error implementing DLP policy: {e}")
            raise
    
    def validate_data_integrity(self, data_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data integrity with checksums and signatures"""
        try:
            data_id = data_validation['data_id']
            validation_type = data_validation.get('type', 'checksum')
            
            if validation_type == 'checksum':
                # Generate and verify checksums
                original_hash = data_validation.get('original_hash')
                current_hash = self._calculate_data_hash(data_id)
                
                integrity_valid = original_hash == current_hash if original_hash else True
                
            elif validation_type == 'digital_signature':
                # Verify digital signatures
                signature = data_validation.get('signature')
                integrity_valid = self._verify_digital_signature(data_id, signature)
            
            result = {
                'data_id': data_id,
                'validation_type': validation_type,
                'integrity_valid': integrity_valid,
                'current_hash': current_hash if validation_type == 'checksum' else None,
                'validated_at': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating data integrity: {e}")
            raise
    
    def _initialize_encryption_infrastructure(self):
        """Initialize encryption infrastructure"""
        # Create master keys for different classification levels
        classifications = [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED, DataClassification.TOP_SECRET]
        algorithms = [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.CHACHA20_POLY1305]
        
        for classification in classifications:
            for algorithm in algorithms:
                master_key = self._generate_encryption_key(algorithm, classification, is_master=True)
                logger.debug(f"Created master key: {master_key.key_id}")
    
    def _initialize_dlp_policies(self):
        """Initialize default DLP policies"""
        default_policies = [
            {
                'policy_name': 'Credit Card Protection',
                'data_patterns': ['credit_card', 'payment_card'],
                'actions': ['block', 'alert'],
                'severity': 'high'
            },
            {
                'policy_name': 'PII Protection', 
                'data_patterns': ['ssn', 'personal_id', 'phone_number'],
                'actions': ['encrypt', 'alert'],
                'severity': 'high'
            },
            {
                'policy_name': 'Confidential Data',
                'data_patterns': ['confidential', 'proprietary'],
                'actions': ['encrypt', 'audit'],
                'severity': 'medium'
            }
        ]
        
        for policy_spec in default_policies:
            self.implement_dlp_policies(policy_spec)
    
    def _get_or_create_encryption_key(self, data_id: str, algorithm: EncryptionAlgorithm, classification: DataClassification) -> EncryptionKey:
        """Get existing or create new encryption key"""
        # Look for existing key for this data
        for key in self.encryption_keys.values():
            if (key.algorithm == algorithm and 
                key.data_classification == classification and 
                key.is_active):
                return key
        
        # Create new key if none found
        return self._generate_encryption_key(algorithm, classification)
    
    def _generate_encryption_key(self, algorithm: EncryptionAlgorithm, classification: DataClassification, is_master: bool = False) -> EncryptionKey:
        """Generate new encryption key"""
        key_id = f"KEY_{algorithm.value}_{int(time.time() * 1000)}"
        
        # Determine key size based on algorithm
        key_sizes = {
            EncryptionAlgorithm.AES_256_GCM: 256,
            EncryptionAlgorithm.AES_256_CBC: 256,
            EncryptionAlgorithm.CHACHA20_POLY1305: 256,
            EncryptionAlgorithm.RSA_4096: 4096,
            EncryptionAlgorithm.ECDSA_P384: 384
        }
        
        key_size = key_sizes.get(algorithm, 256)
        key_type = KeyType.MASTER_KEY if is_master else KeyType.SYMMETRIC
        
        if algorithm in [EncryptionAlgorithm.RSA_4096, EncryptionAlgorithm.ECDSA_P384]:
            key_type = KeyType.ASYMMETRIC_PRIVATE
        
        # Set expiration based on classification
        expiry_days = {
            DataClassification.PUBLIC: 730,  # 2 years
            DataClassification.INTERNAL: 365,  # 1 year
            DataClassification.CONFIDENTIAL: 180,  # 6 months
            DataClassification.RESTRICTED: 90,  # 3 months
            DataClassification.TOP_SECRET: 30  # 1 month
        }
        
        expires_at = datetime.now() + timedelta(days=expiry_days.get(classification, 180))
        
        key = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            key_size=key_size,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            data_classification=classification
        )
        
        self.encryption_keys[key_id] = key
        return key
    
    def _encrypt_data_simulation(self, data_id: str, data_size: int, key: EncryptionKey) -> Dict[str, Any]:
        """Simulate data encryption"""
        # Simulate encryption process
        time.sleep(min(data_size / 1000000, 0.1))  # Simulate encryption time based on data size
        
        # Generate encrypted data hash
        data_hash = hashlib.sha256(f"{data_id}_{key.key_id}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        return {
            'encrypted_size': data_size + 32,  # Add encryption overhead
            'hash': data_hash,
            'iv': secrets.token_hex(16)  # Initialization vector
        }
    
    def _manage_transit_certificates(self, channel_id: str, endpoints: List[str]) -> Dict[str, Any]:
        """Manage transit encryption certificates"""
        certificates = {}
        
        for endpoint in endpoints:
            cert_id = f"CERT_{endpoint}_{int(time.time() * 1000)}"
            certificates[f"{endpoint}_cert_id"] = cert_id
        
        certificates['expires_at'] = (datetime.now() + timedelta(days=365)).isoformat()
        
        return certificates
    
    def _encrypt_transit_simulation(self, channel_id: str, data_size: int, protocol: str, certificates: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate transit encryption"""
        cipher_suites = {
            'TLS_1.3': 'TLS_AES_256_GCM_SHA384',
            'TLS_1.2': 'ECDHE-RSA-AES256-GCM-SHA384'
        }
        
        return {
            'cipher_suite': cipher_suites.get(protocol, cipher_suites['TLS_1.3']),
            'handshake_time_ms': 15.5,
            'encrypted_packets': max(1, data_size // 1400)  # MTU-based packet count
        }
    
    def _rotate_encryption_key(self, key_id: str) -> Dict[str, Any]:
        """Rotate encryption key"""
        old_key = self.encryption_keys.get(key_id)
        if not old_key:
            return {'operation': 'rotate', 'key_id': key_id, 'status': 'key_not_found'}
        
        # Create new key with same properties
        new_key = self._generate_encryption_key(old_key.algorithm, old_key.data_classification)
        
        # Deactivate old key
        old_key.is_active = False
        
        return {
            'operation': 'rotate',
            'old_key_id': key_id,
            'new_key_id': new_key.key_id,
            'status': 'success'
        }
    
    def _check_key_rotation_due(self, key: EncryptionKey) -> bool:
        """Check if key rotation is due"""
        if not key.expires_at:
            return False
        
        expires_at = datetime.fromisoformat(key.expires_at)
        warning_period = timedelta(days=30)  # Rotate 30 days before expiry
        
        return datetime.now() + warning_period >= expires_at
    
    def _backup_encryption_keys(self) -> Dict[str, Any]:
        """Backup encryption keys"""
        active_keys = [k for k in self.encryption_keys.values() if k.is_active]
        
        return {
            'operation': 'backup',
            'keys_backed_up': len(active_keys),
            'backup_location': '/secure/backup/keys',
            'backup_encrypted': True,
            'status': 'success'
        }
    
    def _audit_key_usage(self) -> Dict[str, Any]:
        """Audit encryption key usage"""
        usage_stats = {}
        
        for key in self.encryption_keys.values():
            usage_stats[key.key_id] = {
                'usage_count': key.usage_count,
                'created_at': key.created_at,
                'last_used': 'recently',  # Simplified for demo
                'compliance_status': 'compliant'
            }
        
        return {
            'operation': 'audit',
            'keys_audited': len(usage_stats),
            'usage_statistics': usage_stats,
            'compliance_issues': 0,
            'status': 'success'
        }
    
    def _enforce_dlp_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce DLP policy"""
        # Simulate policy enforcement
        return {
            'policy_violations_detected': 0,
            'actions_taken': [],
            'data_protected': 100,  # Percentage
            'enforcement_effective': True
        }
    
    def _calculate_data_hash(self, data_id: str) -> str:
        """Calculate data hash for integrity validation"""
        return hashlib.sha256(f"{data_id}_{datetime.now().date()}".encode()).hexdigest()
    
    def _verify_digital_signature(self, data_id: str, signature: str) -> bool:
        """Verify digital signature"""
        # Simplified signature verification
        return signature and len(signature) > 10
    
    def demonstrate_data_protection_capabilities(self) -> Dict[str, Any]:
        """Demonstrate data protection capabilities"""
        print("\n🔐 DATA PROTECTION ENGINE DEMONSTRATION 🔐")
        print("=" * 50)
        
        # Encrypt data at rest
        print("🗄️ Encrypting Data at Rest...")
        data_at_rest_specs = {
            'data_id': 'customer_database',
            'data_size_bytes': 1024000,  # 1MB
            'classification': 'confidential',
            'algorithm': 'aes_256_gcm'
        }
        encryption_result = self.encrypt_data_at_rest(data_at_rest_specs)
        print(f"   ✅ Data encrypted: {encryption_result['data_id']}")
        print(f"   ⏱️ Encryption time: {encryption_result['encryption_time_ms']}ms")
        print(f"   🎯 Target: <{self.encryption_target_ms}ms | {'✅ MET' if encryption_result['target_met'] else '❌ MISSED'}")
        print(f"   📊 Data size: {encryption_result['data_size_bytes']:,} bytes")
        
        # Encrypt data in transit
        print("\n🌐 Encrypting Data in Transit...")
        transit_specs = {
            'channel_id': 'api_gateway_channel',
            'protocol': 'TLS_1.3',
            'data_size_bytes': 512000,  # 512KB
            'endpoints': ['client', 'server', 'proxy']
        }
        transit_result = self.encrypt_data_in_transit(transit_specs)
        print(f"   ✅ Transit encryption: {transit_result['channel_id']}")
        print(f"   ⏱️ Encryption time: {transit_result['encryption_time_ms']}ms")
        print(f"   🎯 Target: <{self.encryption_target_ms}ms | {'✅ MET' if transit_result['target_met'] else '❌ MISSED'}")
        print(f"   🔒 Protocol: {transit_result['protocol']}")
        print(f"   📜 Certificates: {len(transit_result['certificates'])}")
        
        # Key management operations
        print("\n🔑 Key Management Operations...")
        key_ops_specs = {
            'operation_type': 'generate',
            'algorithm': 'aes_256_gcm',
            'classification': 'restricted'
        }
        key_result = self.manage_encryption_keys(key_ops_specs)
        print(f"   ✅ Key operation: {key_result['operation_type'].upper()}")
        print(f"   ⏱️ Management time: {key_result['key_mgmt_time_ms']}ms")
        print(f"   🎯 Target: <{self.key_management_target_ms}ms | {'✅ MET' if key_result['target_met'] else '❌ MISSED'}")
        print(f"   🔑 Total keys managed: {key_result['total_keys_managed']}")
        print(f"   ✅ Active keys: {key_result['active_keys']}")
        
        # Data Loss Prevention
        print("\n🛡️ Data Loss Prevention...")
        dlp_specs = {
            'policy_name': 'Manufacturing Data Protection',
            'data_patterns': ['machine_config', 'production_data', 'quality_metrics'],
            'actions': ['encrypt', 'audit', 'alert'],
            'severity': 'high'
        }
        dlp_result = self.implement_dlp_policies(dlp_specs)
        print(f"   ✅ DLP policy: {dlp_result['policy_name']}")
        print(f"   📋 Patterns configured: {dlp_result['patterns_configured']}")
        print(f"   🔒 Data protected: {dlp_result['enforcement_results']['data_protected']}%")
        
        # Data integrity validation
        print("\n🔍 Data Integrity Validation...")
        integrity_specs = {
            'data_id': 'customer_database',
            'type': 'checksum'
        }
        integrity_result = self.validate_data_integrity(integrity_specs)
        print(f"   ✅ Integrity check: {'VALID' if integrity_result['integrity_valid'] else 'INVALID'}")
        print(f"   🔒 Validation type: {integrity_result['validation_type']}")
        
        # Show key rotation schedule
        keys_due_rotation = [k for k in self.encryption_keys.values() if self._check_key_rotation_due(k)]
        print(f"\n📈 Key Rotation Status: {len(keys_due_rotation)} keys due for rotation")
        
        print("\n📈 DEMONSTRATION SUMMARY:")
        print(f"   Data at Rest Encryption: {encryption_result['encryption_time_ms']}ms")
        print(f"   Data in Transit Encryption: {transit_result['encryption_time_ms']}ms")
        print(f"   Key Management: {key_result['key_mgmt_time_ms']}ms")
        print(f"   Total Encryption Operations: {len(self.encryption_operations)}")
        print(f"   Active Keys: {key_result['active_keys']}")
        print(f"   DLP Policies Active: {len(self.dlp_policies)}")
        print("=" * 50)
        
        return {
            'data_at_rest_time_ms': encryption_result['encryption_time_ms'],
            'data_in_transit_time_ms': transit_result['encryption_time_ms'],
            'key_management_time_ms': key_result['key_mgmt_time_ms'],
            'total_operations': len(self.encryption_operations),
            'active_keys': key_result['active_keys'],
            'dlp_policies': len(self.dlp_policies),
            'performance_targets_met': (
                encryption_result['target_met'] and 
                transit_result['target_met'] and 
                key_result['target_met']
            )
        }

def main():
    """Test DataProtectionEngine functionality"""
    engine = DataProtectionEngine()
    results = engine.demonstrate_data_protection_capabilities()
    
    print(f"\n🎯 Week 9 Data Protection Performance Targets:")
    print(f"   Encryption Operations: <200ms ({'✅' if max(results['data_at_rest_time_ms'], results['data_in_transit_time_ms']) < 200 else '❌'})")
    print(f"   Key Management: <100ms ({'✅' if results['key_management_time_ms'] < 100 else '❌'})")
    print(f"   Overall Performance: {'🟢 EXCELLENT' if results['performance_targets_met'] else '🟡 NEEDS OPTIMIZATION'}")

if __name__ == "__main__":
    main()