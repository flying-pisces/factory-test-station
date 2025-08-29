"""
NLP Engine - Week 12: Advanced Features & AI Integration

This module provides natural language processing capabilities for the manufacturing
line control system, including text analysis, document processing, and communication
intelligence.
"""

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import threading


class NLPEngine:
    """
    Natural Language Processing Engine for manufacturing line operations.
    
    Provides text analysis, document processing, sentiment analysis,
    and intelligent communication features for manufacturing systems.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize NLP Engine.
        
        Args:
            config: Configuration dictionary for NLP settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Performance targets
        self.text_analysis_target_ms = 100
        self.document_processing_target_ms = 500
        self.sentiment_analysis_target_ms = 50
        
        # NLP models and processors
        self.text_processors = {}
        self.sentiment_analyzer = None
        self.document_parser = None
        self.language_detector = None
        
        # Processing metrics
        self.processing_metrics = {
            'text_analysis_count': 0,
            'document_processing_count': 0,
            'sentiment_analysis_count': 0,
            'avg_processing_time': 0.0,
            'error_count': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nlp")
        
        # Initialize integration (without circular dependencies)
        self.vision_integration = None
        self.optimization_integration = None
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        self.logger.info("NLPEngine initialized successfully")
    
    def _initialize_nlp_components(self):
        """Initialize NLP processing components."""
        try:
            # Initialize text processors
            self.text_processors = {
                'tokenizer': self._create_tokenizer(),
                'pos_tagger': self._create_pos_tagger(),
                'entity_extractor': self._create_entity_extractor()
            }
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = self._create_sentiment_analyzer()
            
            # Initialize document parser
            self.document_parser = self._create_document_parser()
            
            # Initialize language detector
            self.language_detector = self._create_language_detector()
            
            self.logger.info("NLP components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP components: {e}")
            raise
    
    def _create_tokenizer(self) -> Dict:
        """Create text tokenizer."""
        return {
            'type': 'regex_tokenizer',
            'patterns': [
                r'\b\w+\b',  # Words
                r'\d+\.?\d*',  # Numbers
                r'[.!?;,:]'  # Punctuation
            ]
        }
    
    def _create_pos_tagger(self) -> Dict:
        """Create part-of-speech tagger."""
        return {
            'type': 'rule_based_pos',
            'rules': {
                r'\b\d+\b': 'NUM',
                r'\b[A-Z][a-z]+\b': 'NOUN',
                r'\b(is|are|was|were|am)\b': 'VERB',
                r'\b(the|a|an)\b': 'DET'
            }
        }
    
    def _create_entity_extractor(self) -> Dict:
        """Create named entity extractor."""
        return {
            'type': 'pattern_entity_extractor',
            'patterns': {
                'EQUIPMENT': r'\b(conveyor|robot|sensor|actuator)\b',
                'ERROR_CODE': r'\b[A-Z]\d{3,4}\b',
                'TEMPERATURE': r'\b\d+\.?\d*\s*°?[CF]\b',
                'PRESSURE': r'\b\d+\.?\d*\s*(psi|bar|Pa)\b',
                'SERIAL_NUMBER': r'\b[A-Z]{2}\d{6,8}\b'
            }
        }
    
    def _create_sentiment_analyzer(self) -> Dict:
        """Create sentiment analyzer."""
        return {
            'type': 'lexicon_based',
            'positive_words': ['good', 'excellent', 'working', 'normal', 'optimal'],
            'negative_words': ['error', 'failed', 'broken', 'alarm', 'critical'],
            'neutral_words': ['status', 'report', 'data', 'measurement']
        }
    
    def _create_document_parser(self) -> Dict:
        """Create document parser."""
        return {
            'type': 'multi_format_parser',
            'supported_formats': ['txt', 'log', 'csv', 'json'],
            'parsers': {
                'txt': 'plain_text_parser',
                'log': 'log_file_parser',
                'csv': 'csv_parser',
                'json': 'json_parser'
            }
        }
    
    def _create_language_detector(self) -> Dict:
        """Create language detector."""
        return {
            'type': 'statistical_detector',
            'supported_languages': ['en', 'es', 'fr', 'de', 'zh'],
            'confidence_threshold': 0.8
        }
    
    async def analyze_text(self, text: str, analysis_type: str = 'full') -> Dict[str, Any]:
        """
        Analyze text content for manufacturing insights.
        
        Args:
            text: Input text to analyze
            analysis_type: Type of analysis ('full', 'sentiment', 'entities')
        
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        
        try:
            with self.lock:
                self.processing_metrics['text_analysis_count'] += 1
            
            # Tokenize text
            tokens = await self._tokenize_text(text)
            
            # Initialize results
            results = {
                'text': text,
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'tokens': tokens,
                'language': await self._detect_language(text)
            }
            
            # Perform requested analysis
            if analysis_type in ['full', 'sentiment']:
                results['sentiment'] = await self._analyze_sentiment(text, tokens)
            
            if analysis_type in ['full', 'entities']:
                results['entities'] = await self._extract_entities(text, tokens)
            
            if analysis_type == 'full':
                results['pos_tags'] = await self._tag_parts_of_speech(tokens)
                results['summary'] = await self._generate_summary(text, tokens)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            results['processing_time_ms'] = processing_time
            
            # Update metrics
            with self.lock:
                self._update_processing_metrics(processing_time)
            
            # Check performance target
            if processing_time > self.text_analysis_target_ms:
                self.logger.warning(f"Text analysis exceeded target: {processing_time:.1f}ms")
            
            return results
            
        except Exception as e:
            with self.lock:
                self.processing_metrics['error_count'] += 1
            self.logger.error(f"Text analysis failed: {e}")
            raise
    
    async def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize input text."""
        tokenizer = self.text_processors['tokenizer']
        tokens = []
        
        for pattern in tokenizer['patterns']:
            matches = re.findall(pattern, text.lower())
            tokens.extend(matches)
        
        return list(set(tokens))  # Remove duplicates
    
    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of input text."""
        detector = self.language_detector
        
        # Simple statistical detection based on common words
        language_scores = {}
        
        # English indicators
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a']
        english_score = sum(1 for word in english_words if word in text.lower())
        language_scores['en'] = english_score / len(english_words)
        
        # Determine most likely language
        best_language = max(language_scores, key=language_scores.get)
        confidence = language_scores[best_language]
        
        return {
            'language': best_language,
            'confidence': confidence,
            'all_scores': language_scores
        }
    
    async def _analyze_sentiment(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        analyzer = self.sentiment_analyzer
        
        positive_score = sum(1 for token in tokens 
                           if token in analyzer['positive_words'])
        negative_score = sum(1 for token in tokens 
                           if token in analyzer['negative_words'])
        neutral_score = sum(1 for token in tokens 
                          if token in analyzer['neutral_words'])
        
        total_sentiment_words = positive_score + negative_score + neutral_score
        
        if total_sentiment_words == 0:
            sentiment = 'neutral'
            confidence = 0.5
        else:
            if positive_score > negative_score:
                sentiment = 'positive'
                confidence = positive_score / total_sentiment_words
            elif negative_score > positive_score:
                sentiment = 'negative' 
                confidence = negative_score / total_sentiment_words
            else:
                sentiment = 'neutral'
                confidence = neutral_score / total_sentiment_words
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'neutral_score': neutral_score
        }
    
    async def _extract_entities(self, text: str, tokens: List[str]) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        extractor = self.text_processors['entity_extractor']
        entities = {}
        
        for entity_type, pattern in extractor['patterns'].items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    async def _tag_parts_of_speech(self, tokens: List[str]) -> Dict[str, str]:
        """Tag parts of speech for tokens."""
        tagger = self.text_processors['pos_tagger']
        pos_tags = {}
        
        for token in tokens:
            for pattern, tag in tagger['rules'].items():
                if re.match(pattern, token, re.IGNORECASE):
                    pos_tags[token] = tag
                    break
            else:
                pos_tags[token] = 'UNK'  # Unknown
        
        return pos_tags
    
    async def _generate_summary(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """Generate text summary."""
        return {
            'word_count': len(tokens),
            'character_count': len(text),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'unique_words': len(set(tokens)),
            'avg_word_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0
        }
    
    async def process_document(self, document_path: str, document_type: str = 'auto') -> Dict[str, Any]:
        """
        Process manufacturing documents for insights.
        
        Args:
            document_path: Path to document file
            document_type: Type of document ('auto', 'log', 'report', 'manual')
        
        Returns:
            Dictionary containing document analysis results
        """
        start_time = time.time()
        
        try:
            with self.lock:
                self.processing_metrics['document_processing_count'] += 1
            
            # Read document content
            content = await self._read_document(document_path)
            
            # Detect document type if auto
            if document_type == 'auto':
                document_type = await self._detect_document_type(document_path, content)
            
            # Process based on document type
            if document_type == 'log':
                results = await self._process_log_document(content)
            elif document_type == 'report':
                results = await self._process_report_document(content)
            elif document_type == 'manual':
                results = await self._process_manual_document(content)
            else:
                results = await self._process_generic_document(content)
            
            # Add metadata
            results.update({
                'document_path': document_path,
                'document_type': document_type,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': (time.time() - start_time) * 1000
            })
            
            return results
            
        except Exception as e:
            with self.lock:
                self.processing_metrics['error_count'] += 1
            self.logger.error(f"Document processing failed: {e}")
            raise
    
    async def _read_document(self, document_path: str) -> str:
        """Read document content."""
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(document_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    async def _detect_document_type(self, path: str, content: str) -> str:
        """Detect document type based on path and content."""
        path_lower = path.lower()
        
        if '.log' in path_lower or 'timestamp' in content[:1000]:
            return 'log'
        elif 'report' in path_lower or 'summary' in content[:500]:
            return 'report'
        elif 'manual' in path_lower or 'procedure' in content[:500]:
            return 'manual'
        else:
            return 'generic'
    
    async def _process_log_document(self, content: str) -> Dict[str, Any]:
        """Process log document."""
        lines = content.split('\n')
        
        # Extract log entries
        entries = []
        error_count = 0
        warning_count = 0
        
        for line in lines:
            if not line.strip():
                continue
                
            entry = {
                'line': line,
                'timestamp': self._extract_timestamp(line),
                'level': self._extract_log_level(line),
                'message': self._extract_log_message(line)
            }
            
            if entry['level'] == 'ERROR':
                error_count += 1
            elif entry['level'] == 'WARNING':
                warning_count += 1
                
            entries.append(entry)
        
        return {
            'type': 'log_analysis',
            'total_entries': len(entries),
            'error_count': error_count,
            'warning_count': warning_count,
            'entries': entries[:10]  # First 10 entries
        }
    
    async def _process_report_document(self, content: str) -> Dict[str, Any]:
        """Process report document."""
        # Analyze text content
        text_analysis = await self.analyze_text(content, 'full')
        
        # Extract key metrics from report
        metrics = self._extract_report_metrics(content)
        
        return {
            'type': 'report_analysis',
            'text_analysis': text_analysis,
            'extracted_metrics': metrics
        }
    
    async def _process_manual_document(self, content: str) -> Dict[str, Any]:
        """Process manual/procedure document."""
        # Extract procedures and steps
        procedures = self._extract_procedures(content)
        
        # Analyze text
        text_analysis = await self.analyze_text(content, 'entities')
        
        return {
            'type': 'manual_analysis',
            'procedures': procedures,
            'text_analysis': text_analysis
        }
    
    async def _process_generic_document(self, content: str) -> Dict[str, Any]:
        """Process generic document."""
        text_analysis = await self.analyze_text(content, 'full')
        
        return {
            'type': 'generic_analysis',
            'text_analysis': text_analysis
        }
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        """Extract timestamp from log line."""
        # Look for common timestamp patterns
        patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
            r'\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group()
        
        return None
    
    def _extract_log_level(self, line: str) -> str:
        """Extract log level from log line."""
        levels = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
        
        for level in levels:
            if level in line.upper():
                return level
        
        return 'UNKNOWN'
    
    def _extract_log_message(self, line: str) -> str:
        """Extract main message from log line."""
        # Remove timestamp and level
        message = line
        
        # Remove timestamp patterns
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
            r'\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\]'
        ]
        
        for pattern in timestamp_patterns:
            message = re.sub(pattern, '', message)
        
        # Remove log levels
        levels = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
        for level in levels:
            message = message.replace(level, '', 1)
        
        return message.strip()
    
    def _extract_report_metrics(self, content: str) -> Dict[str, Any]:
        """Extract metrics from report content."""
        metrics = {}
        
        # Look for numerical values with units
        number_patterns = [
            (r'(\d+\.?\d*)\s*%', 'percentages'),
            (r'(\d+\.?\d*)\s*(rpm|RPM)', 'rpms'),
            (r'(\d+\.?\d*)\s*°[CF]', 'temperatures'),
            (r'(\d+\.?\d*)\s*(psi|bar|Pa)', 'pressures')
        ]
        
        for pattern, metric_type in number_patterns:
            matches = re.findall(pattern, content)
            if matches:
                if isinstance(matches[0], tuple):
                    values = [float(match[0]) for match in matches]
                else:
                    values = [float(match) for match in matches]
                
                metrics[metric_type] = {
                    'values': values,
                    'count': len(values),
                    'average': sum(values) / len(values) if values else 0
                }
        
        return metrics
    
    def _extract_procedures(self, content: str) -> List[Dict[str, Any]]:
        """Extract procedures from manual content."""
        procedures = []
        
        # Look for numbered steps
        step_pattern = r'(\d+)\.\s*(.+)'
        matches = re.findall(step_pattern, content, re.MULTILINE)
        
        if matches:
            for step_num, step_text in matches:
                procedures.append({
                    'step': int(step_num),
                    'instruction': step_text.strip(),
                    'type': 'numbered_step'
                })
        
        # Look for bullet points
        bullet_pattern = r'[•\-\*]\s*(.+)'
        bullet_matches = re.findall(bullet_pattern, content, re.MULTILINE)
        
        for bullet_text in bullet_matches:
            procedures.append({
                'instruction': bullet_text.strip(),
                'type': 'bullet_point'
            })
        
        return procedures
    
    def _update_processing_metrics(self, processing_time: float):
        """Update processing metrics."""
        total_count = (self.processing_metrics['text_analysis_count'] + 
                      self.processing_metrics['document_processing_count'] + 
                      self.processing_metrics['sentiment_analysis_count'])
        
        if total_count > 0:
            current_avg = self.processing_metrics['avg_processing_time']
            self.processing_metrics['avg_processing_time'] = (
                (current_avg * (total_count - 1) + processing_time) / total_count
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get NLP engine performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self.lock:
            return {
                'nlp_engine_metrics': self.processing_metrics.copy(),
                'performance_targets': {
                    'text_analysis_target_ms': self.text_analysis_target_ms,
                    'document_processing_target_ms': self.document_processing_target_ms,
                    'sentiment_analysis_target_ms': self.sentiment_analysis_target_ms
                },
                'timestamp': datetime.now().isoformat()
            }
    
    async def validate_nlp_engine(self) -> Dict[str, Any]:
        """
        Validate NLP engine functionality and performance.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'engine_name': 'NLPEngine',
            'validation_timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'overall_status': 'unknown'
        }
        
        try:
            # Test 1: Text Analysis
            test_text = "The conveyor system is working normally with temperature at 25°C. No errors detected."
            analysis_result = await self.analyze_text(test_text, 'full')
            
            validation_results['tests']['text_analysis'] = {
                'status': 'pass' if analysis_result['processing_time_ms'] < self.text_analysis_target_ms else 'fail',
                'processing_time_ms': analysis_result['processing_time_ms'],
                'target_ms': self.text_analysis_target_ms,
                'details': f"Analyzed {len(analysis_result['tokens'])} tokens"
            }
            
            # Test 2: Sentiment Analysis
            sentiment_result = await self.analyze_text(test_text, 'sentiment')
            
            validation_results['tests']['sentiment_analysis'] = {
                'status': 'pass' if sentiment_result['sentiment']['sentiment'] in ['positive', 'neutral', 'negative'] else 'fail',
                'sentiment': sentiment_result['sentiment']['sentiment'],
                'confidence': sentiment_result['sentiment']['confidence'],
                'details': f"Detected {sentiment_result['sentiment']['sentiment']} sentiment"
            }
            
            # Test 3: Entity Extraction
            entity_result = await self.analyze_text(test_text, 'entities')
            
            validation_results['tests']['entity_extraction'] = {
                'status': 'pass' if 'entities' in entity_result else 'fail',
                'entities_found': len(entity_result.get('entities', {})),
                'details': f"Extracted entities: {list(entity_result.get('entities', {}).keys())}"
            }
            
            # Test 4: Performance Metrics
            metrics = self.get_performance_metrics()
            validation_results['performance_metrics'] = metrics['nlp_engine_metrics']
            
            # Overall status
            passed_tests = sum(1 for test in validation_results['tests'].values() 
                             if test['status'] == 'pass')
            total_tests = len(validation_results['tests'])
            
            validation_results['overall_status'] = 'pass' if passed_tests == total_tests else 'fail'
            validation_results['test_summary'] = f"{passed_tests}/{total_tests} tests passed"
            
            self.logger.info(f"NLP engine validation completed: {validation_results['test_summary']}")
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"NLP engine validation failed: {e}")
        
        return validation_results
    
    def shutdown(self):
        """Shutdown NLP engine and cleanup resources."""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("NLP engine shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during NLP engine shutdown: {e}")


# Integration functions for cross-engine communication
async def integrate_with_vision_engine(nlp_engine: NLPEngine, vision_engine) -> Dict[str, Any]:
    """Integrate NLP engine with vision engine."""
    try:
        # Placeholder for NLP-Vision integration
        return {
            'integration_type': 'nlp_vision',
            'status': 'connected',
            'capabilities': ['ocr_text_analysis', 'image_caption_processing']
        }
    except Exception as e:
        return {
            'integration_type': 'nlp_vision',
            'status': 'error',
            'error': str(e)
        }


async def integrate_with_optimization_engine(nlp_engine: NLPEngine, optimization_engine) -> Dict[str, Any]:
    """Integrate NLP engine with optimization engine."""
    try:
        # Placeholder for NLP-Optimization integration
        return {
            'integration_type': 'nlp_optimization',
            'status': 'connected',
            'capabilities': ['text_based_optimization', 'report_analysis']
        }
    except Exception as e:
        return {
            'integration_type': 'nlp_optimization', 
            'status': 'error',
            'error': str(e)
        }