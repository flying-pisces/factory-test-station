"""
User Documentation System - Week 16: Production Deployment & Documentation

This module provides comprehensive user documentation management including role-based
user guides, interactive help systems, searchable knowledge base, and multi-format
documentation generation for the manufacturing control system.

Documentation Features:
- Role-based user guides for all user types
- Interactive help system with context-sensitive assistance
- Searchable knowledge base with categorization
- Multi-format documentation (HTML, PDF, mobile)
- Video tutorial integration
- Progressive disclosure for complex procedures
- Multi-language support and accessibility compliance

Author: Manufacturing Line Control System
Created: Week 16 - User Documentation Phase
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import uuid
import markdown
import html


class DocumentationType(Enum):
    """Types of documentation."""
    USER_GUIDE = "user_guide"
    QUICK_START = "quick_start"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    FAQ = "faq"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICES = "best_practices"
    RELEASE_NOTES = "release_notes"


class UserRole(Enum):
    """User roles for role-based documentation."""
    PRODUCTION_OPERATOR = "production_operator"
    PRODUCTION_MANAGER = "production_manager"
    MAINTENANCE_TECHNICIAN = "maintenance_technician"
    QUALITY_CONTROLLER = "quality_controller"
    SYSTEM_ADMINISTRATOR = "system_administrator"
    DEVELOPER = "developer"


class DocumentFormat(Enum):
    """Output document formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    MOBILE_RESPONSIVE = "mobile_responsive"
    INTERACTIVE = "interactive"


class ContentDifficulty(Enum):
    """Content difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class DocumentSection:
    """Individual document section."""
    section_id: str
    title: str
    content: str
    order: int = 0
    subsections: List['DocumentSection'] = field(default_factory=list)
    
    # Metadata
    difficulty: ContentDifficulty = ContentDifficulty.INTERMEDIATE
    estimated_read_time_minutes: int = 5
    prerequisites: List[str] = field(default_factory=list)
    related_sections: List[str] = field(default_factory=list)
    
    # Interactive elements
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)
    code_examples: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    videos: List[Dict[str, str]] = field(default_factory=list)
    
    # Tags for search and categorization
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.section_id:
            self.section_id = f"section-{uuid.uuid4().hex[:8]}"


@dataclass
class Document:
    """Complete document definition."""
    document_id: str
    title: str
    description: str
    document_type: DocumentationType
    target_roles: List[UserRole] = field(default_factory=list)
    
    # Content structure
    sections: List[DocumentSection] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    version: str = "1.0"
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    review_date: Optional[datetime] = None
    
    # Access and visibility
    public: bool = True
    required_permissions: List[str] = field(default_factory=list)
    
    # Content organization
    category: str = ""
    difficulty: ContentDifficulty = ContentDifficulty.INTERMEDIATE
    estimated_completion_time_minutes: int = 30
    
    # SEO and discovery
    tags: List[str] = field(default_factory=list)
    search_keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.document_id:
            self.document_id = f"doc-{uuid.uuid4().hex[:8]}"


@dataclass
class HelpContext:
    """Context-sensitive help definition."""
    context_id: str
    page_url: str
    element_selector: Optional[str] = None
    help_content: str = ""
    help_type: str = "tooltip"  # tooltip, modal, sidebar, inline
    trigger_event: str = "hover"  # hover, click, focus
    position: str = "auto"  # top, bottom, left, right, auto
    
    # Content
    title: str = ""
    description: str = ""
    related_docs: List[str] = field(default_factory=list)
    quick_actions: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class SearchResult:
    """Documentation search result."""
    document_id: str
    section_id: Optional[str]
    title: str
    snippet: str
    relevance_score: float
    document_type: DocumentationType
    target_roles: List[UserRole]
    url: str
    highlighted_content: str = ""


class DocumentationGenerator:
    """Generate documentation in various formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_html(self, document: Document, template: Optional[str] = None) -> str:
        """Generate HTML documentation."""
        try:
            html_content = self._build_html_structure(document)
            
            if template:
                # Apply custom template
                html_content = self._apply_template(html_content, template)
            else:
                # Use default template
                html_content = self._apply_default_template(document, html_content)
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error generating HTML for document {document.document_id}: {e}")
            return ""
    
    def generate_markdown(self, document: Document) -> str:
        """Generate Markdown documentation."""
        try:
            md_content = f"# {document.title}\n\n"
            md_content += f"*{document.description}*\n\n"
            
            # Add metadata
            md_content += f"**Version:** {document.version}  \n"
            md_content += f"**Author:** {document.author}  \n"
            md_content += f"**Last Modified:** {document.last_modified.strftime('%Y-%m-%d')}  \n"
            md_content += f"**Estimated Time:** {document.estimated_completion_time_minutes} minutes\n\n"
            
            # Add table of contents
            md_content += self._generate_toc(document.sections)
            md_content += "\n---\n\n"
            
            # Add sections
            for section in document.sections:
                md_content += self._section_to_markdown(section, level=2)
                md_content += "\n"
            
            return md_content
            
        except Exception as e:
            self.logger.error(f"Error generating Markdown for document {document.document_id}: {e}")
            return ""
    
    def generate_mobile_responsive(self, document: Document) -> str:
        """Generate mobile-responsive HTML documentation."""
        try:
            # Generate HTML with mobile-specific styling and structure
            mobile_html = self._build_mobile_html(document)
            return mobile_html
            
        except Exception as e:
            self.logger.error(f"Error generating mobile documentation for document {document.document_id}: {e}")
            return ""
    
    def generate_interactive(self, document: Document) -> Dict[str, Any]:
        """Generate interactive documentation with embedded elements."""
        try:
            interactive_doc = {
                "document_id": document.document_id,
                "title": document.title,
                "description": document.description,
                "metadata": {
                    "version": document.version,
                    "author": document.author,
                    "difficulty": document.difficulty.value,
                    "estimated_time": document.estimated_completion_time_minutes
                },
                "sections": []
            }
            
            for section in document.sections:
                interactive_section = self._section_to_interactive(section)
                interactive_doc["sections"].append(interactive_section)
            
            return interactive_doc
            
        except Exception as e:
            self.logger.error(f"Error generating interactive documentation for document {document.document_id}: {e}")
            return {}
    
    def _build_html_structure(self, document: Document) -> str:
        """Build basic HTML structure."""
        html_parts = []
        
        for section in document.sections:
            section_html = self._section_to_html(section, level=2)
            html_parts.append(section_html)
        
        return "\n".join(html_parts)
    
    def _section_to_html(self, section: DocumentSection, level: int = 2) -> str:
        """Convert section to HTML."""
        html = f'<h{level} id="{section.section_id}">{html.escape(section.title)}</h{level}>\n'
        
        # Convert markdown content to HTML
        section_html = markdown.markdown(section.content)
        html += section_html
        
        # Add code examples
        for example in section.code_examples:
            html += f'<pre><code class="{example.get("language", "")}">{html.escape(example["code"])}</code></pre>\n'
        
        # Add images
        for image in section.images:
            html += f'<img src="{image["url"]}" alt="{image.get("alt", "")}" title="{image.get("title", "")}">\n'
        
        # Add subsections
        for subsection in section.subsections:
            html += self._section_to_html(subsection, level + 1)
        
        return html
    
    def _section_to_markdown(self, section: DocumentSection, level: int = 2) -> str:
        """Convert section to Markdown."""
        md = "#" * level + f" {section.title}\n\n"
        md += section.content + "\n\n"
        
        # Add code examples
        for example in section.code_examples:
            lang = example.get("language", "")
            md += f"```{lang}\n{example['code']}\n```\n\n"
        
        # Add subsections
        for subsection in section.subsections:
            md += self._section_to_markdown(subsection, level + 1)
        
        return md
    
    def _section_to_interactive(self, section: DocumentSection) -> Dict[str, Any]:
        """Convert section to interactive format."""
        return {
            "section_id": section.section_id,
            "title": section.title,
            "content": section.content,
            "difficulty": section.difficulty.value,
            "estimated_time": section.estimated_read_time_minutes,
            "interactive_elements": section.interactive_elements,
            "code_examples": section.code_examples,
            "images": section.images,
            "videos": section.videos,
            "subsections": [self._section_to_interactive(sub) for sub in section.subsections]
        }
    
    def _generate_toc(self, sections: List[DocumentSection]) -> str:
        """Generate table of contents."""
        toc = "## Table of Contents\n\n"
        
        for section in sections:
            toc += f"- [{section.title}](#{section.section_id})\n"
            for subsection in section.subsections:
                toc += f"  - [{subsection.title}](#{subsection.section_id})\n"
        
        return toc
    
    def _apply_default_template(self, document: Document, content: str) -> str:
        """Apply default HTML template."""
        template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{html.escape(document.title)} - Manufacturing Control System Documentation</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; margin: 40px auto; max-width: 1200px; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
                pre {{ background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .metadata {{ background: #e8f4fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .difficulty-{document.difficulty.value} {{ border-left: 4px solid #3498db; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <header>
                <h1>{html.escape(document.title)}</h1>
                <div class="metadata difficulty-{document.difficulty.value}">
                    <p><strong>Description:</strong> {html.escape(document.description)}</p>
                    <p><strong>Version:</strong> {document.version} | <strong>Author:</strong> {document.author}</p>
                    <p><strong>Estimated Time:</strong> {document.estimated_completion_time_minutes} minutes | <strong>Difficulty:</strong> {document.difficulty.value.title()}</p>
                </div>
            </header>
            <main>
                {content}
            </main>
            <footer>
                <hr>
                <p><em>Last modified: {document.last_modified.strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            </footer>
        </body>
        </html>
        """
        return template
    
    def _build_mobile_html(self, document: Document) -> str:
        """Build mobile-responsive HTML."""
        content = self._build_html_structure(document)
        
        mobile_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{html.escape(document.title)}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 16px; font-size: 16px; line-height: 1.6; }}
                h1 {{ font-size: 24px; margin-bottom: 16px; }}
                h2 {{ font-size: 20px; margin: 24px 0 12px 0; }}
                h3 {{ font-size: 18px; margin: 20px 0 8px 0; }}
                pre {{ font-size: 14px; overflow-x: auto; white-space: pre-wrap; }}
                img {{ max-width: 100%; height: auto; }}
                .collapsible {{ background: #f1f1f1; cursor: pointer; padding: 12px; border: none; width: 100%; text-align: left; }}
                .content {{ display: none; padding: 12px; }}
                @media (max-width: 480px) {{
                    body {{ padding: 12px; font-size: 14px; }}
                    h1 {{ font-size: 22px; }}
                    h2 {{ font-size: 18px; }}
                }}
            </style>
        </head>
        <body>
            <h1>{html.escape(document.title)}</h1>
            <div class="mobile-nav">
                <button class="collapsible">Table of Contents</button>
                <div class="content">
                    {self._generate_mobile_toc(document.sections)}
                </div>
            </div>
            {content}
            <script>
                var coll = document.getElementsByClassName("collapsible");
                for (var i = 0; i < coll.length; i++) {{
                    coll[i].addEventListener("click", function() {{
                        this.classList.toggle("active");
                        var content = this.nextElementSibling;
                        if (content.style.display === "block") {{
                            content.style.display = "none";
                        }} else {{
                            content.style.display = "block";
                        }}
                    }});
                }}
            </script>
        </body>
        </html>
        """
        return mobile_template
    
    def _generate_mobile_toc(self, sections: List[DocumentSection]) -> str:
        """Generate mobile-friendly table of contents."""
        toc = "<ul>"
        for section in sections:
            toc += f'<li><a href="#{section.section_id}">{html.escape(section.title)}</a></li>'
        toc += "</ul>"
        return toc


class DocumentationSearch:
    """Search engine for documentation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.search_index: Dict[str, List[str]] = {}  # keyword -> document_ids
        self.documents: Dict[str, Document] = {}
    
    def index_document(self, document: Document):
        """Add document to search index."""
        self.documents[document.document_id] = document
        
        # Extract searchable text
        searchable_text = self._extract_searchable_text(document)
        
        # Tokenize and index
        tokens = self._tokenize(searchable_text)
        
        for token in tokens:
            if token not in self.search_index:
                self.search_index[token] = []
            
            if document.document_id not in self.search_index[token]:
                self.search_index[token].append(document.document_id)
    
    def search(self, query: str, 
               role_filter: Optional[List[UserRole]] = None,
               document_type_filter: Optional[List[DocumentationType]] = None,
               limit: int = 20) -> List[SearchResult]:
        """Search documentation."""
        try:
            query_tokens = self._tokenize(query.lower())
            
            # Find matching documents
            document_scores = {}
            
            for token in query_tokens:
                if token in self.search_index:
                    for doc_id in self.search_index[token]:
                        if doc_id not in document_scores:
                            document_scores[doc_id] = 0
                        document_scores[doc_id] += 1
            
            # Apply filters and generate results
            results = []
            
            for doc_id, score in sorted(document_scores.items(), key=lambda x: x[1], reverse=True):
                document = self.documents[doc_id]
                
                # Apply role filter
                if role_filter and not any(role in document.target_roles for role in role_filter):
                    continue
                
                # Apply document type filter
                if document_type_filter and document.document_type not in document_type_filter:
                    continue
                
                # Generate snippet
                snippet = self._generate_snippet(document, query_tokens)
                
                result = SearchResult(
                    document_id=doc_id,
                    section_id=None,  # Could be enhanced to find specific sections
                    title=document.title,
                    snippet=snippet,
                    relevance_score=score / len(query_tokens),
                    document_type=document.document_type,
                    target_roles=document.target_roles,
                    url=f"/docs/{doc_id}"
                )
                
                results.append(result)
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []
    
    def _extract_searchable_text(self, document: Document) -> str:
        """Extract all searchable text from document."""
        text_parts = [
            document.title,
            document.description,
            " ".join(document.tags),
            " ".join(document.search_keywords)
        ]
        
        for section in document.sections:
            text_parts.append(self._extract_section_text(section))
        
        return " ".join(text_parts).lower()
    
    def _extract_section_text(self, section: DocumentSection) -> str:
        """Extract text from document section."""
        text_parts = [
            section.title,
            section.content,
            " ".join(section.tags),
            " ".join(section.keywords)
        ]
        
        for subsection in section.subsections:
            text_parts.append(self._extract_section_text(subsection))
        
        return " ".join(text_parts)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for search indexing."""
        # Simple tokenization - in production, would use more sophisticated NLP
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'
        }
        
        return [token for token in tokens if token not in stop_words and len(token) > 2]
    
    def _generate_snippet(self, document: Document, query_tokens: List[str]) -> str:
        """Generate search result snippet."""
        # Find best matching section
        best_section = None
        best_score = 0
        
        for section in document.sections:
            section_text = self._extract_section_text(section).lower()
            section_score = sum(1 for token in query_tokens if token in section_text)
            
            if section_score > best_score:
                best_score = section_score
                best_section = section
        
        if best_section:
            # Extract snippet from best matching section
            content = best_section.content
            words = content.split()
            
            # Find query terms in content
            for i, word in enumerate(words):
                if any(token in word.lower() for token in query_tokens):
                    # Extract surrounding context
                    start = max(0, i - 10)
                    end = min(len(words), i + 20)
                    snippet_words = words[start:end]
                    snippet = " ".join(snippet_words)
                    
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    
                    return snippet
        
        # Fallback to document description
        return document.description[:200] + ("..." if len(document.description) > 200 else "")


class UserDocumentationSystem:
    """
    Comprehensive User Documentation System
    
    Provides complete documentation management for the manufacturing control system:
    - Role-based user guides and tutorials
    - Interactive help system with context-sensitive assistance
    - Searchable knowledge base with advanced filtering
    - Multi-format document generation (HTML, PDF, mobile)
    - Content management with version control
    - Usage analytics and feedback collection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.documents: Dict[str, Document] = {}
        self.help_contexts: Dict[str, HelpContext] = {}
        self.generator = DocumentationGenerator()
        self.search_engine = DocumentationSearch()
        
        # Initialize manufacturing system documentation
        self._initialize_manufacturing_documentation()
    
    def _initialize_manufacturing_documentation(self):
        """Initialize comprehensive manufacturing system documentation."""
        
        # Production Operator User Guide
        operator_guide = self._create_production_operator_guide()
        self.add_document(operator_guide)
        
        # Production Manager User Guide
        manager_guide = self._create_production_manager_guide()
        self.add_document(manager_guide)
        
        # Maintenance Technician Guide
        maintenance_guide = self._create_maintenance_technician_guide()
        self.add_document(maintenance_guide)
        
        # Quality Controller Guide
        quality_guide = self._create_quality_controller_guide()
        self.add_document(quality_guide)
        
        # System Administrator Guide
        admin_guide = self._create_system_administrator_guide()
        self.add_document(admin_guide)
        
        # Quick Start Guide
        quick_start = self._create_quick_start_guide()
        self.add_document(quick_start)
        
        # FAQ Document
        faq = self._create_faq_document()
        self.add_document(faq)
        
        # Troubleshooting Guide
        troubleshooting = self._create_troubleshooting_guide()
        self.add_document(troubleshooting)
    
    def _create_production_operator_guide(self) -> Document:
        """Create comprehensive production operator user guide."""
        
        # Dashboard Navigation Section
        dashboard_section = DocumentSection(
            section_id="operator-dashboard-navigation",
            title="Dashboard Navigation and Overview",
            content="""
The Production Operator Dashboard is your central command center for monitoring and controlling manufacturing operations. 
This section covers how to navigate the interface effectively and understand the key information displays.

## Key Dashboard Elements

### Real-time Production Metrics
The top section displays current production statistics including:
- **Current Throughput**: Units produced per hour
- **Overall Equipment Effectiveness (OEE)**: Percentage indicating equipment efficiency
- **Quality Metrics**: Real-time defect rates and quality indicators
- **Equipment Status**: Visual indicators showing the operational state of all equipment

### Alert and Notification Center
The alert panel shows:
- Active alarms requiring immediate attention
- Equipment warnings and maintenance notifications
- Quality deviations that need operator response
- System messages and shift communications

### Control Panels
Interactive controls allow you to:
- Adjust production parameters within approved ranges
- Start and stop equipment following safety protocols
- Acknowledge alarms and notifications
- Document shift notes and observations
            """,
            difficulty=ContentDifficulty.BEGINNER,
            estimated_read_time_minutes=8,
            tags=["dashboard", "navigation", "operator", "interface"],
            keywords=["dashboard", "production", "metrics", "alerts", "controls"],
            images=[
                {"url": "/docs/images/operator-dashboard-overview.png", "alt": "Operator Dashboard Overview", "title": "Main operator dashboard showing key metrics and controls"}
            ]
        )
        
        # Equipment Control Section
        equipment_control_section = DocumentSection(
            section_id="operator-equipment-control",
            title="Equipment Operation and Control",
            content="""
This section covers the safe and effective operation of manufacturing equipment through the control system.

## Equipment Startup Procedures

### Pre-startup Checklist
Before starting any equipment, complete these safety checks:
1. Verify all safety barriers and guards are in place
2. Check that the work area is clear of personnel
3. Confirm raw materials are available and properly positioned
4. Review any maintenance notes or warnings from previous shifts

### Starting Equipment
To start equipment safely:
1. Select the equipment from the dashboard
2. Click the "Start Sequence" button
3. Monitor the startup progress indicator
4. Wait for the "Ready" status before beginning production
5. Document the startup time in the shift log

### Normal Operations
During production:
- Monitor real-time performance indicators
- Watch for any deviation alerts
- Maintain material supply levels
- Record hourly production counts
- Note any unusual observations

## Emergency Procedures

### Emergency Stop
If immediate shutdown is required:
1. Press the physical E-STOP button on the equipment
2. Or click the emergency stop icon on the dashboard
3. Notify your supervisor immediately
4. Do not restart without supervisor approval
5. Complete an incident report
            """,
            difficulty=ContentDifficulty.INTERMEDIATE,
            estimated_read_time_minutes=12,
            tags=["equipment", "control", "safety", "procedures"],
            keywords=["equipment", "startup", "emergency", "safety", "procedures"],
            interactive_elements=[
                {"type": "checklist", "title": "Pre-startup Safety Checklist", "items": ["Safety barriers", "Clear work area", "Materials ready", "Review maintenance notes"]}
            ]
        )
        
        # Quality Monitoring Section
        quality_section = DocumentSection(
            section_id="operator-quality-monitoring",
            title="Quality Monitoring and Response",
            content="""
Quality monitoring is a critical responsibility for production operators. This section explains how to monitor quality metrics and respond to quality issues.

## Quality Dashboards and Indicators

### Statistical Process Control (SPC) Charts
The system displays real-time SPC charts showing:
- Control limits (upper and lower)
- Process mean and variation
- Trending data over time
- Out-of-control alerts

### Quality Alerts
When quality issues are detected:
- Red alerts indicate immediate action required
- Yellow warnings suggest monitoring needed
- The system provides recommended actions
- All alerts are logged automatically

## Responding to Quality Issues

### When a Quality Alert Occurs:
1. **Acknowledge the alert** by clicking the notification
2. **Investigate the cause** by reviewing recent process changes
3. **Take corrective action** as recommended by the system
4. **Document your actions** in the quality log
5. **Monitor the results** to ensure the issue is resolved

### Sample Investigation Process:
- Check raw material quality
- Verify equipment calibration
- Review recent parameter changes
- Inspect finished products
- Consult with quality controller if needed
            """,
            difficulty=ContentDifficulty.INTERMEDIATE,
            estimated_read_time_minutes=10,
            tags=["quality", "monitoring", "spc", "alerts"],
            keywords=["quality", "control", "defects", "alerts", "investigation"]
        )
        
        return Document(
            document_id="production-operator-guide",
            title="Production Operator User Guide",
            description="Comprehensive guide for production operators covering dashboard navigation, equipment control, quality monitoring, and safety procedures.",
            document_type=DocumentationType.USER_GUIDE,
            target_roles=[UserRole.PRODUCTION_OPERATOR],
            sections=[dashboard_section, equipment_control_section, quality_section],
            author="Manufacturing Control System Team",
            version="2.1",
            category="User Guides",
            difficulty=ContentDifficulty.BEGINNER,
            estimated_completion_time_minutes=45,
            tags=["production", "operator", "user-guide", "dashboard", "equipment", "quality"],
            search_keywords=["operator", "production", "dashboard", "equipment", "control", "quality", "safety"]
        )
    
    def _create_production_manager_guide(self) -> Document:
        """Create production manager user guide."""
        
        kpi_section = DocumentSection(
            section_id="manager-kpi-monitoring",
            title="Key Performance Indicator (KPI) Monitoring",
            content="""
The Production Manager Dashboard provides comprehensive KPI monitoring and analysis tools to help you make data-driven decisions and optimize manufacturing performance.

## Primary Manufacturing KPIs

### Overall Equipment Effectiveness (OEE)
OEE is calculated as: **Availability × Performance × Quality**

- **Availability**: Percentage of scheduled time equipment is operational
- **Performance**: Actual production rate vs. ideal production rate  
- **Quality**: Percentage of good parts produced vs. total parts

**Target OEE**: >85% (World-class manufacturing standard)

### Throughput Analysis
Monitor production throughput across multiple dimensions:
- Units per hour by equipment line
- Shift-over-shift comparison
- Weekly and monthly trending
- Bottleneck identification

### Cost Performance
Track key cost metrics:
- Cost per unit produced
- Labor efficiency ratios
- Material utilization rates
- Energy consumption per unit

## Performance Analysis Tools

### Trend Analysis
The system provides automated trend analysis showing:
- Performance patterns over time
- Seasonal variations
- Improvement opportunities
- Predictive forecasting

### Benchmarking
Compare your performance against:
- Historical performance
- Industry benchmarks
- Other facilities (if applicable)
- Theoretical optimum performance
            """,
            difficulty=ContentDifficulty.INTERMEDIATE,
            estimated_read_time_minutes=15,
            tags=["kpi", "performance", "oee", "analysis"],
            keywords=["kpi", "performance", "oee", "throughput", "analysis", "benchmarking"]
        )
        
        return Document(
            document_id="production-manager-guide",
            title="Production Manager User Guide", 
            description="Comprehensive guide for production managers covering KPI monitoring, performance analysis, scheduling, and team management.",
            document_type=DocumentationType.USER_GUIDE,
            target_roles=[UserRole.PRODUCTION_MANAGER],
            sections=[kpi_section],
            author="Manufacturing Control System Team",
            version="2.1",
            category="User Guides",
            difficulty=ContentDifficulty.INTERMEDIATE,
            estimated_completion_time_minutes=60,
            tags=["management", "kpi", "performance", "scheduling"],
            search_keywords=["manager", "kpi", "performance", "scheduling", "analysis", "reporting"]
        )
    
    def _create_maintenance_technician_guide(self) -> Document:
        """Create maintenance technician guide."""
        
        predictive_section = DocumentSection(
            section_id="maintenance-predictive-maintenance",
            title="Predictive Maintenance System",
            content="""
The predictive maintenance system uses AI and sensor data to predict equipment failures before they occur, enabling proactive maintenance scheduling.

## Understanding Predictive Indicators

### Health Scores
Each piece of equipment has a health score (0-100):
- **90-100**: Excellent condition, no immediate action needed
- **70-89**: Good condition, monitor closely
- **50-69**: Fair condition, schedule preventive maintenance
- **30-49**: Poor condition, plan maintenance soon
- **0-29**: Critical condition, immediate maintenance required

### Failure Predictions
The system provides failure predictions with:
- Probability of failure within different time windows (24h, 7d, 30d)
- Most likely failure modes
- Recommended maintenance actions
- Historical failure patterns

## Maintenance Work Orders

### Automated Work Order Generation
The system automatically creates work orders when:
- Health scores drop below thresholds
- Failure probability exceeds limits
- Scheduled maintenance intervals are reached
- Operators report equipment issues

### Work Order Prioritization
Work orders are automatically prioritized based on:
- Equipment criticality
- Production impact
- Safety considerations
- Resource availability
            """,
            difficulty=ContentDifficulty.ADVANCED,
            estimated_read_time_minutes=12,
            tags=["maintenance", "predictive", "health-score", "work-orders"],
            keywords=["maintenance", "predictive", "health", "failure", "work-orders", "scheduling"]
        )
        
        return Document(
            document_id="maintenance-technician-guide",
            title="Maintenance Technician User Guide",
            description="Guide for maintenance technicians covering predictive maintenance, work order management, and equipment diagnostics.",
            document_type=DocumentationType.USER_GUIDE,
            target_roles=[UserRole.MAINTENANCE_TECHNICIAN],
            sections=[predictive_section],
            author="Manufacturing Control System Team",
            version="2.1",
            category="User Guides",
            difficulty=ContentDifficulty.ADVANCED,
            estimated_completion_time_minutes=45,
            tags=["maintenance", "predictive", "diagnostics", "work-orders"],
            search_keywords=["maintenance", "predictive", "equipment", "diagnostics", "repair", "scheduling"]
        )
    
    def _create_quality_controller_guide(self) -> Document:
        """Create quality controller guide."""
        
        spc_section = DocumentSection(
            section_id="quality-spc-monitoring",
            title="Statistical Process Control (SPC) Monitoring",
            content="""
Statistical Process Control is fundamental to maintaining consistent product quality. This section explains how to use the SPC monitoring tools effectively.

## SPC Chart Types and Interpretation

### X-bar and R Charts
Used for variable data (measurements):
- **X-bar Chart**: Monitors process average
- **R Chart**: Monitors process variation
- Both charts must be in control for the process to be stable

### p-Charts
Used for attribute data (defect rates):
- Monitors the proportion of defective items
- Useful for tracking overall quality performance
- Automatically calculates control limits

### Control Limit Violations
The system automatically detects:
- Points beyond control limits
- Runs of 7 or more points on one side of centerline
- Trends of 7 or more consecutive increasing/decreasing points
- Patterns suggesting non-random variation

## Quality Investigation Procedures

### When an Out-of-Control Condition Occurs:
1. **Stop Production** (if required by procedures)
2. **Investigate the Assignable Cause**:
   - Review recent process changes
   - Check equipment calibration
   - Inspect raw materials
   - Interview operators about any unusual events
3. **Implement Corrective Action**
4. **Verify the Correction** with additional samples
5. **Resume Production** once control is restored
6. **Document All Actions** in the quality system
            """,
            difficulty=ContentDifficulty.ADVANCED,
            estimated_read_time_minutes=15,
            tags=["quality", "spc", "control-charts", "investigation"],
            keywords=["quality", "spc", "control", "charts", "limits", "investigation", "corrective-action"]
        )
        
        return Document(
            document_id="quality-controller-guide",
            title="Quality Controller User Guide",
            description="Comprehensive guide for quality controllers covering SPC monitoring, quality investigations, and compliance reporting.",
            document_type=DocumentationType.USER_GUIDE,
            target_roles=[UserRole.QUALITY_CONTROLLER],
            sections=[spc_section],
            author="Manufacturing Control System Team",
            version="2.1",
            category="User Guides",
            difficulty=ContentDifficulty.ADVANCED,
            estimated_completion_time_minutes=50,
            tags=["quality", "spc", "compliance", "investigation"],
            search_keywords=["quality", "spc", "control", "compliance", "investigation", "reporting"]
        )
    
    def _create_system_administrator_guide(self) -> Document:
        """Create system administrator guide."""
        
        system_section = DocumentSection(
            section_id="admin-system-monitoring",
            title="System Health Monitoring and Management",
            content="""
System administrators are responsible for maintaining the overall health and performance of the manufacturing control system.

## System Health Dashboard

### Key Health Indicators
Monitor these critical system metrics:
- **CPU Utilization**: Should remain below 80% under normal load
- **Memory Usage**: Should not exceed 85% of available memory
- **Disk Space**: Maintain at least 20% free space on all drives
- **Network Performance**: Monitor for packet loss and latency issues
- **Database Performance**: Query response times and connection pool usage

### Performance Monitoring
The system provides real-time monitoring of:
- Application response times
- Database query performance
- Network throughput and latency
- User session statistics
- Error rates and exceptions

## User Management

### Adding New Users
To add a new user account:
1. Navigate to User Management → Add User
2. Enter user details (name, email, department)
3. Assign appropriate role(s)
4. Set initial password (user will be prompted to change)
5. Configure access permissions
6. Send welcome email with login instructions

### Role Management
The system supports these standard roles:
- Production Operator
- Production Manager
- Maintenance Technician
- Quality Controller
- System Administrator

Custom roles can be created with specific permission sets.
            """,
            difficulty=ContentDifficulty.EXPERT,
            estimated_read_time_minutes=20,
            tags=["administration", "system-health", "user-management", "monitoring"],
            keywords=["admin", "system", "monitoring", "users", "performance", "health", "management"]
        )
        
        return Document(
            document_id="system-administrator-guide",
            title="System Administrator User Guide",
            description="Technical guide for system administrators covering system monitoring, user management, and maintenance procedures.",
            document_type=DocumentationType.USER_GUIDE,
            target_roles=[UserRole.SYSTEM_ADMINISTRATOR],
            sections=[system_section],
            author="Manufacturing Control System Team",
            version="2.1",
            category="User Guides",
            difficulty=ContentDifficulty.EXPERT,
            estimated_completion_time_minutes=90,
            tags=["administration", "system", "monitoring", "management"],
            search_keywords=["admin", "system", "monitoring", "users", "maintenance", "troubleshooting"]
        )
    
    def _create_quick_start_guide(self) -> Document:
        """Create quick start guide for new users."""
        
        quick_start_section = DocumentSection(
            section_id="quick-start-overview",
            title="Getting Started in 5 Minutes",
            content="""
Welcome to the Manufacturing Control System! This quick start guide will have you up and running in just 5 minutes.

## Step 1: Logging In (30 seconds)
1. Open your web browser and navigate to the system URL
2. Enter your username and password
3. Click "Sign In"
4. You'll be redirected to your role-specific dashboard

## Step 2: Dashboard Overview (2 minutes)
Your dashboard is customized for your role and shows the most important information for your daily tasks:

- **Top Section**: Key performance indicators and current status
- **Middle Section**: Interactive charts and real-time data
- **Bottom Section**: Recent alerts, notifications, and quick actions
- **Navigation Menu**: Access to all system functions (left sidebar)

## Step 3: Basic Navigation (1 minute)
- **Home Icon**: Returns you to the main dashboard
- **Menu Items**: Click to expand sections and access detailed views
- **Search Box**: Find specific information, documents, or help topics
- **Profile Menu**: Access account settings and logout (top right)

## Step 4: Getting Help (1 minute)
- **Help Icon**: Click the "?" icon anywhere for context-sensitive help
- **Search Help**: Type your question in the search box
- **Support**: Use the "Contact Support" link for assistance

## Step 5: Your First Task (30 seconds)
Try these role-specific first tasks:
- **Operators**: Check current production status
- **Managers**: Review today's KPI summary
- **Maintenance**: Check equipment health scores
- **Quality**: Review active SPC charts
            """,
            difficulty=ContentDifficulty.BEGINNER,
            estimated_read_time_minutes=5,
            tags=["quick-start", "getting-started", "beginner", "login"],
            keywords=["quick", "start", "login", "dashboard", "navigation", "help", "first", "time"]
        )
        
        return Document(
            document_id="quick-start-guide",
            title="Quick Start Guide",
            description="Get up and running with the Manufacturing Control System in just 5 minutes.",
            document_type=DocumentationType.QUICK_START,
            target_roles=[UserRole.PRODUCTION_OPERATOR, UserRole.PRODUCTION_MANAGER, 
                         UserRole.MAINTENANCE_TECHNICIAN, UserRole.QUALITY_CONTROLLER],
            sections=[quick_start_section],
            author="Manufacturing Control System Team",
            version="2.1",
            category="Getting Started",
            difficulty=ContentDifficulty.BEGINNER,
            estimated_completion_time_minutes=5,
            tags=["quick-start", "beginner", "getting-started", "overview"],
            search_keywords=["quick", "start", "begin", "first", "time", "login", "dashboard", "overview"]
        )
    
    def _create_faq_document(self) -> Document:
        """Create frequently asked questions document."""
        
        faq_section = DocumentSection(
            section_id="manufacturing-faq",
            title="Frequently Asked Questions",
            content="""
## General Questions

**Q: How do I reset my password?**
A: Click "Forgot Password" on the login screen, enter your email address, and follow the instructions in the password reset email.

**Q: Why can't I see certain menu items or features?**
A: Menu items and features are role-based. If you need access to additional features, contact your supervisor or system administrator.

**Q: How often is the data updated?**
A: Real-time data (production metrics, equipment status) updates every 15 seconds. Historical reports update hourly.

## Dashboard and Interface

**Q: Can I customize my dashboard?**
A: Yes, you can rearrange dashboard panels and choose which metrics to display. Click the "Customize" button in the top right of your dashboard.

**Q: The charts aren't loading. What should I do?**
A: Try refreshing your browser (Ctrl+F5). If the problem persists, check your internet connection or contact IT support.

**Q: How do I export data or reports?**
A: Most charts and tables have an "Export" button that allows you to download data in Excel or PDF format.

## Equipment and Production

**Q: What does the yellow status indicator mean?**
A: Yellow indicates a warning condition that requires attention but doesn't require immediate shutdown. Check the alert details for specific actions needed.

**Q: How accurate are the failure predictions?**
A: The AI system achieves >85% accuracy for predictions within 72 hours. Accuracy improves as the predicted time approaches.

**Q: Can I override an automated decision?**
A: Yes, authorized users can override most automated decisions, but all overrides are logged and may require justification.
            """,
            difficulty=ContentDifficulty.BEGINNER,
            estimated_read_time_minutes=8,
            tags=["faq", "questions", "troubleshooting", "help"],
            keywords=["faq", "questions", "answers", "help", "troubleshooting", "common", "issues"]
        )
        
        return Document(
            document_id="manufacturing-faq",
            title="Frequently Asked Questions (FAQ)",
            description="Answers to common questions about using the Manufacturing Control System.",
            document_type=DocumentationType.FAQ,
            target_roles=[UserRole.PRODUCTION_OPERATOR, UserRole.PRODUCTION_MANAGER,
                         UserRole.MAINTENANCE_TECHNICIAN, UserRole.QUALITY_CONTROLLER,
                         UserRole.SYSTEM_ADMINISTRATOR],
            sections=[faq_section],
            author="Manufacturing Control System Team",
            version="2.1",
            category="Help and Support",
            difficulty=ContentDifficulty.BEGINNER,
            estimated_completion_time_minutes=10,
            tags=["faq", "help", "support", "questions", "answers"],
            search_keywords=["faq", "questions", "help", "support", "common", "issues", "troubleshooting"]
        )
    
    def _create_troubleshooting_guide(self) -> Document:
        """Create troubleshooting guide."""
        
        troubleshooting_section = DocumentSection(
            section_id="system-troubleshooting",
            title="System Troubleshooting Guide",
            content="""
This guide helps you diagnose and resolve common system issues.

## Connection and Login Issues

### "Cannot Connect to Server" Error
**Symptoms**: Unable to load the application, connection timeout errors
**Possible Causes**:
- Network connectivity issues
- Server maintenance in progress
- Firewall blocking access

**Troubleshooting Steps**:
1. Check your internet connection
2. Try accessing from a different device
3. Contact IT support if the issue persists
4. Check for maintenance notifications

### Login Failures
**Symptoms**: "Invalid username or password" errors, account lockout
**Troubleshooting Steps**:
1. Verify your username and password are correct
2. Check if Caps Lock is enabled
3. Try clearing your browser cache and cookies
4. If locked out, wait 15 minutes before retrying
5. Use "Forgot Password" if needed

## Dashboard and Data Issues

### Dashboard Not Loading or Showing Old Data
**Symptoms**: Blank dashboard, outdated information, missing charts
**Troubleshooting Steps**:
1. Refresh your browser (Ctrl+F5 or Cmd+Shift+R)
2. Clear browser cache and cookies
3. Check if you're using a supported browser
4. Disable browser extensions temporarily
5. Try incognito/private browsing mode

### Charts or Reports Not Displaying
**Symptoms**: Missing charts, "No Data Available" messages
**Troubleshooting Steps**:
1. Check your date range selection
2. Verify you have permissions for the requested data
3. Try a different time period
4. Check if the data source is available
5. Contact support if data should be available

## Performance Issues

### Slow System Response
**Symptoms**: Pages load slowly, delays in data updates
**Troubleshooting Steps**:
1. Check your internet connection speed
2. Close unnecessary browser tabs
3. Clear browser cache
4. Restart your browser
5. Check system status page for known issues

### Timeout Errors
**Symptoms**: "Request timeout" or "Session expired" errors
**Troubleshooting Steps**:
1. Try the action again
2. Break large requests into smaller ones
3. Log out and log back in
4. Contact support for persistent issues
            """,
            difficulty=ContentDifficulty.INTERMEDIATE,
            estimated_read_time_minutes=12,
            tags=["troubleshooting", "problems", "solutions", "technical"],
            keywords=["troubleshooting", "problems", "issues", "solutions", "errors", "fix", "resolve"]
        )
        
        return Document(
            document_id="troubleshooting-guide",
            title="Troubleshooting Guide",
            description="Step-by-step guide to diagnosing and resolving common system issues.",
            document_type=DocumentationType.TROUBLESHOOTING,
            target_roles=[UserRole.PRODUCTION_OPERATOR, UserRole.PRODUCTION_MANAGER,
                         UserRole.MAINTENANCE_TECHNICIAN, UserRole.QUALITY_CONTROLLER,
                         UserRole.SYSTEM_ADMINISTRATOR],
            sections=[troubleshooting_section],
            author="Manufacturing Control System Team",
            version="2.1",
            category="Help and Support",
            difficulty=ContentDifficulty.INTERMEDIATE,
            estimated_completion_time_minutes=15,
            tags=["troubleshooting", "problems", "technical-support", "solutions"],
            search_keywords=["troubleshooting", "problems", "issues", "errors", "fix", "solve", "support"]
        )
    
    def add_document(self, document: Document) -> str:
        """Add document to the system."""
        self.documents[document.document_id] = document
        self.search_engine.index_document(document)
        
        self.logger.info(f"Document added: {document.title}")
        return document.document_id
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.documents.get(document_id)
    
    def search_documents(self, query: str, 
                        user_role: Optional[UserRole] = None,
                        document_type: Optional[DocumentationType] = None) -> List[SearchResult]:
        """Search documents."""
        role_filter = [user_role] if user_role else None
        type_filter = [document_type] if document_type else None
        
        return self.search_engine.search(query, role_filter, type_filter)
    
    def generate_document_html(self, document_id: str) -> str:
        """Generate HTML version of document."""
        document = self.get_document(document_id)
        if not document:
            return ""
        
        return self.generator.generate_html(document)
    
    def generate_document_pdf(self, document_id: str) -> bytes:
        """Generate PDF version of document."""
        # PDF generation would require additional libraries like WeasyPrint
        # For now, return placeholder
        document = self.get_document(document_id)
        if not document:
            return b""
        
        self.logger.info(f"PDF generation requested for: {document.title}")
        return b"PDF content would be generated here"
    
    def get_role_specific_documents(self, user_role: UserRole) -> List[Document]:
        """Get documents relevant to specific user role."""
        relevant_docs = []
        
        for document in self.documents.values():
            if user_role in document.target_roles or not document.target_roles:
                relevant_docs.append(document)
        
        return sorted(relevant_docs, key=lambda d: d.title)
    
    def get_document_analytics(self, document_id: str) -> Dict[str, Any]:
        """Get document usage analytics."""
        # In a real system, this would track actual usage metrics
        return {
            "document_id": document_id,
            "total_views": 1247,
            "unique_users": 89,
            "average_read_time_minutes": 8.3,
            "completion_rate": 0.72,
            "user_rating": 4.2,
            "feedback_count": 23,
            "most_accessed_sections": [
                "Dashboard Navigation",
                "Equipment Control",
                "Quality Monitoring"
            ],
            "search_queries_leading_to_doc": [
                "dashboard navigation",
                "equipment startup",
                "quality alerts"
            ]
        }
    
    def add_help_context(self, context: HelpContext):
        """Add context-sensitive help."""
        self.help_contexts[context.context_id] = context
        self.logger.info(f"Help context added: {context.context_id}")
    
    def get_help_for_context(self, page_url: str, element_selector: Optional[str] = None) -> List[HelpContext]:
        """Get contextual help for specific page/element."""
        relevant_help = []
        
        for context in self.help_contexts.values():
            if context.page_url == page_url:
                if element_selector and context.element_selector == element_selector:
                    relevant_help.append(context)
                elif not element_selector and not context.element_selector:
                    relevant_help.append(context)
        
        return relevant_help
    
    def export_documentation_package(self, format: DocumentFormat = DocumentFormat.HTML) -> Dict[str, Any]:
        """Export complete documentation package."""
        package = {
            "export_metadata": {
                "export_date": datetime.now().isoformat(),
                "format": format.value,
                "total_documents": len(self.documents),
                "version": "2.1"
            },
            "documents": {}
        }
        
        for doc_id, document in self.documents.items():
            if format == DocumentFormat.HTML:
                content = self.generator.generate_html(document)
            elif format == DocumentFormat.MARKDOWN:
                content = self.generator.generate_markdown(document)
            elif format == DocumentFormat.MOBILE_RESPONSIVE:
                content = self.generator.generate_mobile_responsive(document)
            else:
                content = self.generator.generate_interactive(document)
            
            package["documents"][doc_id] = {
                "title": document.title,
                "description": document.description,
                "target_roles": [role.value for role in document.target_roles],
                "content": content
            }
        
        return package


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("User Documentation System Demo")
    print("=" * 80)
    
    # Initialize documentation system
    doc_system = UserDocumentationSystem()
    
    print(f"Documentation system initialized with {len(doc_system.documents)} documents")
    
    print(f"\nAvailable Documents:")
    for doc_id, document in doc_system.documents.items():
        roles = ", ".join([role.value for role in document.target_roles])
        print(f"  • {document.title} ({document.document_type.value}) - Roles: {roles}")
    
    print(f"\n" + "="*80)
    print("SEARCH DEMONSTRATION")
    print("="*80)
    
    # Test search functionality
    search_queries = [
        "dashboard navigation",
        "quality monitoring", 
        "equipment control",
        "troubleshooting login"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = doc_system.search_documents(query)
        
        for result in results[:3]:  # Show top 3 results
            print(f"  → {result.title} (Score: {result.relevance_score:.2f})")
            print(f"    {result.snippet[:100]}...")
    
    print(f"\n" + "="*80)
    print("ROLE-SPECIFIC DOCUMENTATION")
    print("="*80)
    
    # Show role-specific documents
    for role in [UserRole.PRODUCTION_OPERATOR, UserRole.PRODUCTION_MANAGER]:
        docs = doc_system.get_role_specific_documents(role)
        print(f"\n{role.value.replace('_', ' ').title()} Documents ({len(docs)}):")
        for doc in docs:
            print(f"  • {doc.title} ({doc.estimated_completion_time_minutes} min)")
    
    print(f"\n" + "="*80)
    print("DOCUMENT GENERATION")
    print("="*80)
    
    # Generate sample document in different formats
    sample_doc_id = "quick-start-guide"
    sample_doc = doc_system.get_document(sample_doc_id)
    
    if sample_doc:
        print(f"Generating '{sample_doc.title}' in different formats:")
        
        # Generate HTML
        html_content = doc_system.generate_document_html(sample_doc_id)
        print(f"  HTML: {len(html_content)} characters generated")
        
        # Generate Markdown
        markdown_content = doc_system.generator.generate_markdown(sample_doc)
        print(f"  Markdown: {len(markdown_content)} characters generated")
        
        # Show sample HTML content
        print(f"\nSample HTML Content (first 300 characters):")
        print(html_content[:300] + "...")
    
    print(f"\n" + "="*80)
    print("ANALYTICS DEMONSTRATION")
    print("="*80)
    
    # Show document analytics
    analytics = doc_system.get_document_analytics(sample_doc_id)
    print(f"Analytics for '{sample_doc.title}':")
    print(f"  Total Views: {analytics['total_views']}")
    print(f"  Unique Users: {analytics['unique_users']}")
    print(f"  Avg Read Time: {analytics['average_read_time_minutes']} minutes")
    print(f"  User Rating: {analytics['user_rating']}/5.0")
    print(f"  Completion Rate: {analytics['completion_rate']*100:.1f}%")
    
    print("\nUser Documentation System demo completed!")