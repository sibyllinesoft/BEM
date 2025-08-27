#!/usr/bin/env python3
"""
BEM Paper Factory - Anonymization Linter
Ensures double-blind compliance by detecting and flagging identifying information.

Scans for:
- Author names, affiliations, institutions
- File paths with usernames or org names  
- URLs to personal/institutional repositories
- Email addresses and contact information
- Acknowledgments and funding sources
- Dataset attributions that could identify authors
"""

import argparse
import logging
import re
import os
import json
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnonymizationViolation:
    file_path: str
    line_number: int
    line_content: str
    violation_type: str
    matched_text: str
    severity: str  # 'critical', 'warning', 'info'
    suggestion: str = ""

class AnonymizationLinter:
    """
    Scans codebase and paper files for potential anonymization violations.
    """
    
    def __init__(self):
        # Define patterns for different types of identifying information
        self.patterns = {
            'email_addresses': [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                re.compile(r'\\texttt\{[^}]*@[^}]*\}'),  # LaTeX email format
            ],
            'personal_names': [
                # Common academic name patterns
                re.compile(r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b'),  # John D. Smith
                re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),          # John Smith
                # Author field patterns
                re.compile(r'\\author\{[^}]+\}'),
                re.compile(r'author.*?:.*?[A-Z][a-z]+ [A-Z][a-z]+'),
            ],
            'institutions': [
                re.compile(r'\b(?:University|Institute|College|Laboratory|Lab)\s+(?:of\s+)?[A-Z][a-z]+', re.IGNORECASE),
                re.compile(r'\b(?:MIT|Stanford|Harvard|CMU|Berkeley|NYU|UCLA|USC|Caltech|Princeton)\b'),
                re.compile(r'\b(?:Google|Microsoft|Facebook|Meta|OpenAI|Anthropic|DeepMind)\b'),
                re.compile(r'\\textit\{[^}]*(?:University|Institute|College)[^}]*\}'),
            ],
            'file_paths': [
                re.compile(r'/(?:home|Users)/[a-zA-Z0-9_-]+'),  # Unix/Mac home paths
                re.compile(r'[C-Z]:\\Users\\[a-zA-Z0-9_-]+'),   # Windows user paths
                re.compile(r'[a-zA-Z0-9_-]+@[a-zA-Z0-9.-]+:'),  # SSH paths
            ],
            'urls_repositories': [
                re.compile(r'https?://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+'),
                re.compile(r'https?://gitlab\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+'),
                re.compile(r'https?://bitbucket\.org/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+'),
                re.compile(r'git@github\.com:[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+'),
            ],
            'acknowledgments': [
                re.compile(r'\\section\{Acknowledgments?\}', re.IGNORECASE),
                re.compile(r'thanks? to .{1,100}', re.IGNORECASE),
                re.compile(r'supported by .{1,100}', re.IGNORECASE),
                re.compile(r'funded by .{1,100}', re.IGNORECASE),
            ],
            'contact_info': [
                re.compile(r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}'),  # Phone numbers
                re.compile(r'\b\d{1,5}\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive)\b'),  # Addresses
            ],
            'funding_sources': [
                re.compile(r'NSF[\s-]?\w*\d+'),  # NSF grant numbers
                re.compile(r'NIH[\s-]?\w*\d+'),  # NIH grant numbers
                re.compile(r'DARPA[\s-]?\w*\d+'),  # DARPA grant numbers
                re.compile(r'Grant\s+(?:No\.?\s*)?[A-Z0-9-]+', re.IGNORECASE),
            ]
        }
        
        # File extensions to scan
        self.text_extensions = {'.tex', '.txt', '.md', '.py', '.yaml', '.yml', '.json', '.sh', '.bib'}
        
        # Whitelist patterns (acceptable mentions)
        self.whitelist_patterns = [
            re.compile(r'anonymous', re.IGNORECASE),
            re.compile(r'placeholder', re.IGNORECASE),
            re.compile(r'example\.com'),
            re.compile(r'university.*example', re.IGNORECASE),
        ]
        
    def scan_file(self, file_path: Path) -> List[AnonymizationViolation]:
        """Scan a single file for anonymization violations."""
        violations = []
        
        if file_path.suffix not in self.text_extensions:
            return violations
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return violations
        
        for line_num, line in enumerate(lines, 1):
            line_clean = line.strip()
            
            # Skip empty lines and comments
            if not line_clean or line_clean.startswith('%') or line_clean.startswith('#'):
                continue
                
            # Check each pattern category
            for category, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = pattern.finditer(line)
                    for match in matches:
                        matched_text = match.group(0)
                        
                        # Check if this is whitelisted
                        if self._is_whitelisted(matched_text):
                            continue
                            
                        severity = self._determine_severity(category, matched_text, file_path)
                        suggestion = self._generate_suggestion(category, matched_text)
                        
                        violation = AnonymizationViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            line_content=line_clean,
                            violation_type=category,
                            matched_text=matched_text,
                            severity=severity,
                            suggestion=suggestion
                        )
                        violations.append(violation)
        
        return violations
    
    def _is_whitelisted(self, text: str) -> bool:
        """Check if text matches whitelist patterns."""
        for pattern in self.whitelist_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _determine_severity(self, category: str, matched_text: str, file_path: Path) -> str:
        """Determine violation severity based on category and context."""
        # Critical violations that would definitely break anonymity
        critical_categories = {'email_addresses', 'personal_names', 'acknowledgments', 'funding_sources'}
        
        if category in critical_categories:
            return 'critical'
        
        # File paths in code might be acceptable, but in paper text are problematic
        if category == 'file_paths':
            if file_path.suffix in {'.tex', '.md'}:
                return 'critical'
            else:
                return 'warning'
        
        # URLs to personal repositories are critical
        if category == 'urls_repositories':
            return 'critical'
        
        # Institutions mentioned in passing might be warnings
        if category == 'institutions':
            # Check if it's just a dataset or method name
            academic_institutions = ['MIT', 'Stanford', 'Harvard', 'CMU', 'Berkeley']
            if any(inst in matched_text for inst in academic_institutions):
                return 'warning'  # Could be dataset name
            return 'critical'
        
        return 'warning'
    
    def _generate_suggestion(self, category: str, matched_text: str) -> str:
        """Generate helpful suggestion for fixing violation."""
        suggestions = {
            'email_addresses': "Replace with 'anonymous@email.com' or remove entirely",
            'personal_names': "Replace with 'Anonymous Authors' or remove",
            'institutions': "Replace with 'Anonymous Institution' or remove",
            'file_paths': "Use relative paths or replace with '~/path/to/file'",
            'urls_repositories': "Replace with anonymous repository URL or remove",
            'acknowledgments': "Remove entire acknowledgments section for blind review",
            'contact_info': "Remove all contact information",
            'funding_sources': "Remove funding information for blind review"
        }
        return suggestions.get(category, "Remove or anonymize this content")
    
    def scan_directory(self, directory: Path, exclude_dirs: Set[str] = None) -> List[AnonymizationViolation]:
        """Scan entire directory tree for violations."""
        if exclude_dirs is None:
            exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache'}
        
        all_violations = []
        
        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            root_path = Path(root)
            
            for file in files:
                file_path = root_path / file
                violations = self.scan_file(file_path)
                all_violations.extend(violations)
        
        return all_violations
    
    def generate_report(self, violations: List[AnonymizationViolation], output_path: str) -> None:
        """Generate comprehensive anonymization report."""
        # Group violations by severity and category
        by_severity = {'critical': [], 'warning': [], 'info': []}
        by_category = {}
        by_file = {}
        
        for violation in violations:
            by_severity[violation.severity].append(violation)
            
            if violation.violation_type not in by_category:
                by_category[violation.violation_type] = []
            by_category[violation.violation_type].append(violation)
            
            if violation.file_path not in by_file:
                by_file[violation.file_path] = []
            by_file[violation.file_path].append(violation)
        
        report = {
            'summary': {
                'total_violations': len(violations),
                'critical_violations': len(by_severity['critical']),
                'warning_violations': len(by_severity['warning']),
                'info_violations': len(by_severity['info']),
                'files_with_violations': len(by_file),
                'ready_for_submission': len(by_severity['critical']) == 0
            },
            'violations_by_severity': {
                severity: [
                    {
                        'file': v.file_path,
                        'line': v.line_number,
                        'type': v.violation_type,
                        'matched_text': v.matched_text,
                        'line_content': v.line_content,
                        'suggestion': v.suggestion
                    }
                    for v in violations_list
                ]
                for severity, violations_list in by_severity.items()
            },
            'violations_by_category': {
                category: len(violations_list)
                for category, violations_list in by_category.items()
            },
            'violations_by_file': {
                file_path: len(violations_list)
                for file_path, violations_list in by_file.items()
            }
        }
        
        # Write JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also write human-readable summary
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("BEM Paper Factory - Anonymization Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total violations: {report['summary']['total_violations']}\n")
            f.write(f"Critical violations: {report['summary']['critical_violations']}\n")
            f.write(f"Warning violations: {report['summary']['warning_violations']}\n")
            f.write(f"Files affected: {report['summary']['files_with_violations']}\n\n")
            
            if report['summary']['ready_for_submission']:
                f.write("✅ READY FOR SUBMISSION - No critical violations found\n\n")
            else:
                f.write("❌ NOT READY FOR SUBMISSION - Critical violations must be fixed\n\n")
            
            # List critical violations first
            if by_severity['critical']:
                f.write("CRITICAL VIOLATIONS (must fix):\n")
                f.write("-" * 35 + "\n")
                for violation in by_severity['critical'][:10]:  # Show first 10
                    f.write(f"File: {violation.file_path}:{violation.line_number}\n")
                    f.write(f"Type: {violation.violation_type}\n")
                    f.write(f"Found: {violation.matched_text}\n")
                    f.write(f"Suggestion: {violation.suggestion}\n")
                    f.write(f"Context: {violation.line_content[:80]}...\n\n")
                
                if len(by_severity['critical']) > 10:
                    f.write(f"... and {len(by_severity['critical']) - 10} more critical violations\n\n")
        
        logger.info(f"Anonymization report written to {output_path}")
        logger.info(f"Summary: {len(by_severity['critical'])} critical, {len(by_severity['warning'])} warning violations")

def main():
    parser = argparse.ArgumentParser(description='BEM Paper Factory - Anonymization Linter')
    parser.add_argument('--paths', nargs='+', required=True,
                       help='Paths to scan (files or directories)')
    parser.add_argument('--output', default='anonymization_report.json',
                       help='Output report file')
    parser.add_argument('--exclude-dirs', nargs='*', 
                       default=['.git', '__pycache__', '.venv', 'venv', 'node_modules'],
                       help='Directories to exclude from scanning')
    parser.add_argument('--fail-on-critical', action='store_true',
                       help='Exit with error code if critical violations found')
    
    args = parser.parse_args()
    
    # Initialize linter
    linter = AnonymizationLinter()
    
    # Scan all specified paths
    all_violations = []
    
    for path_str in args.paths:
        path = Path(path_str)
        
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            continue
            
        if path.is_file():
            violations = linter.scan_file(path)
        else:
            violations = linter.scan_directory(path, set(args.exclude_dirs))
            
        all_violations.extend(violations)
    
    # Generate report
    linter.generate_report(all_violations, args.output)
    
    # Check if ready for submission
    critical_count = sum(1 for v in all_violations if v.severity == 'critical')
    
    if critical_count > 0:
        logger.error(f"Found {critical_count} critical anonymization violations!")
        logger.error("Paper is NOT ready for blind review submission.")
        
        if args.fail_on_critical:
            exit(1)
    else:
        logger.info("✅ No critical anonymization violations found.")
        logger.info("Paper appears ready for blind review submission.")
        
        warning_count = sum(1 for v in all_violations if v.severity == 'warning')
        if warning_count > 0:
            logger.info(f"Note: {warning_count} warning-level issues found. Review recommended.")

if __name__ == '__main__':
    main()