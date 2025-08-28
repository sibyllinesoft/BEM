#!/usr/bin/env python3
"""
Paper Generation Tests for BEM Pipeline

Tests for automated research paper generation including LaTeX templating,
figure generation, claim formatting, and versioned paper management.

Test Categories:
    - LaTeX template rendering
    - Figure generation and formatting
    - Claim formatting and validation
    - Paper compilation and output
    - Versioned paper management
    - Error handling and edge cases

Usage:
    python -m pytest tests/test_paper_generation.py -v
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from datetime import datetime
import numpy as np

# Import components to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

from paper_generator import (
    PaperGenerator,
    LaTeXTemplateManager, 
    ClaimFormatter,
    FigureGenerator,
    VersionedPaperManager,
    PromotedClaim
)


class TestLaTeXTemplateManager(unittest.TestCase):
    """Test LaTeX template management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.template_manager = LaTeXTemplateManager(self.temp_dir / "templates")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_template_creation(self):
        """Test default template creation."""
        template_dir = self.temp_dir / "templates"
        
        # Check that default templates were created
        expected_templates = [
            'main.tex', 'abstract.tex', 'introduction.tex',
            'methodology.tex', 'results.tex', 'conclusion.tex'
        ]
        
        for template in expected_templates:
            template_path = template_dir / template
            self.assertTrue(template_path.exists(), f"Template {template} was not created")
            
            # Check that template has content
            with open(template_path, 'r') as f:
                content = f.read()
            self.assertGreater(len(content), 0, f"Template {template} is empty")
    
    def test_template_rendering(self):
        """Test template rendering with variables."""
        # Test variables
        test_vars = {
            'title': 'Test Paper Title',
            'authors': ['John Doe', 'Jane Smith'],
            'generation_date': '2024-01-01',
            'promoted_claims': [],
            'total_claims_tested': 5,
            'claims_promoted': 2,
            'claims_demoted': 3
        }
        
        # Render main template
        rendered = self.template_manager.render_template('main.tex', **test_vars)
        
        # Check that variables were substituted
        self.assertIn('Test Paper Title', rendered)
        self.assertIn('John Doe', rendered)
        self.assertIn('Jane Smith', rendered)
        self.assertIn('2024-01-01', rendered)
    
    def test_template_rendering_error_handling(self):
        """Test template rendering error handling."""
        # Try to render non-existent template
        with self.assertRaises(Exception):
            self.template_manager.render_template('nonexistent.tex')
    
    def test_jinja2_environment_setup(self):
        """Test Jinja2 environment configuration for LaTeX."""
        # Test that LaTeX-specific delimiters are used
        env = self.template_manager.env
        
        self.assertEqual(env.block_start_string, '\\BLOCK{')
        self.assertEqual(env.block_end_string, '}')
        self.assertEqual(env.variable_start_string, '\\VAR{')
        self.assertEqual(env.variable_end_string, '}')


class TestClaimFormatter(unittest.TestCase):
    """Test claim formatting for LaTeX inclusion."""
    
    def setUp(self):
        """Set up test environment."""
        self.formatter = ClaimFormatter()
        
        # Create mock promotion results
        self.promotion_results = {
            'promoted_claims': {
                'exact_match_improvement': {
                    'metric_name': 'exact_match',
                    'baseline_value': 0.75,
                    'bem_value': 0.85,
                    'improvement_percent': 13.3,
                    'confidence_interval': [0.08, 0.12],
                    'p_value': 0.001,
                    'effect_size': 0.8,
                    'effect_size_interpretation': 'Large',
                    'statistical_test': 'BCa Bootstrap',
                    'sample_size': 1000,
                    'validation_method': 'Cross-validation'
                },
                'f1_improvement': {
                    'metric_name': 'f1_score',
                    'baseline_value': 0.82,
                    'bem_value': 0.90,
                    'improvement_percent': 9.8,
                    'confidence_interval': [0.05, 0.11],
                    'p_value': 0.003,
                    'effect_size': 0.65,
                    'effect_size_interpretation': 'Medium',
                    'statistical_test': 'BCa Bootstrap',
                    'sample_size': 1000,
                    'validation_method': 'Cross-validation'
                }
            }
        }
    
    def test_promoted_claims_formatting(self):
        """Test formatting of promoted claims."""
        claims = self.formatter.format_promoted_claims(self.promotion_results)
        
        # Should return list of PromotedClaim objects
        self.assertIsInstance(claims, list)
        self.assertEqual(len(claims), 2)
        
        for claim in claims:
            self.assertIsInstance(claim, PromotedClaim)
            
        # Check sorting (by effect size descending)
        self.assertGreaterEqual(claims[0].effect_size, claims[1].effect_size)
        
        # Check specific claim content
        exact_match_claim = next(c for c in claims if c.metric_name == 'exact_match')
        self.assertEqual(exact_match_claim.baseline_value, 0.75)
        self.assertEqual(exact_match_claim.bem_value, 0.85)
        self.assertEqual(exact_match_claim.improvement_percent, 13.3)
        self.assertEqual(exact_match_claim.effect_size_interpretation, 'Large')
    
    def test_empty_promotion_results(self):
        """Test handling of empty promotion results."""
        empty_results = {'promoted_claims': {}}
        claims = self.formatter.format_promoted_claims(empty_results)
        
        self.assertIsInstance(claims, list)
        self.assertEqual(len(claims), 0)
    
    def test_results_table_creation(self):
        """Test creation of LaTeX results table."""
        claims = self.formatter.format_promoted_claims(self.promotion_results)
        table_data = self.formatter.create_results_table(claims)
        
        # Check table structure
        self.assertIn('caption', table_data)
        self.assertIn('label', table_data)
        self.assertIn('column_spec', table_data)
        self.assertIn('header', table_data)
        self.assertIn('rows', table_data)
        
        # Check content
        self.assertEqual(len(table_data['rows']), 2)  # Two promoted claims
        
        # Check that metric names are LaTeX-escaped
        first_row = table_data['rows'][0]
        self.assertIn('exact\\_match', ' & '.join(first_row))  # Underscore should be escaped
    
    def test_empty_results_table(self):
        """Test results table with no promoted claims."""
        table_data = self.formatter.create_results_table([])
        
        self.assertIn('No claims were promoted', table_data['caption'])
        self.assertEqual(len(table_data['rows']), 1)
        self.assertIn('No results', table_data['rows'][0])


class TestFigureGenerator(unittest.TestCase):
    """Test figure generation for papers."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.figure_generator = FigureGenerator(self.temp_dir)
        
        # Create test claims
        self.test_claims = [
            PromotedClaim(
                claim_id="test_1",
                metric_name="exact_match",
                baseline_value=0.75,
                bem_value=0.85,
                improvement_percent=13.3,
                confidence_interval=(0.08, 0.12),
                p_value=0.001,
                effect_size=0.8,
                effect_size_interpretation="Large",
                statistical_test="BCa Bootstrap",
                sample_size=1000,
                validation_method="Cross-validation"
            ),
            PromotedClaim(
                claim_id="test_2",
                metric_name="f1_score",
                baseline_value=0.82,
                bem_value=0.90,
                improvement_percent=9.8,
                confidence_interval=(0.05, 0.11),
                p_value=0.003,
                effect_size=0.65,
                effect_size_interpretation="Medium",
                statistical_test="BCa Bootstrap",
                sample_size=1000,
                validation_method="Cross-validation"
            )
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_improvement_barplot(self, mock_close, mock_savefig):
        """Test improvement bar plot generation."""
        figure_path = self.figure_generator.create_improvement_barplot(
            self.test_claims, "test_barplot.png"
        )
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
        
        # Check return path
        expected_path = str(self.temp_dir / "test_barplot.png")
        self.assertEqual(figure_path, expected_path)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_effect_size_plot(self, mock_close, mock_savefig):
        """Test effect size plot generation."""
        figure_path = self.figure_generator.create_effect_size_plot(
            self.test_claims, "test_effect_sizes.png"
        )
        
        mock_savefig.assert_called_once()
        expected_path = str(self.temp_dir / "test_effect_sizes.png")
        self.assertEqual(figure_path, expected_path)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_confidence_interval_plot(self, mock_close, mock_savefig):
        """Test confidence interval plot generation."""
        figure_path = self.figure_generator.create_confidence_interval_plot(
            self.test_claims, "test_ci.png"
        )
        
        mock_savefig.assert_called_once()
        expected_path = str(self.temp_dir / "test_ci.png")
        self.assertEqual(figure_path, expected_path)
    
    def test_empty_claims_handling(self):
        """Test handling of empty claims list."""
        figure_path = self.figure_generator.create_improvement_barplot([], "empty.png")
        self.assertEqual(figure_path, "")  # Should return empty string for no claims
    
    @patch('seaborn.set_palette')
    @patch('matplotlib.pyplot.style.use')
    def test_style_initialization(self, mock_style, mock_palette):
        """Test that plotting style is properly initialized."""
        FigureGenerator(self.temp_dir)
        
        # Should set style and palette
        mock_style.assert_called_once()
        mock_palette.assert_called_once()


class TestPaperGenerator(unittest.TestCase):
    """Test complete paper generation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.paper_generator = PaperGenerator(
            template_dir=self.temp_dir / "templates",
            output_dir=self.temp_dir / "papers",
            figures_dir=self.temp_dir / "figures"
        )
        
        # Create mock promotion results
        self.promotion_results = {
            'promoted_claims': {
                'exact_match_improvement': {
                    'metric_name': 'exact_match',
                    'baseline_value': 0.75,
                    'bem_value': 0.85,
                    'improvement_percent': 13.3,
                    'confidence_interval': [0.08, 0.12],
                    'p_value': 0.001,
                    'effect_size': 0.8,
                    'effect_size_interpretation': 'Large',
                    'statistical_test': 'BCa Bootstrap',
                    'sample_size': 1000,
                    'validation_method': 'Cross-validation'
                }
            },
            'summary': {
                'total_claims': 5,
                'promoted': 1,
                'demoted': 4
            },
            'metadata': {
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('paper_generator.PaperGenerator._compile_pdf')
    def test_paper_generation_workflow(self, mock_compile):
        """Test complete paper generation workflow."""
        mock_compile.return_value = str(self.temp_dir / "papers" / "test_paper" / "main.pdf")
        
        # Generate paper
        paper_path = self.paper_generator.generate_paper(
            promotion_results=self.promotion_results,
            metadata={
                'title': 'Test BEM Validation Results',
                'authors': ['Test Author'],
                'institution': 'Test Institution'
            },
            paper_id='test_paper'
        )
        
        # Check that paper path is returned
        self.assertIn("main.pdf", paper_path)
        
        # Check that LaTeX sections were generated
        paper_dir = self.temp_dir / "papers" / "test_paper"
        expected_sections = [
            'introduction.tex', 'methodology.tex', 'results.tex', 'conclusion.tex'
        ]
        
        for section in expected_sections:
            section_path = paper_dir / section
            self.assertTrue(section_path.exists(), f"Section {section} was not generated")
        
        # Check main document
        main_tex_path = paper_dir / 'main.tex'
        self.assertTrue(main_tex_path.exists())
        
        # Check that main document includes sections
        with open(main_tex_path, 'r') as f:
            main_content = f.read()
        
        for section in expected_sections:
            section_include = f"\\input{{{section.replace('.tex', '')}}}"
            self.assertIn(section_include, main_content)
    
    def test_template_variable_preparation(self):
        """Test preparation of template variables."""
        promoted_claims = self.paper_generator.claim_formatter.format_promoted_claims(
            self.promotion_results
        )
        
        template_vars = self.paper_generator._prepare_template_variables(
            self.promotion_results, promoted_claims, None, "test_paper"
        )
        
        # Check required variables
        required_vars = [
            'title', 'authors', 'promoted_claims', 'total_claims_tested',
            'claims_promoted', 'claims_demoted', 'generation_date'
        ]
        
        for var in required_vars:
            self.assertIn(var, template_vars)
        
        # Check specific values
        self.assertEqual(template_vars['claims_promoted'], 1)
        self.assertEqual(template_vars['claims_demoted'], 4)
        self.assertEqual(len(template_vars['promoted_claims']), 1)
    
    @patch('subprocess.run')
    def test_pdf_compilation(self, mock_run):
        """Test PDF compilation process."""
        mock_run.return_value = Mock(returncode=0)
        
        # Create temporary paper directory
        paper_dir = self.temp_dir / "test_paper"
        paper_dir.mkdir()
        
        # Create main.tex file
        main_tex_path = paper_dir / "main.tex"
        main_tex_path.write_text("\\documentclass{article}\n\\begin{document}Test\\end{document}")
        
        # Test PDF compilation
        pdf_path = self.paper_generator._compile_pdf(paper_dir, main_tex_path)
        
        # Check that pdflatex was called
        self.assertTrue(mock_run.called)
        
        # Check PDF path
        expected_pdf_path = str(paper_dir / "main.pdf")
        self.assertEqual(pdf_path, expected_pdf_path)
    
    def test_paper_generation_with_no_promoted_claims(self):
        """Test paper generation when no claims are promoted."""
        empty_results = {
            'promoted_claims': {},
            'summary': {'total_claims': 5, 'promoted': 0, 'demoted': 5},
            'metadata': {'version': '1.0.0'}
        }
        
        # Should handle gracefully without crashing
        with patch('paper_generator.PaperGenerator._compile_pdf') as mock_compile:
            mock_compile.return_value = str(self.temp_dir / "empty_paper.pdf")
            
            paper_path = self.paper_generator.generate_paper(
                promotion_results=empty_results
            )
            
            self.assertIn(".pdf", paper_path)


class TestVersionedPaperManager(unittest.TestCase):
    """Test versioned paper management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.versioned_manager = VersionedPaperManager(self.temp_dir / "versioned_papers")
        
        # Mock promotion results
        self.promotion_results = {
            'promoted_claims': {
                'test_claim': {
                    'metric_name': 'exact_match',
                    'improvement_percent': 10.0,
                    'p_value': 0.02,
                    'effect_size': 0.6
                }
            },
            'summary': {'promoted': 1, 'demoted': 2},
            'metadata': {'version': '1.0.0'}
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_registry_initialization(self):
        """Test paper registry initialization."""
        # Check that registry file exists
        registry_file = self.temp_dir / "versioned_papers" / "paper_registry.json"
        self.assertTrue(registry_file.exists())
        
        # Check registry structure
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        self.assertIn('papers', registry)
        self.assertIn('created', registry)
        self.assertEqual(len(registry['papers']), 0)  # Initially empty
    
    @patch('paper_generator.PaperGenerator.generate_paper')
    def test_versioned_paper_creation(self, mock_generate):
        """Test creation of versioned paper."""
        mock_generate.return_value = str(self.temp_dir / "test_paper.pdf")
        
        # Create versioned paper
        paper_path = self.versioned_manager.create_versioned_paper(
            promotion_results=self.promotion_results,
            metadata={'title': 'Test Versioned Paper'},
            version_tag='v1.0.0'
        )
        
        # Check that paper was generated
        mock_generate.assert_called_once()
        self.assertIn(".pdf", paper_path)
        
        # Check that version directory was created
        version_dir = self.temp_dir / "versioned_papers" / "v1.0.0"
        self.assertTrue(version_dir.exists())
        
        # Check that reproducibility metadata was created
        metadata_file = version_dir / "reproducibility_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        # Check registry was updated
        with open(self.temp_dir / "versioned_papers" / "paper_registry.json", 'r') as f:
            registry = json.load(f)
        
        self.assertIn('v1.0.0', registry['papers'])
        self.assertEqual(len(registry['papers']), 1)
    
    def test_paper_listing(self):
        """Test listing of versioned papers."""
        # Initially empty
        papers = self.versioned_manager.list_papers()
        self.assertEqual(len(papers), 0)
        
        # Add mock paper to registry
        mock_paper_info = {
            'version_tag': 'v1.0.0',
            'creation_timestamp': datetime.now().isoformat(),
            'paper_path': '/test/path/paper.pdf',
            'reproducibility_hash': 'abc123'
        }
        
        self.versioned_manager.registry['papers']['v1.0.0'] = mock_paper_info
        self.versioned_manager._save_registry()
        
        # Should return the paper
        papers = self.versioned_manager.list_papers()
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['version_tag'], 'v1.0.0')
    
    def test_paper_info_retrieval(self):
        """Test retrieval of specific paper information."""
        # Add mock paper to registry
        mock_paper_info = {
            'version_tag': 'v1.0.0',
            'creation_timestamp': datetime.now().isoformat(),
            'paper_path': '/test/path/paper.pdf',
            'reproducibility_hash': 'abc123'
        }
        
        self.versioned_manager.registry['papers']['v1.0.0'] = mock_paper_info
        
        # Retrieve paper info
        info = self.versioned_manager.get_paper_info('v1.0.0')
        self.assertIsNotNone(info)
        self.assertEqual(info['version_tag'], 'v1.0.0')
        
        # Non-existent paper
        info = self.versioned_manager.get_paper_info('v999.999.999')
        self.assertIsNone(info)
    
    def test_reproducibility_metadata_creation(self):
        """Test creation of reproducibility metadata."""
        metadata = self.versioned_manager._create_reproducibility_metadata(
            self.promotion_results,
            {'title': 'Test Paper'},
            'v1.0.0',
            '/test/paper.pdf'
        )
        
        # Check required fields
        required_fields = [
            'version_tag', 'creation_timestamp', 'reproducibility_hash',
            'environment', 'validation_summary'
        ]
        
        for field in required_fields:
            self.assertIn(field, metadata)
        
        # Check environment info
        self.assertIn('python_version', metadata['environment'])
        self.assertIn('platform', metadata['environment'])
        
        # Check validation summary
        self.assertEqual(metadata['validation_summary']['promoted_claims'], 1)


class TestPaperGenerationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_malformed_promotion_results(self):
        """Test handling of malformed promotion results."""
        malformed_results = {
            'promoted_claims': {
                'incomplete_claim': {
                    'metric_name': 'exact_match',
                    # Missing required fields
                }
            }
        }
        
        formatter = ClaimFormatter()
        
        # Should handle gracefully without crashing
        claims = formatter.format_promoted_claims(malformed_results)
        
        # May return empty list or skip malformed claims
        self.assertIsInstance(claims, list)
    
    def test_latex_special_characters(self):
        """Test handling of LaTeX special characters in claims."""
        results_with_special_chars = {
            'promoted_claims': {
                'special_claim': {
                    'metric_name': 'test_metric_with_underscore',  # Should be escaped
                    'baseline_value': 0.75,
                    'bem_value': 0.85,
                    'improvement_percent': 13.3,
                    'confidence_interval': [0.08, 0.12],
                    'p_value': 0.001,
                    'effect_size': 0.8,
                    'effect_size_interpretation': 'Large',
                    'statistical_test': 'BCa Bootstrap',
                    'sample_size': 1000,
                    'validation_method': 'Cross-validation'
                }
            }
        }
        
        formatter = ClaimFormatter()
        claims = formatter.format_promoted_claims(results_with_special_chars)
        table_data = formatter.create_results_table(claims)
        
        # Check that underscores are escaped in table
        table_content = ' & '.join(table_data['rows'][0])
        self.assertIn('test\\_metric\\_with\\_underscore', table_content)
    
    @patch('subprocess.run')
    def test_pdf_compilation_failure(self, mock_run):
        """Test handling of PDF compilation failure."""
        mock_run.return_value = Mock(returncode=1)  # Failure
        
        paper_generator = PaperGenerator(output_dir=self.temp_dir)
        
        paper_dir = self.temp_dir / "test_paper"
        paper_dir.mkdir()
        main_tex_path = paper_dir / "main.tex"
        main_tex_path.write_text("\\documentclass{article}\n\\begin{document}Test\\end{document}")
        
        # Should handle compilation failure gracefully
        pdf_path = paper_generator._compile_pdf(paper_dir, main_tex_path)
        
        # Should still return expected path even if compilation failed
        expected_path = str(paper_dir / "main.pdf")
        self.assertEqual(pdf_path, expected_path)
    
    def test_figure_generation_with_matplotlib_unavailable(self):
        """Test graceful handling when matplotlib is not available."""
        # This would typically be tested by mocking matplotlib import failure
        # For now, we just test that the figure generator can be initialized
        figure_gen = FigureGenerator(self.temp_dir)
        self.assertIsNotNone(figure_gen)
    
    def test_very_long_claim_names(self):
        """Test handling of very long claim names."""
        long_name_results = {
            'promoted_claims': {
                'extremely_long_claim_name_that_might_cause_issues': {
                    'metric_name': 'very_long_metric_name_with_many_underscores_and_details',
                    'baseline_value': 0.75,
                    'bem_value': 0.85,
                    'improvement_percent': 13.3,
                    'confidence_interval': [0.08, 0.12],
                    'p_value': 0.001,
                    'effect_size': 0.8,
                    'effect_size_interpretation': 'Large',
                    'statistical_test': 'BCa Bootstrap',
                    'sample_size': 1000,
                    'validation_method': 'Cross-validation'
                }
            }
        }
        
        formatter = ClaimFormatter()
        
        # Should handle long names without issues
        claims = formatter.format_promoted_claims(long_name_results)
        self.assertEqual(len(claims), 1)
        
        table_data = formatter.create_results_table(claims)
        self.assertEqual(len(table_data['rows']), 1)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)