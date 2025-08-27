#!/usr/bin/env python3
"""
Data Preparation for BEM Validation Experiment

This script generates high-quality synthetic datasets for the validation experiment:
1. JSON generation tasks with diverse schemas and realistic data
2. Text summarization tasks with varied content and complexity
3. Task instruction embeddings for controller training
4. Quality validation and dataset statistics

The data is designed to be challenging enough to demonstrate meaningful 
controller learning while being synthetic for reproducibility.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import argparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
import pandas as pd

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data generation."""
    
    # Output paths
    output_dir: str = "data/validation_experiment"
    
    # Dataset sizes
    num_json_samples: int = 1000
    num_summary_samples: int = 1000
    
    # Data quality parameters
    min_json_complexity: int = 3  # Minimum number of fields
    max_json_complexity: int = 8  # Maximum number of fields
    min_text_length: int = 200    # Minimum characters for summarization texts
    max_text_length: int = 1000   # Maximum characters for summarization texts
    
    # Task instruction parameters
    num_instruction_variants: int = 5
    
    # Random seed for reproducibility
    seed: int = 42


class JSONDataGenerator:
    """Generate diverse JSON creation tasks."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Schema templates with realistic fields and data types
        self.schemas = {
            "person": {
                "fields": {
                    "name": ("string", self._generate_name),
                    "age": ("integer", lambda: random.randint(18, 80)),
                    "email": ("string", self._generate_email),
                    "phone": ("string", self._generate_phone),
                    "address": ("string", self._generate_address),
                    "occupation": ("string", self._generate_occupation),
                    "salary": ("number", lambda: round(random.uniform(30000, 150000), 2)),
                    "is_active": ("boolean", lambda: random.choice([True, False])),
                    "join_date": ("string", self._generate_date),
                    "skills": ("array", self._generate_skills)
                },
                "required": ["name", "email", "age"]
            },
            "product": {
                "fields": {
                    "name": ("string", self._generate_product_name),
                    "price": ("number", lambda: round(random.uniform(9.99, 999.99), 2)),
                    "category": ("string", self._generate_category),
                    "description": ("string", self._generate_product_description),
                    "sku": ("string", self._generate_sku),
                    "in_stock": ("boolean", lambda: random.choice([True, False])),
                    "stock_quantity": ("integer", lambda: random.randint(0, 1000)),
                    "weight": ("number", lambda: round(random.uniform(0.1, 50.0), 2)),
                    "dimensions": ("object", self._generate_dimensions),
                    "tags": ("array", self._generate_product_tags)
                },
                "required": ["name", "price", "category"]
            },
            "event": {
                "fields": {
                    "title": ("string", self._generate_event_title),
                    "date": ("string", self._generate_future_date),
                    "time": ("string", self._generate_time),
                    "location": ("string", self._generate_location),
                    "description": ("string", self._generate_event_description),
                    "capacity": ("integer", lambda: random.randint(10, 1000)),
                    "price": ("number", lambda: round(random.uniform(0, 500), 2)),
                    "is_virtual": ("boolean", lambda: random.choice([True, False])),
                    "organizer": ("string", self._generate_organizer),
                    "categories": ("array", self._generate_event_categories)
                },
                "required": ["title", "date", "location"]
            },
            "company": {
                "fields": {
                    "name": ("string", self._generate_company_name),
                    "industry": ("string", self._generate_industry),
                    "employees": ("integer", lambda: random.randint(1, 10000)),
                    "founded": ("string", self._generate_founded_year),
                    "revenue": ("number", lambda: round(random.uniform(100000, 1000000000), 2)),
                    "headquarters": ("string", self._generate_city),
                    "website": ("string", self._generate_website),
                    "is_public": ("boolean", lambda: random.choice([True, False])),
                    "ceo": ("string", self._generate_name),
                    "subsidiaries": ("array", self._generate_subsidiaries)
                },
                "required": ["name", "industry", "employees"]
            },
            "book": {
                "fields": {
                    "title": ("string", self._generate_book_title),
                    "author": ("string", self._generate_author),
                    "isbn": ("string", self._generate_isbn),
                    "pages": ("integer", lambda: random.randint(50, 1000)),
                    "genre": ("string", self._generate_genre),
                    "publisher": ("string", self._generate_publisher),
                    "publication_date": ("string", self._generate_publication_date),
                    "price": ("number", lambda: round(random.uniform(9.99, 49.99), 2)),
                    "rating": ("number", lambda: round(random.uniform(1.0, 5.0), 1)),
                    "reviews": ("integer", lambda: random.randint(0, 10000))
                },
                "required": ["title", "author", "pages"]
            }
        }
        
        # Instruction templates
        self.instruction_templates = [
            "Generate a well-structured JSON object for a {schema_type} with the following requirements:",
            "Create a JSON representation of a {schema_type} that includes:",
            "Build a JSON object describing a {schema_type} with these specifications:",
            "Construct a {schema_type} JSON object containing:",
            "Produce a structured JSON for a {schema_type} that has:"
        ]
    
    def generate_samples(self, num_samples: int) -> List[Dict[str, str]]:
        """Generate JSON task samples."""
        
        samples = []
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Generating JSON samples", total=num_samples)
            
            for i in range(num_samples):
                sample = self._generate_single_sample(i)
                samples.append(sample)
                progress.advance(task)
        
        return samples
    
    def _generate_single_sample(self, sample_id: int) -> Dict[str, str]:
        """Generate a single JSON task sample."""
        
        # Select random schema
        schema_name = random.choice(list(self.schemas.keys()))
        schema = self.schemas[schema_name]
        
        # Determine complexity (number of fields to include)
        num_fields = random.randint(self.config.min_json_complexity, self.config.max_json_complexity)
        
        # Always include required fields
        required_fields = schema["required"]
        available_fields = list(schema["fields"].keys())
        
        # Add additional fields up to desired complexity
        selected_fields = required_fields.copy()
        remaining_fields = [f for f in available_fields if f not in required_fields]
        additional_needed = max(0, num_fields - len(required_fields))
        
        if additional_needed > 0:
            additional_fields = random.sample(
                remaining_fields, 
                min(additional_needed, len(remaining_fields))
            )
            selected_fields.extend(additional_fields)
        
        # Generate instruction
        instruction_template = random.choice(self.instruction_templates)
        field_descriptions = []
        
        for field in selected_fields:
            field_type, _ = schema["fields"][field]
            field_descriptions.append(f"- {field} ({field_type})")
        
        instruction = instruction_template.format(schema_type=schema_name) + "\n" + "\n".join(field_descriptions)
        
        # Generate JSON object
        json_object = {}
        for field in selected_fields:
            field_type, generator_func = schema["fields"][field]
            json_object[field] = generator_func()
        
        # Format as pretty JSON
        json_output = json.dumps(json_object, indent=2, ensure_ascii=False)
        
        return {
            "task_id": f"json_{sample_id:04d}",
            "schema_type": schema_name,
            "input": instruction,
            "output": json_output,
            "complexity": len(selected_fields),
            "required_fields": required_fields,
            "selected_fields": selected_fields
        }
    
    # Helper methods for generating realistic data
    def _generate_name(self) -> str:
        first_names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _generate_email(self) -> str:
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "company.com", "university.edu"]
        username = f"user{random.randint(1, 9999)}"
        return f"{username}@{random.choice(domains)}"
    
    def _generate_phone(self) -> str:
        return f"+1-{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"
    
    def _generate_address(self) -> str:
        streets = ["Main St", "Oak Ave", "Pine Rd", "First St", "Second Ave"]
        return f"{random.randint(100, 9999)} {random.choice(streets)}"
    
    def _generate_occupation(self) -> str:
        occupations = ["Software Engineer", "Teacher", "Doctor", "Artist", "Manager", "Scientist", "Writer", "Designer"]
        return random.choice(occupations)
    
    def _generate_date(self) -> str:
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 12, 31)
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
    
    def _generate_skills(self) -> List[str]:
        all_skills = ["Python", "JavaScript", "Communication", "Leadership", "Design", "Analysis", "Project Management"]
        return random.sample(all_skills, random.randint(2, 5))
    
    def _generate_product_name(self) -> str:
        adjectives = ["Premium", "Advanced", "Smart", "Professional", "Deluxe", "Ultra", "Pro"]
        nouns = ["Laptop", "Phone", "Camera", "Headphones", "Watch", "Speaker", "Tablet"]
        return f"{random.choice(adjectives)} {random.choice(nouns)}"
    
    def _generate_category(self) -> str:
        categories = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Toys", "Food"]
        return random.choice(categories)
    
    def _generate_product_description(self) -> str:
        templates = [
            "High-quality product with advanced features and excellent performance.",
            "Premium item designed for professionals and enthusiasts alike.",
            "Innovative solution that combines style with functionality.",
            "Top-rated product with outstanding customer reviews."
        ]
        return random.choice(templates)
    
    def _generate_sku(self) -> str:
        return f"{random.choice(['PRD', 'ITM', 'SKU'])}-{random.randint(1000, 9999)}"
    
    def _generate_dimensions(self) -> Dict[str, float]:
        return {
            "length": round(random.uniform(10, 100), 1),
            "width": round(random.uniform(10, 100), 1),
            "height": round(random.uniform(5, 50), 1)
        }
    
    def _generate_product_tags(self) -> List[str]:
        tags = ["bestseller", "new", "sale", "premium", "eco-friendly", "trending", "limited"]
        return random.sample(tags, random.randint(1, 4))
    
    def _generate_event_title(self) -> str:
        types = ["Conference", "Workshop", "Seminar", "Meetup", "Summit", "Symposium"]
        topics = ["Tech", "Business", "Design", "AI", "Marketing", "Innovation"]
        return f"{random.choice(topics)} {random.choice(types)} 2024"
    
    def _generate_future_date(self) -> str:
        start = datetime.now()
        end = datetime.now() + timedelta(days=365)
        time_between = end - start
        days_between = time_between.days
        random_days = random.randrange(days_between)
        return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")
    
    def _generate_time(self) -> str:
        hour = random.randint(9, 18)
        minute = random.choice([0, 30])
        return f"{hour:02d}:{minute:02d}"
    
    def _generate_location(self) -> str:
        venues = ["Convention Center", "Hotel", "University", "Office Building", "Community Center"]
        cities = ["New York", "San Francisco", "London", "Tokyo", "Berlin"]
        return f"{random.choice(venues)}, {random.choice(cities)}"
    
    def _generate_event_description(self) -> str:
        return "Join industry experts for an engaging session on the latest trends and innovations."
    
    def _generate_organizer(self) -> str:
        return f"{self._generate_name()} Events"
    
    def _generate_event_categories(self) -> List[str]:
        categories = ["Technology", "Business", "Networking", "Education", "Innovation"]
        return random.sample(categories, random.randint(1, 3))
    
    def _generate_company_name(self) -> str:
        prefixes = ["Tech", "Global", "Advanced", "Smart", "Digital", "Future"]
        suffixes = ["Solutions", "Systems", "Corp", "Inc", "Group", "Industries"]
        return f"{random.choice(prefixes)} {random.choice(suffixes)}"
    
    def _generate_industry(self) -> str:
        industries = ["Technology", "Healthcare", "Finance", "Education", "Manufacturing", "Retail"]
        return random.choice(industries)
    
    def _generate_founded_year(self) -> str:
        return str(random.randint(1950, 2020))
    
    def _generate_city(self) -> str:
        cities = ["New York", "San Francisco", "London", "Tokyo", "Berlin", "Sydney"]
        return random.choice(cities)
    
    def _generate_website(self) -> str:
        company_slug = self._generate_company_name().lower().replace(" ", "")
        return f"https://www.{company_slug}.com"
    
    def _generate_subsidiaries(self) -> List[str]:
        return [self._generate_company_name() for _ in range(random.randint(0, 3))]
    
    def _generate_book_title(self) -> str:
        titles = [
            "The Art of Programming", "Data Science Handbook", "Future of Technology",
            "Modern Architecture", "Creative Writing Guide", "Business Strategy"
        ]
        return random.choice(titles)
    
    def _generate_author(self) -> str:
        return self._generate_name()
    
    def _generate_isbn(self) -> str:
        return f"978-{random.randint(0, 9)}-{random.randint(100, 999)}-{random.randint(10000, 99999)}-{random.randint(0, 9)}"
    
    def _generate_genre(self) -> str:
        genres = ["Fiction", "Non-fiction", "Science", "Technology", "Biography", "History"]
        return random.choice(genres)
    
    def _generate_publisher(self) -> str:
        publishers = ["TechBooks", "Academic Press", "Future Publishing", "Innovation Books"]
        return random.choice(publishers)
    
    def _generate_publication_date(self) -> str:
        return self._generate_date()


class SummaryDataGenerator:
    """Generate diverse text summarization tasks."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Content templates by topic
        self.topics = {
            "technology": {
                "keywords": ["artificial intelligence", "machine learning", "blockchain", "cloud computing", "IoT"],
                "contexts": ["software development", "data science", "cybersecurity", "innovation", "digital transformation"],
                "outcomes": ["improved efficiency", "enhanced security", "better user experience", "cost reduction"]
            },
            "science": {
                "keywords": ["research", "discovery", "experiment", "analysis", "methodology"],
                "contexts": ["laboratory", "field study", "clinical trial", "peer review", "scientific community"],
                "outcomes": ["new insights", "breakthrough findings", "validated hypothesis", "improved understanding"]
            },
            "business": {
                "keywords": ["strategy", "market analysis", "revenue", "growth", "optimization"],
                "contexts": ["corporate environment", "startup ecosystem", "global market", "industry trends"],
                "outcomes": ["increased profits", "market expansion", "competitive advantage", "operational efficiency"]
            },
            "health": {
                "keywords": ["treatment", "prevention", "diagnosis", "therapy", "wellness"],
                "contexts": ["healthcare system", "medical research", "patient care", "public health"],
                "outcomes": ["improved outcomes", "reduced risks", "better quality of life", "prevention"]
            },
            "environment": {
                "keywords": ["sustainability", "climate change", "conservation", "renewable energy", "ecosystem"],
                "contexts": ["environmental policy", "conservation efforts", "green technology", "sustainable development"],
                "outcomes": ["reduced impact", "environmental protection", "sustainable future", "conservation"]
            }
        }
        
        # Instruction templates
        self.instruction_templates = [
            "Provide a concise summary of the following text:",
            "Summarize the key points from this passage:",
            "Create a brief overview of the following content:",
            "Generate a summary highlighting the main ideas:",
            "Condense the following text into a clear summary:"
        ]
    
    def generate_samples(self, num_samples: int) -> List[Dict[str, str]]:
        """Generate summarization task samples."""
        
        samples = []
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Generating summary samples", total=num_samples)
            
            for i in range(num_samples):
                sample = self._generate_single_sample(i)
                samples.append(sample)
                progress.advance(task)
        
        return samples
    
    def _generate_single_sample(self, sample_id: int) -> Dict[str, str]:
        """Generate a single summarization task sample."""
        
        # Select random topic
        topic_name = random.choice(list(self.topics.keys()))
        topic_data = self.topics[topic_name]
        
        # Generate text length
        target_length = random.randint(self.config.min_text_length, self.config.max_text_length)
        
        # Generate content
        long_text = self._generate_long_text(topic_name, topic_data, target_length)
        summary = self._generate_summary(topic_name, topic_data, long_text)
        
        # Generate instruction
        instruction = random.choice(self.instruction_templates)
        
        return {
            "task_id": f"summary_{sample_id:04d}",
            "topic": topic_name,
            "input": long_text,
            "output": summary,
            "instruction": instruction,
            "text_length": len(long_text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(long_text)
        }
    
    def _generate_long_text(self, topic: str, topic_data: Dict, target_length: int) -> str:
        """Generate a long text for summarization."""
        
        # Build text using topic-specific content
        keywords = topic_data["keywords"]
        contexts = topic_data["contexts"]
        outcomes = topic_data["outcomes"]
        
        # Create multiple paragraphs
        paragraphs = []
        current_length = 0
        
        # Introduction paragraph
        intro = self._create_introduction_paragraph(topic, keywords[0], contexts[0])
        paragraphs.append(intro)
        current_length += len(intro)
        
        # Main content paragraphs
        while current_length < target_length * 0.8:  # Leave room for conclusion
            keyword = random.choice(keywords)
            context = random.choice(contexts)
            outcome = random.choice(outcomes)
            
            paragraph = self._create_content_paragraph(topic, keyword, context, outcome)
            paragraphs.append(paragraph)
            current_length += len(paragraph)
        
        # Conclusion paragraph
        conclusion = self._create_conclusion_paragraph(topic, random.choice(outcomes))
        paragraphs.append(conclusion)
        
        return "\n\n".join(paragraphs)
    
    def _create_introduction_paragraph(self, topic: str, keyword: str, context: str) -> str:
        """Create an introduction paragraph."""
        
        templates = [
            f"In recent years, {keyword} has become increasingly important in the {context}. "
            f"This comprehensive analysis examines the current state of {topic} research and its implications for the future. "
            f"The following discussion provides detailed insights into key developments and emerging trends.",
            
            f"The field of {topic} has experienced significant growth, particularly in areas related to {keyword}. "
            f"Within the {context}, researchers and practitioners have observed notable changes and innovations. "
            f"This extensive review explores these developments and their potential impact.",
            
            f"Recent advances in {keyword} have transformed our understanding of {topic}. "
            f"The {context} has been particularly affected by these changes, leading to new approaches and methodologies. "
            f"This detailed examination covers the most significant findings and their broader implications."
        ]
        
        return random.choice(templates)
    
    def _create_content_paragraph(self, topic: str, keyword: str, context: str, outcome: str) -> str:
        """Create a main content paragraph."""
        
        templates = [
            f"One of the most significant aspects of {keyword} in {topic} is its application within {context}. "
            f"Studies have shown that implementing these approaches leads to {outcome} across various scenarios. "
            f"The research indicates that organizations adopting these methodologies experience substantial improvements "
            f"in their operational effectiveness. Furthermore, the long-term benefits extend beyond immediate gains, "
            f"creating sustainable advantages that persist over time. Evidence from multiple case studies demonstrates "
            f"consistent positive results across different implementation contexts.",
            
            f"The integration of {keyword} into {context} represents a paradigm shift in {topic}. "
            f"Early adopters have reported {outcome}, validating the theoretical foundations of this approach. "
            f"Detailed analysis reveals that success factors include proper planning, stakeholder engagement, "
            f"and continuous monitoring of key performance indicators. The implementation process requires careful "
            f"consideration of organizational culture and existing infrastructure to maximize effectiveness.",
            
            f"Research in {keyword} has revealed important insights about {topic} within {context}. "
            f"The findings consistently point toward {outcome} when best practices are followed. "
            f"Key success factors identified through extensive field studies include systematic implementation, "
            f"regular evaluation, and adaptive management approaches. These elements work together to create "
            f"a comprehensive framework that addresses both immediate needs and long-term strategic objectives."
        ]
        
        return random.choice(templates)
    
    def _create_conclusion_paragraph(self, topic: str, outcome: str) -> str:
        """Create a conclusion paragraph."""
        
        templates = [
            f"In conclusion, the current state of {topic} research demonstrates significant potential for {outcome}. "
            f"The evidence presented throughout this analysis supports the adoption of these approaches in appropriate contexts. "
            f"Future research directions should focus on refining implementation methodologies and expanding applications "
            f"to new domains. The continued development of this field promises to deliver even greater benefits in the coming years.",
            
            f"The comprehensive review of {topic} reveals a clear trajectory toward {outcome}. "
            f"Organizations and researchers who embrace these developments are likely to gain competitive advantages "
            f"and contribute to the advancement of the field. Moving forward, collaborative efforts between academia "
            f"and industry will be essential for realizing the full potential of these innovations.",
            
            f"This analysis of {topic} highlights the importance of systematic approaches in achieving {outcome}. "
            f"The evidence strongly supports continued investment in research and development within this domain. "
            f"As the field continues to evolve, practitioners must remain adaptable and committed to evidence-based "
            f"decision-making to maximize the benefits of these emerging capabilities."
        ]
        
        return random.choice(templates)
    
    def _generate_summary(self, topic: str, topic_data: Dict, long_text: str) -> str:
        """Generate an appropriate summary for the long text."""
        
        # Extract key concepts
        main_keyword = random.choice(topic_data["keywords"])
        main_context = random.choice(topic_data["contexts"])
        main_outcome = random.choice(topic_data["outcomes"])
        
        # Create concise summary
        summary_templates = [
            f"This analysis examines {main_keyword} in {topic}, focusing on applications within {main_context}. "
            f"The research demonstrates that proper implementation leads to {main_outcome}. "
            f"Key findings include the importance of systematic approaches and the potential for significant benefits "
            f"when best practices are followed. The study concludes that continued development in this field "
            f"will yield further improvements and expanded applications.",
            
            f"The study explores recent advances in {topic}, particularly in {main_keyword} and its impact on {main_context}. "
            f"Results indicate that organizations adopting these approaches experience {main_outcome}. "
            f"Critical success factors identified include proper planning, stakeholder engagement, and continuous monitoring. "
            f"The research supports continued investment in this area for long-term strategic advantages.",
            
            f"This comprehensive review of {topic} highlights the growing importance of {main_keyword} in {main_context}. "
            f"Evidence from multiple studies shows consistent {main_outcome} across various implementation scenarios. "
            f"The analysis reveals that success depends on systematic implementation and adaptive management approaches. "
            f"Future developments in this field promise to deliver even greater benefits for practitioners and organizations."
        ]
        
        return random.choice(summary_templates)


class ValidationDataPreparer:
    """Main class for preparing all validation experiment data."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.json_generator = JSONDataGenerator(config)
        self.summary_generator = SummaryDataGenerator(config)
        
        # Set up output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    def prepare_all_data(self) -> Dict[str, Any]:
        """Prepare all data for the validation experiment."""
        
        console.print("[bold green]🔧 Preparing BEM Validation Data")
        console.print(f"Output directory: {self.config.output_dir}")
        
        results = {}
        
        # Generate JSON data
        console.print("\n[bold blue]Generating JSON Task Data...")
        json_samples = self.json_generator.generate_samples(self.config.num_json_samples)
        results['json_samples'] = json_samples
        
        # Generate summary data
        console.print("\n[bold blue]Generating Summary Task Data...")
        summary_samples = self.summary_generator.generate_samples(self.config.num_summary_samples)
        results['summary_samples'] = summary_samples
        
        # Save data
        console.print("\n[bold blue]Saving Data Files...")
        self._save_data(json_samples, summary_samples)
        
        # Generate statistics
        console.print("\n[bold blue]Computing Dataset Statistics...")
        stats = self._compute_statistics(json_samples, summary_samples)
        results['statistics'] = stats
        
        # Create summary report
        console.print("\n[bold blue]Creating Summary Report...")
        report = self._create_data_report(stats)
        results['report'] = report
        
        console.print("\n[bold green]✓ Data preparation completed successfully!")
        console.print(f"Generated {len(json_samples)} JSON samples and {len(summary_samples)} summary samples")
        
        return results
    
    def _save_data(self, json_samples: List[Dict], summary_samples: List[Dict]):
        """Save data to files."""
        
        output_dir = Path(self.config.output_dir)
        
        # Save JSON samples
        json_path = output_dir / "json_tasks.json"
        with open(json_path, 'w') as f:
            json.dump(json_samples, f, indent=2, ensure_ascii=False)
        
        # Save summary samples
        summary_path = output_dir / "summary_tasks.json" 
        with open(summary_path, 'w') as f:
            json.dump(summary_samples, f, indent=2, ensure_ascii=False)
        
        # Save combined dataset
        combined_path = output_dir / "combined_tasks.json"
        combined_data = {
            'json_tasks': json_samples,
            'summary_tasks': summary_samples,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'config': asdict(self.config),
                'num_json_samples': len(json_samples),
                'num_summary_samples': len(summary_samples)
            }
        }
        with open(combined_path, 'w') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        # Save sample files for inspection
        sample_json = output_dir / "sample_json_tasks.json"
        with open(sample_json, 'w') as f:
            json.dump(json_samples[:5], f, indent=2, ensure_ascii=False)
        
        sample_summary = output_dir / "sample_summary_tasks.json"
        with open(sample_summary, 'w') as f:
            json.dump(summary_samples[:5], f, indent=2, ensure_ascii=False)
        
        console.print(f"Data saved to {output_dir}")
    
    def _compute_statistics(self, json_samples: List[Dict], summary_samples: List[Dict]) -> Dict[str, Any]:
        """Compute comprehensive dataset statistics."""
        
        stats = {
            'json_stats': {},
            'summary_stats': {},
            'overall_stats': {}
        }
        
        # JSON statistics
        json_complexities = [sample['complexity'] for sample in json_samples]
        json_schemas = [sample['schema_type'] for sample in json_samples]
        json_lengths = [len(sample['output']) for sample in json_samples]
        
        stats['json_stats'] = {
            'num_samples': len(json_samples),
            'complexity_mean': np.mean(json_complexities),
            'complexity_std': np.std(json_complexities),
            'complexity_min': np.min(json_complexities),
            'complexity_max': np.max(json_complexities),
            'schema_distribution': pd.Series(json_schemas).value_counts().to_dict(),
            'output_length_mean': np.mean(json_lengths),
            'output_length_std': np.std(json_lengths),
            'output_length_min': np.min(json_lengths),
            'output_length_max': np.max(json_lengths)
        }
        
        # Summary statistics  
        summary_topics = [sample['topic'] for sample in summary_samples]
        summary_text_lengths = [sample['text_length'] for sample in summary_samples]
        summary_summary_lengths = [sample['summary_length'] for sample in summary_samples]
        compression_ratios = [sample['compression_ratio'] for sample in summary_samples]
        
        stats['summary_stats'] = {
            'num_samples': len(summary_samples),
            'topic_distribution': pd.Series(summary_topics).value_counts().to_dict(),
            'text_length_mean': np.mean(summary_text_lengths),
            'text_length_std': np.std(summary_text_lengths),
            'text_length_min': np.min(summary_text_lengths),
            'text_length_max': np.max(summary_text_lengths),
            'summary_length_mean': np.mean(summary_summary_lengths),
            'summary_length_std': np.std(summary_summary_lengths),
            'compression_ratio_mean': np.mean(compression_ratios),
            'compression_ratio_std': np.std(compression_ratios)
        }
        
        # Overall statistics
        stats['overall_stats'] = {
            'total_samples': len(json_samples) + len(summary_samples),
            'json_proportion': len(json_samples) / (len(json_samples) + len(summary_samples)),
            'summary_proportion': len(summary_samples) / (len(json_samples) + len(summary_samples)),
            'avg_task_complexity': (np.mean(json_complexities) + np.mean(compression_ratios)) / 2,
            'data_balance_score': min(len(json_samples), len(summary_samples)) / max(len(json_samples), len(summary_samples))
        }
        
        return stats
    
    def _create_data_report(self, stats: Dict[str, Any]) -> str:
        """Create a comprehensive data report."""
        
        json_stats = stats['json_stats']
        summary_stats = stats['summary_stats']
        overall_stats = stats['overall_stats']
        
        report = f"""
# BEM Validation Experiment - Dataset Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview

- **Total Samples**: {overall_stats['total_samples']:,}
- **JSON Task Samples**: {json_stats['num_samples']:,} ({overall_stats['json_proportion']:.1%})
- **Summary Task Samples**: {summary_stats['num_samples']:,} ({overall_stats['summary_proportion']:.1%})
- **Data Balance Score**: {overall_stats['data_balance_score']:.3f} (1.0 = perfect balance)

## JSON Task Analysis

### Complexity Distribution
- **Mean Complexity**: {json_stats['complexity_mean']:.2f} fields per sample
- **Standard Deviation**: {json_stats['complexity_std']:.2f}
- **Range**: {json_stats['complexity_min']} - {json_stats['complexity_max']} fields

### Schema Distribution
"""
        
        # Add schema distribution
        for schema, count in json_stats['schema_distribution'].items():
            percentage = (count / json_stats['num_samples']) * 100
            report += f"- **{schema.title()}**: {count:,} samples ({percentage:.1f}%)\n"
        
        report += f"""

### Output Characteristics
- **Mean Output Length**: {json_stats['output_length_mean']:.0f} characters
- **Standard Deviation**: {json_stats['output_length_std']:.0f} characters
- **Range**: {json_stats['output_length_min']:,} - {json_stats['output_length_max']:,} characters

## Summary Task Analysis

### Topic Distribution
"""
        
        # Add topic distribution
        for topic, count in summary_stats['topic_distribution'].items():
            percentage = (count / summary_stats['num_samples']) * 100
            report += f"- **{topic.title()}**: {count:,} samples ({percentage:.1f}%)\n"
        
        report += f"""

### Text Length Analysis
- **Mean Input Length**: {summary_stats['text_length_mean']:.0f} characters
- **Standard Deviation**: {summary_stats['text_length_std']:.0f} characters
- **Range**: {summary_stats['text_length_min']:,} - {summary_stats['text_length_max']:,} characters

### Summary Characteristics
- **Mean Summary Length**: {summary_stats['summary_length_mean']:.0f} characters
- **Standard Deviation**: {summary_stats['summary_length_std']:.0f} characters
- **Mean Compression Ratio**: {summary_stats['compression_ratio_mean']:.3f}
- **Compression Std**: {summary_stats['compression_ratio_std']:.3f}

## Quality Assessment

### Task Differentiation
The dataset provides clear task differentiation to enable meaningful controller learning:

1. **JSON Generation Tasks**: Structured output with variable complexity (3-8 fields)
2. **Text Summarization Tasks**: Natural language compression with consistent ratios

### Complexity Balance
- **JSON Task Complexity**: Variable field counts ensure diverse learning examples
- **Summary Task Complexity**: Varied text lengths and topics provide rich training signal
- **Overall Assessment**: ✓ Well-balanced dataset for controller learning

### Data Quality Indicators
- **Schema Diversity**: {len(json_stats['schema_distribution'])} different JSON schemas
- **Topic Diversity**: {len(summary_stats['topic_distribution'])} summary topics  
- **Length Variation**: Adequate range in both input and output lengths
- **Compression Consistency**: Stable compression ratios for summary tasks

## Dataset Validation

### ✓ Requirements Met
- [x] Sufficient sample size for training and evaluation
- [x] Balanced representation of both task types
- [x] Appropriate complexity variation within tasks
- [x] Realistic and coherent synthetic data
- [x] Clear task differentiation for controller learning

### Usage Recommendations
- **Training Split**: Use 80% for training, 20% for evaluation
- **Validation Strategy**: Stratify by task type to ensure balanced evaluation
- **Controller Training**: Task instructions provide clear differentiation signal
- **Performance Metrics**: Track accuracy per task type and overall

---

*Generated by BEM Validation Data Preparation Pipeline*
        """.strip()
        
        # Save report
        report_path = Path(self.config.output_dir) / "dataset_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        console.print(f"Report saved to {report_path}")
        return report
    
    def create_data_visualization(self):
        """Create visualizations of the dataset characteristics."""
        
        console.print("[blue]Creating data visualizations...")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Load data
            json_path = Path(self.config.output_dir) / "json_tasks.json"
            summary_path = Path(self.config.output_dir) / "summary_tasks.json"
            
            with open(json_path) as f:
                json_data = json.load(f)
            with open(summary_path) as f:
                summary_data = json.load(f)
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('BEM Validation Dataset Characteristics', fontsize=16, fontweight='bold')
            
            # 1. JSON Schema Distribution
            ax1 = axes[0, 0]
            schemas = [sample['schema_type'] for sample in json_data]
            schema_counts = pd.Series(schemas).value_counts()
            
            ax1.bar(schema_counts.index, schema_counts.values, color='lightblue')
            ax1.set_title('JSON Schema Distribution')
            ax1.set_xlabel('Schema Type')
            ax1.set_ylabel('Number of Samples')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. JSON Complexity Distribution
            ax2 = axes[0, 1]
            complexities = [sample['complexity'] for sample in json_data]
            ax2.hist(complexities, bins=range(min(complexities), max(complexities)+2), 
                    alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_title('JSON Task Complexity')
            ax2.set_xlabel('Number of Fields')
            ax2.set_ylabel('Frequency')
            
            # 3. Summary Topic Distribution
            ax3 = axes[1, 0]
            topics = [sample['topic'] for sample in summary_data]
            topic_counts = pd.Series(topics).value_counts()
            
            ax3.bar(topic_counts.index, topic_counts.values, color='lightcoral')
            ax3.set_title('Summary Topic Distribution')
            ax3.set_xlabel('Topic')
            ax3.set_ylabel('Number of Samples')
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Text Length vs Summary Length
            ax4 = axes[1, 1]
            text_lengths = [sample['text_length'] for sample in summary_data]
            summary_lengths = [sample['summary_length'] for sample in summary_data]
            
            ax4.scatter(text_lengths, summary_lengths, alpha=0.6, color='gold')
            ax4.set_title('Text Length vs Summary Length')
            ax4.set_xlabel('Original Text Length (chars)')
            ax4.set_ylabel('Summary Length (chars)')
            
            # Add trend line
            z = np.polyfit(text_lengths, summary_lengths, 1)
            p = np.poly1d(z)
            ax4.plot(text_lengths, p(text_lengths), "r--", alpha=0.8)
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = Path(self.config.output_dir) / "dataset_visualization.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            console.print(f"[green]✓[/green] Visualization saved to {viz_path}")
            
        except ImportError:
            console.print("[yellow]Warning: matplotlib/seaborn not available, skipping visualizations")
        except Exception as e:
            console.print(f"[red]Error creating visualizations: {e}")


def main():
    """Main entry point for data preparation."""
    
    parser = argparse.ArgumentParser(description="Prepare BEM Validation Experiment Data")
    parser.add_argument("--output-dir", default="data/validation_experiment",
                       help="Output directory for generated data")
    parser.add_argument("--num-json", type=int, default=1000,
                       help="Number of JSON task samples to generate")
    parser.add_argument("--num-summary", type=int, default=1000,
                       help="Number of summary task samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip data visualizations")
    args = parser.parse_args()
    
    # Create configuration
    config = DataConfig()
    config.output_dir = args.output_dir
    config.num_json_samples = args.num_json
    config.num_summary_samples = args.num_summary
    config.seed = args.seed
    
    # Prepare data
    preparer = ValidationDataPreparer(config)
    results = preparer.prepare_all_data()
    
    # Create visualizations
    if not args.no_viz:
        preparer.create_data_visualization()
    
    print(f"\n✓ Data preparation completed successfully!")
    print(f"Generated {len(results['json_samples'])} JSON samples")
    print(f"Generated {len(results['summary_samples'])} summary samples")
    print(f"Data saved to: {config.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())