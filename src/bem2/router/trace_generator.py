"""
Synthetic Trace Generation for Behavior Cloning

Generates synthetic routing traces for training the macro-policy in BC phase:
- Code Expert: triggered by code blocks, API calls, technical terms
- Formal Expert: triggered by mathematical expressions, proofs, logic
- Safety Expert: triggered by harmful content, ethical concerns

Traces include realistic state transitions and expert preferences.
"""

import torch
import torch.nn.functional as F
import json
import random
import re
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

from .macro_policy import MacroAction, MacroPolicyState

logger = logging.getLogger(__name__)


@dataclass
class SyntheticTrace:
    """Single synthetic routing trace for BC training."""
    states: List[Dict]  # Serializable state representations
    actions: List[Dict]  # Ground truth actions
    rewards: List[float]  # Synthetic rewards
    metadata: Dict  # Trace metadata (length, expert usage, etc.)


class ContentClassifier:
    """
    Classifies content chunks to determine optimal expert routing.
    
    Uses rule-based heuristics to simulate human expert preferences.
    """
    
    def __init__(self):
        # Patterns for expert classification
        self.code_patterns = [
            r'def\s+\w+\s*\(',  # Function definitions
            r'class\s+\w+\s*:',  # Class definitions
            r'import\s+\w+',     # Import statements
            r'from\s+\w+\s+import',  # From imports
            r'\w+\.\w+\(',       # Method calls
            r'if\s+__name__\s*==\s*["\']__main__["\']',  # Main blocks
            r'try:\s*\n.*?except',  # Exception handling
            r'for\s+\w+\s+in\s+',   # For loops
            r'while\s+.*?:',        # While loops
            r'return\s+\w+',        # Return statements
            r'print\s*\(',          # Print statements
            r'#.*?coding[:=]',      # Encoding declarations
        ]
        
        self.formal_patterns = [
            r'\$.*?\$',              # LaTeX math
            r'\\begin\{.*?\}',       # LaTeX environments  
            r'\\end\{.*?\}',
            r'theorem|lemma|proof|corollary',  # Mathematical terms
            r'\\forall|\\exists|\\in|\\subset',  # Logic symbols
            r'\\sum_|\\prod_|\\int_',  # Mathematical operators
            r'QED|∎|□',               # Proof endings
            r'assume|suppose|let|given',  # Proof language
            r'therefore|thus|hence|follows',  # Logical connectives
            r'\d+\.\d+\s+(theorem|lemma)',  # Numbered theorems
            r'axiom|postulate|definition',  # Formal definitions
        ]
        
        self.safety_patterns = [
            r'harm|hurt|damage|injure',    # Harm keywords
            r'illegal|unlawful|criminal',   # Legal concerns  
            r'violence|violent|attack',     # Violence
            r'discriminat|racist|sexist',   # Discrimination
            r'private|confidential|secret', # Privacy
            r'manipulat|deceiv|trick',      # Manipulation
            r'dangerous|unsafe|hazardous',  # Safety
            r'suicide|self.?harm',          # Self-harm
            r'exploit|vulnerabilit',        # Security
            r'bias|unfair|prejudic',        # Fairness
        ]
        
        # Compile patterns
        self.code_regex = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.code_patterns]
        self.formal_regex = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.formal_patterns]  
        self.safety_regex = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.safety_patterns]
    
    def classify_chunk(self, text: str) -> Dict[str, float]:
        """
        Classify text chunk and return expert preference scores.
        
        Args:
            text: Text content to classify
            
        Returns:
            Dictionary mapping expert names to preference scores [0, 1]
        """
        # Count pattern matches
        code_matches = sum(1 for regex in self.code_regex if regex.search(text))
        formal_matches = sum(1 for regex in self.formal_regex if regex.search(text))
        safety_matches = sum(1 for regex in self.safety_regex if regex.search(text))
        
        # Normalize by text length and pattern count
        text_len = max(len(text), 100)  # Avoid division by zero
        code_score = min(code_matches / (text_len / 100), 1.0)
        formal_score = min(formal_matches / (text_len / 100), 1.0)
        safety_score = min(safety_matches / (text_len / 100), 1.0)
        
        # Add some noise and ensure sum doesn't exceed reasonable bounds
        code_score += random.gauss(0, 0.1)
        formal_score += random.gauss(0, 0.1) 
        safety_score += random.gauss(0, 0.1)
        
        # Clamp to [0, 1]
        code_score = max(0, min(1, code_score))
        formal_score = max(0, min(1, formal_score))
        safety_score = max(0, min(1, safety_score))
        
        return {
            'Code': code_score,
            'Formal': formal_score, 
            'Safety': safety_score
        }


class StateGenerator:
    """Generates realistic state representations for synthetic traces."""
    
    def __init__(
        self,
        chunk_summary_dim: int = 512,
        retrieval_dim: int = 64,
        vision_dim: int = 768,
        value_dim: int = 32
    ):
        self.chunk_summary_dim = chunk_summary_dim
        self.retrieval_dim = retrieval_dim
        self.vision_dim = vision_dim
        self.value_dim = value_dim
    
    def generate_state(
        self,
        text: str,
        chunk_index: int,
        prev_action: Optional[MacroAction] = None,
        content_scores: Optional[Dict[str, float]] = None
    ) -> MacroPolicyState:
        """
        Generate state representation for a text chunk.
        
        Args:
            text: Text content
            chunk_index: Position in sequence
            prev_action: Previous macro-action
            content_scores: Expert preference scores
            
        Returns:
            MacroPolicyState for this chunk
        """
        device = torch.device('cpu')  # Generate on CPU, move to GPU later
        
        # Generate chunk summary (simulate text encoder)
        # Use content scores to bias the representation
        if content_scores is None:
            content_scores = {'Code': 0.33, 'Formal': 0.33, 'Safety': 0.33}
        
        # Create base representation with some structure
        chunk_summary = torch.randn(1, self.chunk_summary_dim, device=device)
        
        # Bias based on content type
        code_bias = content_scores['Code'] * torch.randn(1, self.chunk_summary_dim // 3, device=device) 
        formal_bias = content_scores['Formal'] * torch.randn(1, self.chunk_summary_dim // 3, device=device)
        safety_bias = content_scores['Safety'] * torch.randn(1, self.chunk_summary_dim // 3, device=device)
        
        # Combine biases into full representation
        biased_summary = chunk_summary.clone()
        biased_summary[0, :self.chunk_summary_dim // 3] += code_bias[0]
        biased_summary[0, self.chunk_summary_dim // 3:2 * self.chunk_summary_dim // 3] += formal_bias[0]
        biased_summary[0, 2 * self.chunk_summary_dim // 3:] += safety_bias[0]
        
        # Generate retrieval features (simulate retrieval quality)
        retrieval_features = torch.randn(1, self.retrieval_dim, device=device)
        # Make retrieval quality correlate with content complexity
        complexity = len(text) / 1000  # Simple complexity measure
        retrieval_features *= (1 + complexity * 0.5)
        
        # Generate vision features (simulate visual embeddings)
        vision_features = torch.randn(1, self.vision_dim, device=device) * 0.1  # Mostly zeros
        
        # Generate value features (safety/constitutional scores)
        value_features = torch.randn(1, self.value_dim, device=device)
        # Safety content should have distinctive value signature
        if content_scores['Safety'] > 0.3:
            value_features *= (1 + content_scores['Safety'] * 2)
        
        # Previous action encoding
        prev_action_tensor = None
        if prev_action is not None:
            prev_action_tensor = prev_action.to_tensor(device).unsqueeze(0)
        
        # Chunk index
        chunk_index_tensor = torch.tensor([chunk_index], device=device, dtype=torch.float32)
        
        return MacroPolicyState(
            chunk_summary=biased_summary,
            retrieval_features=retrieval_features,
            vision_features=vision_features,
            value_features=value_features,
            prev_action=prev_action_tensor,
            chunk_index=chunk_index_tensor
        )


class ExpertPreferenceModel:
    """Models expert preferences for different content types."""
    
    def __init__(self):
        # Expert behavior profiles
        self.expert_profiles = {
            'Code': {
                'preferred_scope': 'local',  # Code usually local context
                'preferred_span': [1, 2],    # Short spans
                'preferred_rank': [16, 32],  # Medium rank
                'preferred_bias': [0.5, 1.0], # Moderate bias
            },
            'Formal': {
                'preferred_scope': 'global', # Math needs global context  
                'preferred_span': [2, 3, 4], # Longer spans
                'preferred_rank': [32, 64],   # High rank for complexity
                'preferred_bias': [0.8, 1.5], # Higher bias
            },
            'Safety': {
                'preferred_scope': 'global', # Safety needs full context
                'preferred_span': [1, 2],    # Quick reactions
                'preferred_rank': [8, 16],   # Lower rank, interpretability
                'preferred_bias': [1.2, 2.0], # Strong bias
            }
        }
    
    def get_optimal_action(
        self,
        content_scores: Dict[str, float],
        chunk_index: int,
        prev_action: Optional[MacroAction] = None
    ) -> MacroAction:
        """
        Generate optimal action based on content and context.
        
        Args:
            content_scores: Expert preference scores
            chunk_index: Current position
            prev_action: Previous action for continuity
            
        Returns:
            Optimal MacroAction
        """
        # Select expert with highest score
        expert_names = ['Code', 'Formal', 'Safety']
        expert_scores = [content_scores[name] for name in expert_names]
        best_expert_idx = expert_scores.index(max(expert_scores))
        best_expert_name = expert_names[best_expert_idx]
        
        # Get profile for best expert
        profile = self.expert_profiles[best_expert_name]
        
        # Sample action parameters from preferences
        scope = profile['preferred_scope']
        span = random.choice(profile['preferred_span'])
        rank_budget = random.choice(profile['preferred_rank'])
        bias_scale = random.uniform(*profile['preferred_bias'])
        
        # Add some continuity with previous action
        if prev_action is not None and random.random() < 0.3:  # 30% chance of continuity
            span = min(span, prev_action.span)  # Don't increase span too much
            if abs(bias_scale - prev_action.bias_scale) > 0.5:
                bias_scale = prev_action.bias_scale + random.uniform(-0.2, 0.2)
        
        return MacroAction(
            expert_id=best_expert_idx,
            scope=scope,
            span=span,
            rank_budget=rank_budget,
            bias_scale=bias_scale
        )


class TraceGenerator:
    """Main trace generator for synthetic BC data."""
    
    def __init__(
        self,
        chunk_size: int = 128,
        max_chunks_per_trace: int = 20,
        seed: int = 42
    ):
        self.chunk_size = chunk_size
        self.max_chunks_per_trace = max_chunks_per_trace
        
        # Set seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize components
        self.classifier = ContentClassifier()
        self.state_generator = StateGenerator()
        self.preference_model = ExpertPreferenceModel()
        
        # Sample texts for different domains
        self.sample_texts = self._load_sample_texts()
    
    def _load_sample_texts(self) -> Dict[str, List[str]]:
        """Load or generate sample texts for different domains."""
        return {
            'Code': [
                '''
def fibonacci(n):
    """Compute the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []
    
    def process(self, input_data):
        try:
            result = self._transform(input_data)
            return result
        except Exception as e:
            print(f"Error processing data: {e}")
            return None
''',
                '''
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Load and preprocess the dataset
def load_data(file_path):
    data = np.load(file_path)
    X, y = data['features'], data['labels']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)
''',
            ],
            'Formal': [
                '''
Theorem 1.1 (Fundamental Theorem of Calculus). Let $f$ be a continuous function on $[a,b]$. Then the function $F(x) = \\int_a^x f(t) dt$ is differentiable on $(a,b)$ and $F'(x) = f(x)$.

Proof: We need to show that $\\lim_{h \\to 0} \\frac{F(x+h) - F(x)}{h} = f(x)$.

By definition, we have:
$$F(x+h) - F(x) = \\int_a^{x+h} f(t) dt - \\int_a^x f(t) dt = \\int_x^{x+h} f(t) dt$$

Since $f$ is continuous, by the Mean Value Theorem for integrals, there exists $c \\in [x, x+h]$ such that:
$$\\int_x^{x+h} f(t) dt = f(c) \\cdot h$$

Therefore:
$$\\frac{F(x+h) - F(x)}{h} = f(c)$$

As $h \\to 0$, we have $c \\to x$, and by continuity of $f$, $f(c) \\to f(x)$. Hence $F'(x) = f(x)$. QED
''',
                '''
Definition 2.3. A topological space $(X, \\tau)$ is called compact if every open cover of $X$ has a finite subcover.

Lemma 2.4. Every compact subset of a metric space is closed and bounded.

Proof: Let $(X,d)$ be a metric space and $K \\subset X$ be compact.

First, we show $K$ is bounded. Fix $x_0 \\in K$. For each $n \\in \\mathbb{N}$, let $U_n = B(x_0, n)$ be the open ball of radius $n$ centered at $x_0$. Then $\\{U_n\\}_{n=1}^{\\infty}$ is an open cover of $K$. 

Since $K$ is compact, there exists a finite subcover, say $\\{U_{n_1}, \\ldots, U_{n_k}\\}$. Let $N = \\max\\{n_1, \\ldots, n_k\\}$. Then $K \\subset U_N = B(x_0, N)$, so $K$ is bounded.

Next, we show $K$ is closed by showing $K^c$ is open. Let $x \\in K^c$. For each $y \\in K$, since $x \\neq y$, we can choose $r_y > 0$ such that $B(x, r_y) \\cap B(y, r_y) = \\emptyset$.
''',
            ],
            'Safety': [
                '''
I understand you're looking for information, but I need to be careful about providing details that could potentially be harmful or misused. Let me explain why certain types of information require careful handling.

When discussing topics related to security vulnerabilities, it's important to consider the potential for malicious use. While legitimate security researchers need access to vulnerability information for defensive purposes, the same information could be exploited by bad actors.

Instead of providing specific exploitation techniques, I can offer general guidance on:
- Responsible disclosure practices
- Defensive security measures  
- Resources for legitimate security research
- How to report vulnerabilities you discover

This approach helps maintain the balance between enabling legitimate security work while avoiding potential harm from malicious use of the information.
''',
                '''
I notice this request involves content that could potentially promote discriminatory views or harmful stereotypes. It's important to approach such topics with care and consideration for how our words might impact different communities.

Rather than reinforcing potentially harmful narratives, I'd like to suggest we focus on:
- Factual, evidence-based information
- Perspectives that promote understanding and inclusion
- Resources from reputable organizations working on these issues
- Ways to have constructive conversations about complex social topics

If you're researching this topic for academic or educational purposes, I can help you find scholarly sources and frameworks that approach these issues with appropriate nuance and respect for human dignity.
''',
            ]
        }
    
    def generate_trace(
        self,
        domain: str = None,
        length: int = None
    ) -> SyntheticTrace:
        """
        Generate a single synthetic trace.
        
        Args:
            domain: Preferred domain (Code, Formal, Safety) or None for mixed
            length: Number of chunks, or None for random length
            
        Returns:
            SyntheticTrace with states, actions, and rewards
        """
        if length is None:
            length = random.randint(3, self.max_chunks_per_trace)
        
        # Select text samples
        if domain is None:
            # Mixed domain trace
            domains = ['Code', 'Formal', 'Safety']
            selected_texts = []
            for _ in range(length):
                domain_choice = random.choice(domains)
                text_choice = random.choice(self.sample_texts[domain_choice])
                selected_texts.append((domain_choice, text_choice))
        else:
            # Single domain trace
            selected_texts = [(domain, random.choice(self.sample_texts[domain]))] * length
        
        # Generate states and actions
        states = []
        actions = []
        rewards = []
        prev_action = None
        
        for chunk_idx, (chunk_domain, text) in enumerate(selected_texts):
            # Classify content
            content_scores = self.classifier.classify_chunk(text)
            
            # Boost the score for intended domain
            content_scores[chunk_domain] = min(1.0, content_scores[chunk_domain] + 0.3)
            
            # Generate state
            state = self.state_generator.generate_state(
                text=text,
                chunk_index=chunk_idx,
                prev_action=prev_action,
                content_scores=content_scores
            )
            
            # Generate optimal action
            action = self.preference_model.get_optimal_action(
                content_scores=content_scores,
                chunk_index=chunk_idx,
                prev_action=prev_action
            )
            
            # Generate reward (higher for better expert matches)
            reward = content_scores[['Code', 'Formal', 'Safety'][action.expert_id]]
            reward += random.gauss(0, 0.1)  # Add noise
            reward = max(0, min(1, reward))  # Clamp
            
            # Store serializable representations
            states.append({
                'chunk_summary': state.chunk_summary.tolist(),
                'retrieval_features': state.retrieval_features.tolist(),
                'vision_features': state.vision_features.tolist(),
                'value_features': state.value_features.tolist(),
                'prev_action': state.prev_action.tolist() if state.prev_action is not None else None,
                'chunk_index': state.chunk_index.tolist(),
                'content_scores': content_scores
            })
            
            actions.append(asdict(action))
            rewards.append(reward)
            prev_action = action
        
        # Generate metadata
        expert_usage = [0, 0, 0]  # Code, Formal, Safety
        for action_dict in actions:
            expert_usage[action_dict['expert_id']] += 1
        
        metadata = {
            'length': length,
            'domain': domain,
            'expert_usage': expert_usage,
            'avg_reward': sum(rewards) / len(rewards),
            'total_reward': sum(rewards)
        }
        
        return SyntheticTrace(
            states=states,
            actions=actions,
            rewards=rewards,
            metadata=metadata
        )
    
    def generate_dataset(
        self,
        num_traces: int = 1000,
        domain_distribution: Optional[Dict[str, float]] = None,
        output_path: Optional[str] = None
    ) -> List[SyntheticTrace]:
        """
        Generate a full dataset of synthetic traces.
        
        Args:
            num_traces: Number of traces to generate
            domain_distribution: Distribution over domains
            output_path: Path to save traces (JSON format)
            
        Returns:
            List of synthetic traces
        """
        if domain_distribution is None:
            domain_distribution = {'Code': 0.4, 'Formal': 0.3, 'Safety': 0.2, 'Mixed': 0.1}
        
        traces = []
        
        # Generate traces according to distribution
        domains = list(domain_distribution.keys())
        weights = list(domain_distribution.values())
        
        for i in range(num_traces):
            if i % 100 == 0:
                logger.info(f"Generating trace {i}/{num_traces}")
            
            # Select domain
            domain = random.choices(domains, weights=weights)[0]
            if domain == 'Mixed':
                domain = None
            
            # Generate trace
            trace = self.generate_trace(domain=domain)
            traces.append(trace)
        
        # Save if requested
        if output_path is not None:
            self._save_traces(traces, output_path)
        
        logger.info(f"Generated {num_traces} synthetic traces")
        return traces
    
    def _save_traces(self, traces: List[SyntheticTrace], output_path: str):
        """Save traces to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert traces to serializable format
        serializable_traces = []
        for trace in traces:
            serializable_traces.append({
                'states': trace.states,
                'actions': trace.actions,
                'rewards': trace.rewards,
                'metadata': trace.metadata
            })
        
        with open(output_file, 'w') as f:
            json.dump(serializable_traces, f, indent=2)
        
        logger.info(f"Saved {len(traces)} traces to {output_path}")
    
    @staticmethod
    def load_traces(input_path: str) -> List[SyntheticTrace]:
        """Load traces from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        traces = []
        for trace_data in data:
            trace = SyntheticTrace(
                states=trace_data['states'],
                actions=[MacroAction(**action) for action in trace_data['actions']],
                rewards=trace_data['rewards'],
                metadata=trace_data['metadata']
            )
            traces.append(trace)
        
        return traces


def create_trace_generator(config: Dict) -> TraceGenerator:
    """
    Factory function to create TraceGenerator from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TraceGenerator instance
    """
    return TraceGenerator(
        chunk_size=config.get('chunk_size', 128),
        max_chunks_per_trace=config.get('max_chunks_per_trace', 20),
        seed=config.get('seed', 42)
    )