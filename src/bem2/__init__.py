"""
BEM 2.0 - Block-wise Expert Modules Second Generation

MIT License

Copyright (c) 2024 Nathan Rice and BEM Research Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

BEM 2.0 - Block-wise Expert Modules Second Generation

Implements the next generation of BEM with advanced features:
- Agentic Router: Dynamic composition with macro-policy
- Online Learning: Safe controller-only updates
- Multimodal Conditioning: Vision-aware routing
- Value-Aligned Safety: Orthogonal safety basis
- Performance Track: Head-group gating, dynamic rank, etc.

All under strict cache-safety, budget parity, and CI-first evaluation.
"""

__version__ = "2.0.0"

from . import router
from . import online
from . import multimodal 
from . import safety
from . import perftrack

__all__ = [
    "router",
    "online", 
    "multimodal",
    "safety",
    "perftrack"
]