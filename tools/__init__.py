"""
Tools module for Motor Part Number Processing System.

This module contains utilities for:
- Part number scoring and similarity calculation
- Pattern analysis and optimization
- Basic utilities for data processing
- Performance tracking and statistics
"""

from .scoring_system import PartNumberScoringSystem
from .basic_tools import PartNumberCleaner, DescriptionProcessor, PerformanceTracker, ConfigurationManager
from .pattern_optimizer import PatternOptimizer
from .similarity_utils import SimilarityCalculator
from .analyze_pattern_stats import PatternAnalyzer

__all__ = [
    'PartNumberScoringSystem',
    'PartNumberCleaner',
    'DescriptionProcessor', 
    'PerformanceTracker',
    'ConfigurationManager',
    'PatternOptimizer',
    'SimilarityCalculator',
    'PatternAnalyzer'
]

__version__ = '1.0.0'
