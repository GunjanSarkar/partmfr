"""
Pattern optimization tools for the Motor Parts Lookup System.
This module provides pattern analysis and optimization for improving search performance.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import statistics
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PatternOptimizer:
    """
    Analyzes and optimizes search patterns for better performance.
    """
    
    def __init__(self):
        """Initialize the pattern optimizer."""
        self.pattern_stats = defaultdict(lambda: {
            'usage_count': 0,
            'success_count': 0,
            'failure_count': 0,
            'avg_response_time': 0.0,
            'last_used': None,
            'pattern_type': 'unknown'
        })
        
        self.pattern_priority = [
            'exact_match',
            'numeric_only',
            'alphanumeric',
            'fuzzy_match',
            'partial_match',
            'wildcard'
        ]
        
        self.optimization_rules = {
            'high_success_rate': 0.8,
            'min_usage_threshold': 10,
            'performance_weight': 0.3,
            'success_weight': 0.7
        }
    
    def analyze_pattern(self, pattern: str) -> Dict[str, Any]:
        """
        Analyze a search pattern and categorize it.
        
        Args:
            pattern: The search pattern to analyze
            
        Returns:
            Dictionary containing pattern analysis
        """
        if not pattern:
            return {'type': 'empty', 'confidence': 0.0}
        
        analysis = {
            'original': pattern,
            'cleaned': pattern.strip(),
            'length': len(pattern),
            'type': self._classify_pattern(pattern),
            'complexity': self._calculate_complexity(pattern),
            'potential_matches': self._estimate_match_potential(pattern)
        }
        
        return analysis
    
    def _classify_pattern(self, pattern: str) -> str:
        """
        Classify a pattern based on its characteristics.
        
        Args:
            pattern: The pattern to classify
            
        Returns:
            Pattern type as string
        """
        if not pattern:
            return 'empty'
        
        # Check for exact alphanumeric match
        if re.match(r'^[A-Za-z0-9\-\.]+$', pattern):
            if re.match(r'^\d+$', pattern):
                return 'numeric_only'
            elif re.match(r'^[A-Za-z]+$', pattern):
                return 'alpha_only'
            else:
                return 'alphanumeric'
        
        # Check for wildcards
        if '*' in pattern or '?' in pattern or '%' in pattern:
            return 'wildcard'
        
        # Check for special characters
        if re.search(r'[^\w\s\-\.]', pattern):
            return 'special_chars'
        
        # Check for multiple words
        if len(pattern.split()) > 1:
            return 'multi_word'
        
        return 'standard'
    
    def _calculate_complexity(self, pattern: str) -> float:
        """
        Calculate the complexity score of a pattern.
        
        Args:
            pattern: The pattern to analyze
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        if not pattern:
            return 0.0
        
        factors = {
            'length': min(len(pattern) / 50, 1.0),  # Longer patterns are more complex
            'special_chars': len(re.findall(r'[^\w\s]', pattern)) / len(pattern),
            'numeric_ratio': len(re.findall(r'\d', pattern)) / len(pattern),
            'alpha_ratio': len(re.findall(r'[a-zA-Z]', pattern)) / len(pattern),
            'whitespace_ratio': len(re.findall(r'\s', pattern)) / len(pattern)
        }
        
        # Weight the factors
        complexity = (
            factors['length'] * 0.2 +
            factors['special_chars'] * 0.3 +
            factors['numeric_ratio'] * 0.2 +
            factors['alpha_ratio'] * 0.2 +
            factors['whitespace_ratio'] * 0.1
        )
        
        return min(complexity, 1.0)
    
    def _estimate_match_potential(self, pattern: str) -> float:
        """
        Estimate the likelihood of finding matches for a pattern.
        
        Args:
            pattern: The pattern to analyze
            
        Returns:
            Match potential score (0.0 to 1.0)
        """
        if not pattern:
            return 0.0
        
        # Factors that increase match potential
        factors = {
            'optimal_length': 1.0 if 3 <= len(pattern) <= 20 else 0.5,
            'has_numbers': 0.8 if re.search(r'\d', pattern) else 0.3,
            'has_letters': 0.8 if re.search(r'[a-zA-Z]', pattern) else 0.3,
            'not_too_specific': 0.8 if len(pattern) < 30 else 0.3,
            'not_too_generic': 0.8 if len(pattern) > 2 else 0.3
        }
        
        # Calculate weighted potential
        potential = statistics.mean(factors.values())
        
        return min(potential, 1.0)
    
    def record_pattern_usage(self, pattern: str, success: bool, response_time: float = 0.0):
        """
        Record usage statistics for a pattern.
        
        Args:
            pattern: The pattern that was used
            success: Whether the pattern search was successful
            response_time: Time taken for the search
        """
        pattern_type = self._classify_pattern(pattern)
        
        stats = self.pattern_stats[pattern]
        stats['usage_count'] += 1
        stats['last_used'] = datetime.now()
        stats['pattern_type'] = pattern_type
        
        if success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1
        
        # Update average response time
        if response_time > 0:
            current_avg = stats['avg_response_time']
            usage_count = stats['usage_count']
            stats['avg_response_time'] = ((current_avg * (usage_count - 1)) + response_time) / usage_count
    
    def record_success(self, pattern_type: str, response_time: float = 0.0):
        """
        Record a successful pattern match.
        
        Args:
            pattern_type: The type of pattern that succeeded
            response_time: Time taken for the match (optional)
        """
        # Record the attempt with success=True
        self.record_attempt(pattern_type, success=True, response_time=response_time)
        
        # Log the success
        logger.debug(f"Recorded successful pattern match: {pattern_type}")
    
    def record_attempt(self, attempt_type: str, success: bool = False, response_time: float = 0.0):
        """
        Record a pattern matching attempt.
        
        Args:
            attempt_type: The type of attempt (e.g., 'exact_match', 'pattern_extraction')
            success: Whether the attempt was successful
            response_time: Time taken for the attempt
        """
        self.pattern_stats[attempt_type]['usage_count'] += 1
        self.pattern_stats[attempt_type]['last_used'] = datetime.now()
        
        if success:
            self.pattern_stats[attempt_type]['success_count'] += 1
        else:
            self.pattern_stats[attempt_type]['failure_count'] += 1
        
        # Update average response time
        if response_time > 0:
            current_avg = self.pattern_stats[attempt_type]['avg_response_time']
            usage_count = self.pattern_stats[attempt_type]['usage_count']
            self.pattern_stats[attempt_type]['avg_response_time'] = (
                (current_avg * (usage_count - 1)) + response_time
            ) / usage_count
        
        logger.debug(f"Recorded {attempt_type} attempt, success={success}, time={response_time:.3f}s")
    
    def get_pattern_recommendations(self, pattern: str) -> List[str]:
        """
        Get recommendations for improving a search pattern.
        
        Args:
            pattern: The pattern to analyze
            
        Returns:
            List of recommendations
        """
        recommendations = []
        analysis = self.analyze_pattern(pattern)
        
        # Check pattern length
        if analysis['length'] < 3:
            recommendations.append("Consider using a longer search term for better results")
        elif analysis['length'] > 30:
            recommendations.append("Consider shortening the search term for broader results")
        
        # Check pattern type
        if analysis['type'] == 'alpha_only':
            recommendations.append("Consider including numbers if searching for part numbers")
        elif analysis['type'] == 'numeric_only':
            recommendations.append("Consider including letters if searching for part numbers")
        
        # Check complexity
        if analysis['complexity'] > 0.8:
            recommendations.append("Simplify the search pattern to improve performance")
        elif analysis['complexity'] < 0.2:
            recommendations.append("Consider adding more specific terms to narrow results")
        
        # Check historical performance
        if pattern in self.pattern_stats:
            stats = self.pattern_stats[pattern]
            success_rate = stats['success_count'] / stats['usage_count'] if stats['usage_count'] > 0 else 0
            
            if success_rate < 0.5:
                recommendations.append("This pattern has low success rate - consider alternatives")
            if stats['avg_response_time'] > 2.0:
                recommendations.append("This pattern tends to be slow - consider simplifying")
        
        return recommendations
    
    def get_optimized_pattern_order(self) -> List[str]:
        """
        Get patterns ordered by optimization score.
        
        Returns:
            List of pattern types ordered by effectiveness
        """
        pattern_scores = {}
        
        for pattern_type in self.pattern_priority:
            # Find all patterns of this type
            type_patterns = [p for p, stats in self.pattern_stats.items() 
                           if stats['pattern_type'] == pattern_type]
            
            if not type_patterns:
                pattern_scores[pattern_type] = 0.0
                continue
            
            # Calculate average performance for this type
            total_success_rate = 0
            total_performance = 0
            count = 0
            
            for pattern in type_patterns:
                stats = self.pattern_stats[pattern]
                if stats['usage_count'] >= self.optimization_rules['min_usage_threshold']:
                    success_rate = stats['success_count'] / stats['usage_count']
                    performance = 1.0 / max(stats['avg_response_time'], 0.1)  # Inverse of response time
                    
                    total_success_rate += success_rate
                    total_performance += performance
                    count += 1
            
            if count > 0:
                avg_success_rate = total_success_rate / count
                avg_performance = total_performance / count
                
                # Weighted score
                score = (avg_success_rate * self.optimization_rules['success_weight'] + 
                        avg_performance * self.optimization_rules['performance_weight'])
                pattern_scores[pattern_type] = score
            else:
                pattern_scores[pattern_type] = 0.0
        
        # Sort by score (descending)
        return sorted(pattern_scores.keys(), key=lambda x: pattern_scores[x], reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pattern statistics.
        
        Returns:
            Dictionary containing pattern statistics
        """
        total_patterns = len(self.pattern_stats)
        if total_patterns == 0:
            return {'total_patterns': 0, 'message': 'No pattern data available'}
        
        # Calculate aggregate statistics
        total_usage = sum(stats['usage_count'] for stats in self.pattern_stats.values())
        total_success = sum(stats['success_count'] for stats in self.pattern_stats.values())
        total_failures = sum(stats['failure_count'] for stats in self.pattern_stats.values())
        
        avg_response_times = [stats['avg_response_time'] for stats in self.pattern_stats.values() 
                             if stats['avg_response_time'] > 0]
        
        # Pattern type distribution
        type_distribution = defaultdict(int)
        for stats in self.pattern_stats.values():
            type_distribution[stats['pattern_type']] += 1
        
        # Top performing patterns
        top_patterns = sorted(
            [(pattern, stats) for pattern, stats in self.pattern_stats.items()],
            key=lambda x: x[1]['success_count'] / max(x[1]['usage_count'], 1),
            reverse=True
        )[:10]
        
        return {
            'total_patterns': total_patterns,
            'total_usage': total_usage,
            'overall_success_rate': total_success / total_usage if total_usage > 0 else 0,
            'overall_failure_rate': total_failures / total_usage if total_usage > 0 else 0,
            'average_response_time': statistics.mean(avg_response_times) if avg_response_times else 0,
            'pattern_type_distribution': dict(type_distribution),
            'top_patterns': [(pattern, {
                'success_rate': stats['success_count'] / max(stats['usage_count'], 1),
                'usage_count': stats['usage_count'],
                'avg_response_time': stats['avg_response_time']
            }) for pattern, stats in top_patterns[:5]],
            'optimized_order': self.get_optimized_pattern_order()
        }
    
    def get_pattern_priority(self) -> List[str]:
        """
        Get the current pattern priority list.
        
        Returns:
            List of pattern types in priority order
        """
        return self.pattern_priority.copy()
    
    def update_pattern_priority(self, new_priority: List[str]):
        """
        Update the pattern priority list.
        
        Args:
            new_priority: New priority list
        """
        self.pattern_priority = new_priority.copy()
        logger.info(f"Updated pattern priority: {self.pattern_priority}")
    
    def clean_old_patterns(self, days_old: int = 30):
        """
        Clean up old pattern statistics.
        
        Args:
            days_old: Remove patterns older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        patterns_to_remove = []
        for pattern, stats in self.pattern_stats.items():
            if stats['last_used'] and stats['last_used'] < cutoff_date:
                patterns_to_remove.append(pattern)
        
        for pattern in patterns_to_remove:
            del self.pattern_stats[pattern]
        
        logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")


# Global instance
pattern_optimizer = PatternOptimizer()
