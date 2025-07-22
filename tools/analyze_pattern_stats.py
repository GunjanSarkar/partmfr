"""
Pattern analysis statistics for the Motor Parts Lookup System.
This module provides detailed analysis and statistics for search patterns and performance.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import statistics
import json
from datetime import datetime, timedelta
# Optional imports - only used if available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """
    Analyzes search patterns and provides detailed statistics.
    """
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.pattern_data = defaultdict(lambda: {
            'frequency': 0,
            'success_rate': 0.0,
            'avg_response_time': 0.0,
            'result_counts': [],
            'timestamps': [],
            'user_sessions': set(),
            'error_types': Counter(),
            'pattern_variants': set()
        })
        
        self.analysis_cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def record_pattern_usage(self, pattern: str, success: bool, response_time: float, 
                           result_count: int = 0, user_session: str = None, 
                           error_type: str = None):
        """
        Record detailed pattern usage statistics.
        
        Args:
            pattern: The search pattern used
            success: Whether the search was successful
            response_time: Time taken for the search
            result_count: Number of results returned
            user_session: User session identifier
            error_type: Type of error if search failed
        """
        data = self.pattern_data[pattern]
        
        # Update frequency
        data['frequency'] += 1
        
        # Update success rate
        current_success_count = data['success_rate'] * (data['frequency'] - 1)
        if success:
            current_success_count += 1
        data['success_rate'] = current_success_count / data['frequency']
        
        # Update average response time
        current_total_time = data['avg_response_time'] * (data['frequency'] - 1)
        data['avg_response_time'] = (current_total_time + response_time) / data['frequency']
        
        # Record result count
        data['result_counts'].append(result_count)
        
        # Record timestamp
        data['timestamps'].append(datetime.now())
        
        # Record user session
        if user_session:
            data['user_sessions'].add(user_session)
        
        # Record error type
        if error_type:
            data['error_types'][error_type] += 1
        
        # Record pattern variants
        normalized_pattern = self._normalize_pattern(pattern)
        data['pattern_variants'].add(normalized_pattern)
    
    def get_pattern_statistics(self, pattern: str = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a pattern or all patterns.
        
        Args:
            pattern: Specific pattern to analyze (None for all patterns)
            
        Returns:
            Dictionary containing pattern statistics
        """
        if pattern:
            return self._get_single_pattern_stats(pattern)
        else:
            return self._get_all_patterns_stats()
    
    def _get_single_pattern_stats(self, pattern: str) -> Dict[str, Any]:
        """
        Get statistics for a single pattern.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Dictionary containing pattern statistics
        """
        if pattern not in self.pattern_data:
            return {'error': 'Pattern not found'}
        
        data = self.pattern_data[pattern]
        
        # Calculate additional metrics
        result_counts = data['result_counts']
        timestamps = data['timestamps']
        
        stats = {
            'pattern': pattern,
            'frequency': data['frequency'],
            'success_rate': data['success_rate'],
            'avg_response_time': data['avg_response_time'],
            'unique_users': len(data['user_sessions']),
            'error_distribution': dict(data['error_types']),
            'pattern_variants': list(data['pattern_variants'])
        }
        
        if result_counts:
            stats.update({
                'avg_result_count': statistics.mean(result_counts),
                'median_result_count': statistics.median(result_counts),
                'max_result_count': max(result_counts),
                'min_result_count': min(result_counts)
            })
        
        if timestamps:
            stats.update({
                'first_used': min(timestamps),
                'last_used': max(timestamps),
                'usage_timespan': max(timestamps) - min(timestamps)
            })
            
            # Calculate usage frequency over time
            stats['usage_frequency'] = self._calculate_usage_frequency(timestamps)
        
        return stats
    
    def _get_all_patterns_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics for all patterns.
        
        Returns:
            Dictionary containing aggregate statistics
        """
        if not self.pattern_data:
            return {'total_patterns': 0, 'message': 'No pattern data available'}
        
        total_patterns = len(self.pattern_data)
        total_frequency = sum(data['frequency'] for data in self.pattern_data.values())
        
        # Calculate weighted averages
        weighted_success_rate = sum(
            data['success_rate'] * data['frequency'] for data in self.pattern_data.values()
        ) / total_frequency if total_frequency > 0 else 0
        
        weighted_response_time = sum(
            data['avg_response_time'] * data['frequency'] for data in self.pattern_data.values()
        ) / total_frequency if total_frequency > 0 else 0
        
        # Pattern type distribution
        pattern_types = Counter()
        for pattern in self.pattern_data.keys():
            pattern_type = self._classify_pattern_type(pattern)
            pattern_types[pattern_type] += 1
        
        # Top patterns by frequency
        top_patterns = sorted(
            self.pattern_data.items(),
            key=lambda x: x[1]['frequency'],
            reverse=True
        )[:10]
        
        # Performance metrics
        performance_stats = self._calculate_performance_metrics()
        
        return {
            'total_patterns': total_patterns,
            'total_usage_frequency': total_frequency,
            'overall_success_rate': weighted_success_rate,
            'overall_avg_response_time': weighted_response_time,
            'pattern_type_distribution': dict(pattern_types),
            'top_patterns_by_frequency': [
                (pattern, data['frequency']) for pattern, data in top_patterns
            ],
            'performance_metrics': performance_stats,
            'analysis_timestamp': datetime.now()
        }
    
    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze trends over the specified time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary containing trend analysis
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        trends = {
            'period_days': days,
            'daily_usage': defaultdict(int),
            'daily_success_rate': defaultdict(list),
            'pattern_popularity_changes': {},
            'performance_trends': {},
            'new_patterns': []
        }
        
        for pattern, data in self.pattern_data.items():
            timestamps = data['timestamps']
            recent_timestamps = [ts for ts in timestamps if ts >= cutoff_date]
            
            if not recent_timestamps:
                continue
            
            # Daily usage
            for ts in recent_timestamps:
                day_key = ts.date().isoformat()
                trends['daily_usage'][day_key] += 1
            
            # Track new patterns
            if min(timestamps) >= cutoff_date:
                trends['new_patterns'].append(pattern)
        
        # Convert defaultdicts to regular dicts
        trends['daily_usage'] = dict(trends['daily_usage'])
        
        return trends
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on pattern analysis.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        if not self.pattern_data:
            return [{'type': 'info', 'message': 'No pattern data available for analysis'}]
        
        # Analyze success rates
        low_success_patterns = [
            (pattern, data['success_rate']) 
            for pattern, data in self.pattern_data.items()
            if data['success_rate'] < 0.5 and data['frequency'] > 5
        ]
        
        if low_success_patterns:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'Low Success Rate Patterns',
                'message': f'Found {len(low_success_patterns)} patterns with low success rates',
                'patterns': low_success_patterns[:5],
                'action': 'Consider optimizing these search patterns'
            })
        
        # Analyze response times
        slow_patterns = [
            (pattern, data['avg_response_time'])
            for pattern, data in self.pattern_data.items()
            if data['avg_response_time'] > 2.0 and data['frequency'] > 5
        ]
        
        if slow_patterns:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'title': 'Slow Response Patterns',
                'message': f'Found {len(slow_patterns)} patterns with slow response times',
                'patterns': slow_patterns[:5],
                'action': 'Consider adding indexing or caching for these patterns'
            })
        
        # Analyze pattern complexity
        complex_patterns = [
            pattern for pattern in self.pattern_data.keys()
            if len(pattern) > 50 or len(re.findall(r'[^\w\s]', pattern)) > 10
        ]
        
        if complex_patterns:
            recommendations.append({
                'type': 'usability',
                'priority': 'low',
                'title': 'Complex Patterns',
                'message': f'Found {len(complex_patterns)} overly complex patterns',
                'patterns': complex_patterns[:5],
                'action': 'Consider simplifying these patterns for better user experience'
            })
        
        return recommendations
    
    def export_statistics(self, format: str = 'json', filename: str = None) -> str:
        """
        Export pattern statistics to file.
        
        Args:
            format: Export format ('json', 'csv', 'xlsx')
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"pattern_stats_{timestamp}.{format}"
        
        stats = self._get_all_patterns_stats()
        
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        elif format == 'csv':
            self._export_to_csv(filename)
        elif format == 'xlsx':
            self._export_to_xlsx(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filename
    
    def _normalize_pattern(self, pattern: str) -> str:
        """
        Normalize a pattern for variant analysis.
        
        Args:
            pattern: Pattern to normalize
            
        Returns:
            Normalized pattern
        """
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', pattern.lower().strip())
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    def _classify_pattern_type(self, pattern: str) -> str:
        """
        Classify a pattern by type.
        
        Args:
            pattern: Pattern to classify
            
        Returns:
            Pattern type
        """
        if re.match(r'^\d+$', pattern):
            return 'numeric_only'
        elif re.match(r'^[A-Za-z]+$', pattern):
            return 'alpha_only'
        elif re.match(r'^[A-Za-z0-9\-\.]+$', pattern):
            return 'alphanumeric'
        elif len(pattern.split()) > 1:
            return 'multi_word'
        elif '*' in pattern or '?' in pattern:
            return 'wildcard'
        else:
            return 'mixed'
    
    def _calculate_usage_frequency(self, timestamps: List[datetime]) -> Dict[str, int]:
        """
        Calculate usage frequency by time period.
        
        Args:
            timestamps: List of usage timestamps
            
        Returns:
            Dictionary with frequency by time period
        """
        frequency = {
            'hourly': defaultdict(int),
            'daily': defaultdict(int),
            'weekly': defaultdict(int)
        }
        
        for ts in timestamps:
            frequency['hourly'][ts.hour] += 1
            frequency['daily'][ts.date().isoformat()] += 1
            # ISO week number
            year, week, _ = ts.isocalendar()
            frequency['weekly'][f"{year}-W{week:02d}"] += 1
        
        return {k: dict(v) for k, v in frequency.items()}
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate overall performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.pattern_data:
            return {}
        
        all_response_times = []
        all_result_counts = []
        
        for data in self.pattern_data.values():
            if data['avg_response_time'] > 0:
                all_response_times.append(data['avg_response_time'])
            all_result_counts.extend(data['result_counts'])
        
        metrics = {}
        
        if all_response_times:
            metrics['response_time'] = {
                'mean': statistics.mean(all_response_times),
                'median': statistics.median(all_response_times),
                'std_dev': statistics.stdev(all_response_times) if len(all_response_times) > 1 else 0,
                'percentile_95': sorted(all_response_times)[int(len(all_response_times) * 0.95)] if all_response_times else 0
            }
        
        if all_result_counts:
            metrics['result_counts'] = {
                'mean': statistics.mean(all_result_counts),
                'median': statistics.median(all_result_counts),
                'std_dev': statistics.stdev(all_result_counts) if len(all_result_counts) > 1 else 0,
                'zero_results_rate': all_result_counts.count(0) / len(all_result_counts)
            }
        
        return metrics
    
    def _export_to_csv(self, filename: str):
        """Export statistics to CSV format."""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Pattern', 'Frequency', 'Success Rate', 'Avg Response Time',
                'Unique Users', 'Avg Result Count', 'Pattern Type'
            ])
            
            # Write data
            for pattern, data in self.pattern_data.items():
                avg_results = statistics.mean(data['result_counts']) if data['result_counts'] else 0
                pattern_type = self._classify_pattern_type(pattern)
                
                writer.writerow([
                    pattern,
                    data['frequency'],
                    data['success_rate'],
                    data['avg_response_time'],
                    len(data['user_sessions']),
                    avg_results,
                    pattern_type
                ])
    
    def _export_to_xlsx(self, filename: str):
        """Export statistics to Excel format."""
        if not HAS_PANDAS:
            logger.warning("pandas not available, falling back to CSV export")
            csv_filename = filename.replace('.xlsx', '.csv')
            self._export_to_csv(csv_filename)
            return
        
        try:
            import pandas as pd
            
            # Prepare data
            rows = []
            for pattern, data in self.pattern_data.items():
                avg_results = statistics.mean(data['result_counts']) if data['result_counts'] else 0
                pattern_type = self._classify_pattern_type(pattern)
                
                rows.append({
                    'Pattern': pattern,
                    'Frequency': data['frequency'],
                    'Success Rate': data['success_rate'],
                    'Avg Response Time': data['avg_response_time'],
                    'Unique Users': len(data['user_sessions']),
                    'Avg Result Count': avg_results,
                    'Pattern Type': pattern_type
                })
            
            df = pd.DataFrame(rows)
            df.to_excel(filename, index=False)
            
        except ImportError:
            logger.warning("pandas not available, falling back to CSV export")
            csv_filename = filename.replace('.xlsx', '.csv')
            self._export_to_csv(csv_filename)


# Global instance
pattern_analyzer = PatternAnalyzer()
