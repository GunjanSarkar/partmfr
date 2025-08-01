"""
Basic tools and utilities for the Motor Parts Lookup System.
This module provides common utilities used across the application.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import json
import os
from difflib import SequenceMatcher
from config.settings import settings

logger = logging.getLogger(__name__)

class PartNumberCleaner:
    """
    Utility class for cleaning and normalizing part numbers.
    """
    
    def __init__(self):
        """Initialize the part number cleaner."""
        self.common_separators = ['-', '_', '.', '/', '\\', '|', ':', ';', ',', ' ']
        self.noise_patterns = [
            r'[^\w\-\.]',  # Remove special characters except word chars, hyphens, dots
            r'\s+',        # Replace multiple spaces with single space
        ]
    
    def clean_part_number(self, part_number: str) -> str:
        """
        Clean and normalize a part number.
        
        Args:
            part_number: The part number to clean
            
        Returns:
            Cleaned part number
        """
        if not part_number:
            return ""
        
        # Convert to string and strip whitespace
        cleaned = str(part_number).strip()
        
        # Remove common prefixes/suffixes that don't contribute to matching
        prefixes_to_remove = ['PT', 'PART', 'P/N', 'PN', 'OEM', 'MFG']
        for prefix in prefixes_to_remove:
            if cleaned.upper().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Apply noise reduction patterns
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, ' ', cleaned)
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        return cleaned
    
    def extract_numeric_part(self, part_number: str) -> str:
        """
        Extract numeric portion from part number.
        
        Args:
            part_number: The part number to process
            
        Returns:
            Numeric portion of the part number
        """
        if not part_number:
            return ""
        
        # Find all numeric sequences
        numeric_parts = re.findall(r'\d+', part_number)
        
        if not numeric_parts:
            return ""
        
        # Return the longest numeric sequence
        return max(numeric_parts, key=len)
    
    def extract_alpha_part(self, part_number: str) -> str:
        """
        Extract alphabetic portion from part number.
        
        Args:
            part_number: The part number to process
            
        Returns:
            Alphabetic portion of the part number
        """
        if not part_number:
            return ""
        
        # Find all alphabetic sequences
        alpha_parts = re.findall(r'[A-Za-z]+', part_number)
        
        if not alpha_parts:
            return ""
        
        # Return the longest alphabetic sequence
        return max(alpha_parts, key=len)


class DescriptionProcessor:
    """
    Utility class for processing and normalizing part descriptions.
    """
    
    def __init__(self):
        """Initialize the description processor."""
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'do', 'their', 'time', 'if',
            'up', 'out', 'so', 'can', 'her', 'than', 'call', 'its', 'now', 'find',
            'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
        }
        
        self.technical_terms = {
            'assembly', 'assy', 'component', 'part', 'unit', 'module', 'system',
            'motor', 'engine', 'pump', 'valve', 'switch', 'sensor', 'filter',
            'bearing', 'seal', 'gasket', 'bolt', 'screw', 'nut', 'washer',
            'spring', 'plate', 'cover', 'housing', 'bracket', 'mount', 'clamp'
        }
    
    def clean_description(self, description: str) -> str:
        """
        Clean and normalize a description.
        
        Args:
            description: The description to clean
            
        Returns:
            Cleaned description
        """
        if not description:
            return ""
        
        # Convert to lowercase and strip
        cleaned = description.lower().strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep spaces
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        
        # Remove extra spaces again
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def extract_keywords(self, description: str, include_stop_words: bool = False) -> List[str]:
        """
        Extract keywords from description.
        
        Args:
            description: The description to process
            include_stop_words: Whether to include stop words
            
        Returns:
            List of keywords
        """
        if not description:
            return []
        
        # Clean the description
        cleaned = self.clean_description(description)
        
        # Split into words
        words = cleaned.split()
        
        # Filter words
        keywords = []
        for word in words:
            if len(word) > 1:  # Skip single characters
                if include_stop_words or word.lower() not in self.stop_words:
                    keywords.append(word)
        
        return keywords
    
    def find_technical_terms(self, description: str) -> List[str]:
        """
        Find technical terms in description.
        
        Args:
            description: The description to analyze
            
        Returns:
            List of technical terms found
        """
        if not description:
            return []
        
        cleaned = self.clean_description(description)
        words = set(cleaned.split())
        
        # Find technical terms
        found_terms = []
        for word in words:
            if word.lower() in self.technical_terms:
                found_terms.append(word)
        
        return found_terms


class PerformanceTracker:
    """
    Utility class for tracking performance metrics.
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'database_queries': 0,
            'pattern_matches': {},
            'start_time': datetime.now()
        }
    
    def record_request(self, success: bool = True, response_time: float = 0.0):
        """
        Record a request metric.
        
        Args:
            success: Whether the request was successful
            response_time: Time taken to process the request
        """
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update average response time
        total_time = self.metrics['average_response_time'] * (self.metrics['total_requests'] - 1)
        self.metrics['average_response_time'] = (total_time + response_time) / self.metrics['total_requests']
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics['cache_misses'] += 1
    
    def record_database_query(self):
        """Record a database query."""
        self.metrics['database_queries'] += 1
    
    def record_pattern_match(self, pattern: str):
        """
        Record a pattern match.
        
        Args:
            pattern: The pattern that was matched
        """
        if pattern not in self.metrics['pattern_matches']:
            self.metrics['pattern_matches'][pattern] = 0
        self.metrics['pattern_matches'][pattern] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        current_time = datetime.now()
        uptime = (current_time - self.metrics['start_time']).total_seconds()
        
        metrics = self.metrics.copy()
        metrics['uptime_seconds'] = uptime
        metrics['requests_per_second'] = self.metrics['total_requests'] / uptime if uptime > 0 else 0
        
        # Calculate cache hit rate
        total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        metrics['cache_hit_rate'] = (self.metrics['cache_hits'] / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        return metrics
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'database_queries': 0,
            'pattern_matches': {},
            'start_time': datetime.now()
        }


class ConfigurationManager:
    """
    Utility class for managing application configuration.
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                return self.get_default_config()
        else:
            return self.get_default_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "scoring": {
                "part_number_weight": 0.7,
                "description_weight": 0.3,
                "noise_penalty": 0.1,
                "min_score_threshold": 0.3
            },
            "search": {
                "max_results": 100,
                "fuzzy_matching": True,
                "case_sensitive": False,
                "include_variants": True
            },
            "performance": {
                "enable_caching": True,
                "cache_timeout": 3600,
                "max_cache_size": 10000,
                "parallel_processing": True,
                "max_workers": 4
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "app.log"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
        self.save_config()


# Additional utility functions for part number processing

def iterative_search_part(db_manager, part_number: str, max_iterations: int = 5) -> List[Dict]:
    """
    Perform iterative search for part numbers with progressively relaxed matching.
    
    Args:
        db_manager: DatabaseManager instance
        part_number: Part number to search for
        max_iterations: Maximum number of search iterations
        
    Returns:
        List of matching parts
    """
    if not part_number:
        logging.warning("Empty part number provided to iterative_search_part")
        return []
        
    # Validate the db_manager parameter
    if not hasattr(db_manager, '_execute_query'):
        logging.error("Invalid db_manager provided to iterative_search_part: missing _execute_query method")
        return []
    
    results = []
    search_patterns = _generate_progressive_variations(part_number)
    
    for i, pattern in enumerate(search_patterns[:max_iterations]):
        try:
            if not pattern:
                continue
                
            # Make the pattern SQL-safe by escaping single quotes
            safe_pattern = pattern.replace("'", "''")
                
            # Search in multiple part number columns using the DatabaseManager's method
            query = f"""
                SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
                FROM {settings.databricks_table_name}
                WHERE SPARTNUMBER LIKE '%{safe_pattern}%' OR PARTNUMBER LIKE '%{safe_pattern}%'
                ORDER BY 
                    CASE 
                        WHEN PARTNUMBER = '{safe_pattern}' THEN 1
                        WHEN PARTNUMBER LIKE '{safe_pattern}%' THEN 2
                        ELSE 3
                    END
                LIMIT 50
            """
            
            # Use the DatabaseManager's _execute_query method
            iteration_results = db_manager._execute_query(query)
            
            if iteration_results:
                for row in iteration_results:
                    result = {
                        'PARTNUMBER': row.get('PARTNUMBER', ''),
                        'partdesc': row.get('partdesc', ''),
                        'PARTMFR': row.get('PARTMFR', ''),
                        'CLASS': row.get('CLASS', ''),
                        'PARTINDEX': row.get('PARTINDEX', ''),
                        'SPARTNUMBER': row.get('SPARTNUMBER', ''),
                        'search_pattern': pattern,
                        'iteration': i + 1
                    }
                    results.append(result)
                
                # If we found good matches, prioritize them
                if len(iteration_results) > 0:
                    break
            
        except Exception as e:
            logging.error(f"Error in iterative search iteration {i}: {e}")
            continue
    
    return results

def _generate_complex_separator_variations(part_number: str) -> List[str]:
    """Generate variations with different separators."""
    variations = [part_number]
    
    # Common separators to try
    separators = ['-', '_', '.', ' ', '/', '\\', '|']
    
    for sep in separators:
        if sep not in part_number:
            # Add separator at common positions
            for i in range(1, len(part_number)):
                if part_number[i-1].isalnum() and part_number[i].isalnum():
                    variation = part_number[:i] + sep + part_number[i:]
                    if variation not in variations:
                        variations.append(variation)
    
    # Remove separators
    no_sep = re.sub(r'[-_.\s/\\|]', '', part_number)
    if no_sep != part_number and no_sep not in variations:
        variations.append(no_sep)
    
    return variations

def _generate_pattern_variations(part_number: str) -> List[str]:
    """Generate pattern variations for fuzzy matching."""
    variations = [part_number]
    
    # Remove common prefixes/suffixes
    prefixes = ['P', 'PN', 'PART', 'MFG', 'OEM']
    suffixes = ['A', 'B', 'C', 'X', 'Y', 'Z', 'NEW', 'OLD']
    
    clean_part = part_number.upper()
    
    for prefix in prefixes:
        if clean_part.startswith(prefix):
            variation = clean_part[len(prefix):].lstrip('-_. ')
            if variation and variation not in variations:
                variations.append(variation)
    
    for suffix in suffixes:
        if clean_part.endswith(suffix):
            variation = clean_part[:-len(suffix)].rstrip('-_. ')
            if variation and variation not in variations:
                variations.append(variation)
    
    return variations

def _generate_progressive_variations(part_number: str) -> List[str]:
    """Generate progressively more relaxed search patterns."""
    variations = []
    
    # 1. Exact match
    variations.append(part_number)
    
    # 2. Case insensitive
    variations.append(part_number.upper())
    variations.append(part_number.lower())
    
    # 3. Remove spaces and special characters
    clean = re.sub(r'[^\w]', '', part_number)
    if clean not in variations:
        variations.append(clean)
    
    # 4. Separator variations
    sep_variations = _generate_complex_separator_variations(part_number)
    for var in sep_variations:
        if var not in variations:
            variations.append(var)
    
    # 5. Pattern variations
    pattern_variations = _generate_pattern_variations(part_number)
    for var in pattern_variations:
        if var not in variations:
            variations.append(var)
    
    # 6. Substring variations
    substring_variations = _generate_substring_variations(part_number)
    for var in substring_variations:
        if var not in variations:
            variations.append(var)
    
    return variations

def _generate_substring_variations(part_number: str) -> List[str]:
    """Generate substring variations for partial matching."""
    variations = []
    
    # Extract numeric parts
    numeric_parts = re.findall(r'\d+', part_number)
    for num in numeric_parts:
        if len(num) >= 3:  # Only meaningful numeric parts
            variations.append(num)
    
    # Extract alphabetic parts
    alpha_parts = re.findall(r'[A-Za-z]+', part_number)
    for alpha in alpha_parts:
        if len(alpha) >= 2:  # Only meaningful alphabetic parts
            variations.append(alpha)
    
    # Generate sliding window substrings
    min_length = max(3, len(part_number) // 2)
    for i in range(len(part_number) - min_length + 1):
        substring = part_number[i:i + min_length]
        if substring not in variations:
            variations.append(substring)
    
    return variations

def _calculate_similarity_score(str1: str, str2: str) -> float:
    """Calculate similarity score between two strings."""
    if not str1 or not str2:
        return 0.0
    
    # Use SequenceMatcher for basic similarity
    basic_similarity = SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    # Check for substring matches
    str1_lower = str1.lower()
    str2_lower = str2.lower()
    
    substring_bonus = 0.0
    if str1_lower in str2_lower or str2_lower in str1_lower:
        substring_bonus = 0.2
    
    # Check for common parts
    str1_parts = re.findall(r'\w+', str1_lower)
    str2_parts = re.findall(r'\w+', str2_lower)
    
    common_parts = set(str1_parts) & set(str2_parts)
    if common_parts:
        common_bonus = len(common_parts) / max(len(str1_parts), len(str2_parts)) * 0.1
    else:
        common_bonus = 0.0
    
    total_score = min(1.0, basic_similarity + substring_bonus + common_bonus)
    return total_score


# Global instances
part_cleaner = PartNumberCleaner()
description_processor = DescriptionProcessor()
performance_tracker = PerformanceTracker()
config_manager = ConfigurationManager()
