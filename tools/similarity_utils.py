"""
Similarity utilities for the Motor Parts Lookup System.
This module provides various similarity calculation methods for part numbers and descriptions.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from difflib import SequenceMatcher
import math
from collections import Counter

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """
    Provides various similarity calculation methods for part numbers and descriptions.
    """
    
    def __init__(self):
        """Initialize the similarity calculator."""
        self.weights = {
            'exact_match': 1.0,
            'case_insensitive': 0.95,
            'numeric_match': 0.9,
            'partial_match': 0.8,
            'fuzzy_match': 0.7,
            'phonetic_match': 0.6
        }
    
    def calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using multiple methods.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not str1 or not str2:
            return 0.0
        
        # Exact match
        if str1 == str2:
            return 1.0
        
        # Case insensitive match
        if str1.lower() == str2.lower():
            return self.weights['case_insensitive']
        
        # Calculate various similarity scores
        scores = []
        
        # Sequence matching
        seq_similarity = SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        scores.append(seq_similarity)
        
        # Jaccard similarity
        jaccard_similarity = self._jaccard_similarity(str1, str2)
        scores.append(jaccard_similarity)
        
        # Cosine similarity
        cosine_similarity = self._cosine_similarity(str1, str2)
        scores.append(cosine_similarity)
        
        # Levenshtein distance based similarity
        levenshtein_similarity = self._levenshtein_similarity(str1, str2)
        scores.append(levenshtein_similarity)
        
        # Return the maximum score
        return max(scores)
    
    def calculate_part_number_similarity(self, part1: str, part2: str) -> float:
        """
        Calculate similarity between two part numbers with special handling.
        
        Args:
            part1: First part number
            part2: Second part number
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not part1 or not part2:
            return 0.0
        
        # Clean part numbers
        clean1 = self._clean_part_number(part1)
        clean2 = self._clean_part_number(part2)
        
        # Exact match after cleaning
        if clean1 == clean2:
            return 1.0
        
        # Extract numeric and alphabetic parts
        num1 = self._extract_numbers(clean1)
        num2 = self._extract_numbers(clean2)
        alpha1 = self._extract_letters(clean1)
        alpha2 = self._extract_letters(clean2)
        
        scores = []
        
        # Full string similarity
        full_similarity = self.calculate_string_similarity(clean1, clean2)
        scores.append(full_similarity)
        
        # Numeric part similarity
        if num1 and num2:
            num_similarity = self.calculate_string_similarity(num1, num2)
            scores.append(num_similarity * self.weights['numeric_match'])
        
        # Alphabetic part similarity
        if alpha1 and alpha2:
            alpha_similarity = self.calculate_string_similarity(alpha1, alpha2)
            scores.append(alpha_similarity * 0.8)
        
        # Substring matching
        if clean1 in clean2 or clean2 in clean1:
            substring_similarity = min(len(clean1), len(clean2)) / max(len(clean1), len(clean2))
            scores.append(substring_similarity * self.weights['partial_match'])
        
        return max(scores) if scores else 0.0
    
    def calculate_description_similarity(self, desc1: str, desc2: str) -> float:
        """
        Calculate similarity between two descriptions.
        
        Args:
            desc1: First description
            desc2: Second description
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not desc1 or not desc2:
            return 0.0
        
        # Clean descriptions
        clean1 = self._clean_description(desc1)
        clean2 = self._clean_description(desc2)
        
        # Exact match
        if clean1 == clean2:
            return 1.0
        
        # Tokenize into words
        words1 = set(clean1.split())
        words2 = set(clean2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate word-based similarities
        scores = []
        
        # Jaccard similarity of words
        jaccard_score = len(words1.intersection(words2)) / len(words1.union(words2))
        scores.append(jaccard_score)
        
        # Cosine similarity of words
        cosine_score = self._word_cosine_similarity(words1, words2)
        scores.append(cosine_score)
        
        # Sequence similarity
        seq_score = SequenceMatcher(None, clean1, clean2).ratio()
        scores.append(seq_score)
        
        return max(scores)
    
    def find_similar_parts(self, target: str, candidates: List[str], 
                          threshold: float = 0.5, max_results: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar parts from a list of candidates.
        
        Args:
            target: Target part number
            candidates: List of candidate part numbers
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of (part_number, similarity_score) tuples
        """
        if not target or not candidates:
            return []
        
        similarities = []
        
        for candidate in candidates:
            similarity = self.calculate_part_number_similarity(target, candidate)
            if similarity >= threshold:
                similarities.append((candidate, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:max_results]
    
    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate Jaccard similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Jaccard similarity score
        """
        # Convert to character sets
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate cosine similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Cosine similarity score
        """
        # Create character frequency vectors
        chars1 = Counter(str1.lower())
        chars2 = Counter(str2.lower())
        
        # Get all unique characters
        all_chars = set(chars1.keys()) | set(chars2.keys())
        
        if not all_chars:
            return 1.0
        
        # Create vectors
        vec1 = [chars1.get(char, 0) for char in all_chars]
        vec2 = [chars2.get(char, 0) for char in all_chars]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _levenshtein_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity based on Levenshtein distance.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score based on edit distance
        """
        if str1 == str2:
            return 1.0
        
        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(str1.lower(), str2.lower())
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Edit distance
        """
        if len(str1) < len(str2):
            return self._levenshtein_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _word_cosine_similarity(self, words1: Set[str], words2: Set[str]) -> float:
        """
        Calculate cosine similarity between two sets of words.
        
        Args:
            words1: First set of words
            words2: Second set of words
            
        Returns:
            Cosine similarity score
        """
        if not words1 or not words2:
            return 0.0
        
        # Create word frequency vectors
        freq1 = Counter(words1)
        freq2 = Counter(words2)
        
        # Get all unique words
        all_words = set(freq1.keys()) | set(freq2.keys())
        
        if not all_words:
            return 1.0
        
        # Create vectors
        vec1 = [freq1.get(word, 0) for word in all_words]
        vec2 = [freq2.get(word, 0) for word in all_words]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _clean_part_number(self, part_number: str) -> str:
        """
        Clean a part number for comparison.
        
        Args:
            part_number: Part number to clean
            
        Returns:
            Cleaned part number
        """
        if not part_number:
            return ""
        
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w]', '', part_number.upper())
        return cleaned
    
    def _clean_description(self, description: str) -> str:
        """
        Clean a description for comparison.
        
        Args:
            description: Description to clean
            
        Returns:
            Cleaned description
        """
        if not description:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', description.lower().strip())
        
        # Remove special characters but keep spaces
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        
        # Remove extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _extract_numbers(self, text: str) -> str:
        """
        Extract numeric portions from text.
        
        Args:
            text: Text to process
            
        Returns:
            Concatenated numeric portions
        """
        numbers = re.findall(r'\d+', text)
        return ''.join(numbers)
    
    def _extract_letters(self, text: str) -> str:
        """
        Extract alphabetic portions from text.
        
        Args:
            text: Text to process
            
        Returns:
            Concatenated alphabetic portions
        """
        letters = re.findall(r'[A-Za-z]+', text)
        return ''.join(letters).upper()


class FuzzyMatcher:
    """
    Specialized fuzzy matching for part numbers.
    """
    
    def __init__(self, similarity_calculator: SimilarityCalculator = None):
        """
        Initialize the fuzzy matcher.
        
        Args:
            similarity_calculator: Instance of SimilarityCalculator
        """
        self.similarity_calc = similarity_calculator or SimilarityCalculator()
        self.match_strategies = [
            'exact_match',
            'case_insensitive',
            'substring_match',
            'fuzzy_match',
            'numeric_match',
            'partial_match'
        ]
    
    def find_best_matches(self, query: str, candidates: List[str], 
                         strategy: str = 'auto', threshold: float = 0.5) -> List[Tuple[str, float, str]]:
        """
        Find the best matches using the specified strategy.
        
        Args:
            query: Query string
            candidates: List of candidate strings
            strategy: Matching strategy ('auto' or specific strategy)
            threshold: Minimum similarity threshold
            
        Returns:
            List of (candidate, similarity, strategy) tuples
        """
        if not query or not candidates:
            return []
        
        matches = []
        
        if strategy == 'auto':
            # Try all strategies
            for strat in self.match_strategies:
                strategy_matches = self._apply_strategy(query, candidates, strat, threshold)
                matches.extend(strategy_matches)
        else:
            # Use specific strategy
            matches = self._apply_strategy(query, candidates, strategy, threshold)
        
        # Remove duplicates and sort by similarity
        unique_matches = {}
        for candidate, similarity, used_strategy in matches:
            if candidate not in unique_matches or similarity > unique_matches[candidate][0]:
                unique_matches[candidate] = (similarity, used_strategy)
        
        # Convert back to list format
        final_matches = [(candidate, similarity, strategy) 
                        for candidate, (similarity, strategy) in unique_matches.items()]
        
        # Sort by similarity (descending)
        final_matches.sort(key=lambda x: x[1], reverse=True)
        
        return final_matches
    
    def _apply_strategy(self, query: str, candidates: List[str], 
                       strategy: str, threshold: float) -> List[Tuple[str, float, str]]:
        """
        Apply a specific matching strategy.
        
        Args:
            query: Query string
            candidates: List of candidate strings
            strategy: Matching strategy to use
            threshold: Minimum similarity threshold
            
        Returns:
            List of (candidate, similarity, strategy) tuples
        """
        matches = []
        
        for candidate in candidates:
            similarity = 0.0
            
            if strategy == 'exact_match':
                similarity = 1.0 if query == candidate else 0.0
            elif strategy == 'case_insensitive':
                similarity = 1.0 if query.lower() == candidate.lower() else 0.0
            elif strategy == 'substring_match':
                if query.lower() in candidate.lower() or candidate.lower() in query.lower():
                    similarity = min(len(query), len(candidate)) / max(len(query), len(candidate))
            elif strategy == 'fuzzy_match':
                similarity = self.similarity_calc.calculate_string_similarity(query, candidate)
            elif strategy == 'numeric_match':
                query_nums = self.similarity_calc._extract_numbers(query)
                candidate_nums = self.similarity_calc._extract_numbers(candidate)
                if query_nums and candidate_nums:
                    similarity = self.similarity_calc.calculate_string_similarity(query_nums, candidate_nums)
            elif strategy == 'partial_match':
                # Check if significant portion matches
                similarity = SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
            
            if similarity >= threshold:
                matches.append((candidate, similarity, strategy))
        
        return matches

# Additional similarity functions used by processor
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

def calculate_confidence(part_score: float, description_score: float = 0.0) -> str:
    """
    Calculate confidence level based on part and description scores.
    
    Args:
        part_score: Similarity score for part number (0.0 to 1.0)
        description_score: Similarity score for description (0.0 to 1.0)
        
    Returns:
        Confidence code (00-90)
    """
    try:
        # Ensure inputs are proper floats
        part_score = float(part_score)
        description_score = float(description_score)
        
        # Validate ranges to prevent errors
        part_score = max(0.0, min(1.0, part_score))
        description_score = max(0.0, min(1.0, description_score))
        
        # Calculate weighted score
        total_score = part_score + (description_score * 0.3)  # Weight description less
        
        # Cap at 1.0 to prevent scores over 100%
        total_score = min(1.0, total_score)
        
        # Map to confidence buckets
        if total_score >= 0.9:
            return "90"
        elif total_score >= 0.8:
            return "80"
        elif total_score >= 0.7:
            return "70"
        elif total_score >= 0.6:
            return "60"
        elif total_score >= 0.5:
            return "50"
        elif total_score >= 0.4:
            return "40"
        elif total_score >= 0.3:
            return "30"
        elif total_score >= 0.2:
            return "20"
        elif total_score >= 0.1:
            return "10"
        else:
            return "00"
    except Exception as e:
        logging.error(f"Error calculating confidence: {e}")
        return "00"  # Return minimum confidence on error

def calculate_part_confidence(input_part: str, candidate_part: str) -> str:
    """
    Calculate confidence code from part numbers directly.
    
    This function converts part numbers to similarity scores before calling calculate_confidence.
    Use this function when you have part number strings instead of similarity scores.
    
    Args:
        input_part: Input part number string
        candidate_part: Candidate part number string
        
    Returns:
        Confidence code (00-90)
    """
    # Calculate the similarity score between the part numbers
    sim_calculator = SimilarityCalculator()
    part_score = sim_calculator.calculate_part_number_similarity(input_part, candidate_part)
    
    # Call the existing function with the calculated score
    return calculate_confidence(part_score)

def text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity between two strings.
    
    Args:
        text1: First string to compare
        text2: Second string to compare
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Handle None or empty strings
    if not text1 or not text2:
        return 0.0
    
    # Ensure inputs are strings
    text1 = str(text1).lower().strip()
    text2 = str(text2).lower().strip()
    
    # Check for exact match
    if text1 == text2:
        return 1.0
    
    # Use the SimilarityCalculator for consistent results
    calculator = SimilarityCalculator()
    return calculator._levenshtein_similarity(text1, text2)

def calculate_combined_confidence(part_number_score: float, 
                                description_score: float = 0.0,
                                manufacturer_match: bool = False,
                                category_match: bool = False) -> str:
    """
    Calculate combined confidence from multiple factors.
    
    Args:
        part_number_score: Similarity score for part number (0.0 to 1.0)
        description_score: Similarity score for description (0.0 to 1.0)
        manufacturer_match: Whether manufacturer matches
        category_match: Whether category matches
        
    Returns:
        Confidence code (00-90)
    """
    try:
        # Ensure scores are proper floats
        part_number_score = float(part_number_score)
        description_score = float(description_score)
        
        # Validate ranges
        part_number_score = max(0.0, min(1.0, part_number_score))
        description_score = max(0.0, min(1.0, description_score))
        
        # Base score from part number
        base_score = part_number_score
        
        # Add description bonus
        if description_score > 0:
            base_score += description_score * 0.2
        
        # Add manufacturer match bonus
        if manufacturer_match:
            base_score += 0.1
        
        # Add category match bonus
        if category_match:
            base_score += 0.05
        
        # Cap at 1.0
        base_score = min(1.0, base_score)
        
        # Convert to confidence code
        return calculate_confidence(base_score)
    except Exception as e:
        logging.error(f"Error calculating combined confidence: {e}")
        return "00"  # Return minimum confidence on error

def is_description_match(input_desc: str, matched_desc: str, threshold: float = 0.7) -> bool:
    """Check if descriptions match above threshold."""
    if not input_desc or not matched_desc:
        return False
    
    similarity = text_similarity(input_desc, matched_desc)
    return similarity >= threshold

def generate_description_patterns(description: str) -> List[str]:
    """Generate search patterns from description."""
    patterns = []
    
    if not description:
        return patterns
    
    # Clean and normalize
    clean_desc = re.sub(r'[^\w\s]', ' ', description.lower())
    words = clean_desc.split()
    
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Add individual meaningful words
    patterns.extend(meaningful_words)
    
    # Add combinations of 2-3 words
    for i in range(len(meaningful_words) - 1):
        patterns.append(f"{meaningful_words[i]} {meaningful_words[i+1]}")
        if i < len(meaningful_words) - 2:
            patterns.append(f"{meaningful_words[i]} {meaningful_words[i+1]} {meaningful_words[i+2]}")
    
    # Add technical terms
    technical_patterns = re.findall(r'\b\d+[A-Z]+\b|\b[A-Z]+\d+\b|\b\d+[-.]?\d*\s*[A-Z]+\b', description.upper())
    patterns.extend(technical_patterns)
    
    return list(set(patterns))  # Remove duplicates

def description_matches_patterns(description: str, patterns: List[str]) -> bool:
    """Check if description matches any of the patterns."""
    if not description or not patterns:
        return False
    
    description_lower = description.lower()
    
    for pattern in patterns:
        if pattern.lower() in description_lower:
            return True
    
    return False

def priority_description_match(input_desc: str, matched_desc: str) -> Dict[str, Any]:
    """Perform priority-based description matching."""
    result = {
        'match': False,
        'score': 0.0,
        'priority': 'none',
        'details': []
    }
    
    if not input_desc or not matched_desc:
        return result
    
    # Calculate basic similarity
    similarity = text_similarity(input_desc, matched_desc)
    result['score'] = similarity
    
    # High priority: exact match
    if input_desc.lower().strip() == matched_desc.lower().strip():
        result['match'] = True
        result['priority'] = 'exact'
        result['details'].append('Exact description match')
        return result
    
    # Medium priority: high similarity
    if similarity >= 0.8:
        result['match'] = True
        result['priority'] = 'high'
        result['details'].append(f'High similarity: {similarity:.2f}')
        return result
    
    # Low priority: pattern matching
    input_patterns = generate_description_patterns(input_desc)
    if description_matches_patterns(matched_desc, input_patterns):
        result['match'] = True
        result['priority'] = 'pattern'
        result['details'].append('Pattern match found')
        return result
    
    # Check for key technical terms
    technical_match = False
    technical_terms = re.findall(r'\b\d+[A-Z]+\b|\b[A-Z]+\d+\b|\b\d+[-.]?\d*\s*[A-Z]+\b', 
                               input_desc.upper())
    for term in technical_terms:
        if term.lower() in matched_desc.lower():
            technical_match = True
            result['details'].append(f'Technical term match: {term}')
    
    if technical_match:
        result['match'] = True
        result['priority'] = 'technical'
        return result
    
    # No significant match
    if similarity >= 0.5:
        result['match'] = True
        result['priority'] = 'weak'
        result['details'].append(f'Weak similarity: {similarity:.2f}')
    
    return result
