import re
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class PartNumberScoringSystem:
    """
    Implements the scoring system for part number and description matching.
    Provides character-based scoring for part numbers, word-based scoring for descriptions,
    noise detection, and confidence code generation.
    """
    
    def __init__(self):
        """Initialize the scoring system."""
        pass
    
    def custom_round(self, value: float) -> int:
        """
        Custom rounding function that only rounds up if decimal part >= 0.8
        
        Args:
            value: The float value to round
            
        Returns:
            Rounded integer value
        """
        if value < 0:
            return 0
        
        # Use round() to handle floating point precision issues
        # Round to 2 decimal places first to avoid precision errors
        value = round(value, 2)
        
        integer_part = int(value)
        decimal_part = value - integer_part
        
        # Round up if decimal part is >= 0.8, otherwise round down
        if decimal_part >= 0.8:
            return integer_part + 1
        else:
            return integer_part
    
    def calculate_part_number_score(self, input_part: str, matched_part: str) -> int:
        """
        Calculate character-based scoring for part number matching (out of 5).
        
        Based on: matching_characters / total_input_characters
        Example: "BBM1693" vs "1693" = 4/7 = 57.14% = 2.857 -> 3 (since 0.857 >= 0.8)
        
        Args:
            input_part: The input part number (cleaned)
            matched_part: The matched part number from database
            
        Returns:
            Score from 0-5 based on character matching percentage
        """
        if not input_part or not matched_part:
            return 0
        
        # Clean both part numbers for comparison
        input_clean = self._clean_for_comparison(input_part)
        matched_clean = self._clean_for_comparison(matched_part)
        
        if not input_clean or not matched_clean:
            return 0
        
        # Convert to uppercase for case-insensitive comparison
        input_upper = input_clean.upper()
        matched_upper = matched_clean.upper()
        
        # Count matching characters
        matching_chars = 0
        
        # Create character frequency maps
        input_char_count = {}
        for char in input_upper:
            input_char_count[char] = input_char_count.get(char, 0) + 1
        
        matched_char_count = {}
        for char in matched_upper:
            matched_char_count[char] = matched_char_count.get(char, 0) + 1
        
        # Count overlapping characters
        for char, input_count in input_char_count.items():
            if char in matched_char_count:
                matching_chars += min(input_count, matched_char_count[char])
        
        # Calculate percentage match based on input length
        total_input_chars = len(input_upper)
        match_percentage = matching_chars / total_input_chars if total_input_chars > 0 else 0
        
        # Scale to 5-point system
        raw_score = match_percentage * 5
        
        # Apply custom rounding (>= 0.8 rounds up)
        final_score = self.custom_round(raw_score)
        
        return final_score
    
    def detect_noise(self, input_part: str, matched_part: str) -> int:
        """
        Detect noise in part number matching.
        
        Noise = 1 when matched part has extra characters beyond the input.
        Example: Input "1693" vs Match "1693B" -> Noise = 1 (extra "B")
        Example: Input "BB M1693" vs Match "1693" -> Noise = 0 (no extra chars)
        
        Args:
            input_part: The input part number (cleaned)
            matched_part: The matched part number from database
            
        Returns:
            1 if noise detected (extra characters), 0 if no noise
        """
        if not input_part or not matched_part:
            return 0
        
        # Clean both part numbers for comparison
        input_clean = self._clean_for_comparison(input_part)
        matched_clean = self._clean_for_comparison(matched_part)
        
        if not input_clean or not matched_clean:
            return 0
        
        # Convert to uppercase for case-insensitive comparison
        input_upper = input_clean.upper()
        matched_upper = matched_clean.upper()
        
        # If matched part is longer than input, check for extra characters
        if len(matched_upper) > len(input_upper):
            # Check if input contains all characters of matched part
            # If matched has extra characters, it's noise
            return 1  # Noise detected - matched is longer
        
        # If matched part is same length or shorter, check for different characters
        # Count character frequency in both parts
        input_char_count = {}
        for char in input_upper:
            input_char_count[char] = input_char_count.get(char, 0) + 1
        
        matched_char_count = {}
        for char in matched_upper:
            matched_char_count[char] = matched_char_count.get(char, 0) + 1
        
        # Check if matched part has any characters not in input, or more of any character
        for char, count in matched_char_count.items():
            if char not in input_char_count or count > input_char_count[char]:
                return 1  # Noise detected - extra or different characters
        
        return 0  # No noise detected
    
    def calculate_description_score(self, input_description: str, matched_description: str) -> int:
        """
        Calculate word-based scoring for description matching (out of 5).
        
        Based on: matching_words / total_words_in_input_description
        Example: "BREAK SHOE" vs "SOMETHING SOMETHING AND SOMETHING MORE BUT A BREAK SHOE"
        = 2/2 = 100% = 5 (all input words are found in the database description)
        
        Args:
            input_description: The input description
            matched_description: The matched description from database
            
        Returns:
            Score from 0-5 based on word matching percentage
        """
        if not input_description or not matched_description:
            return 0
        
        # Clean and tokenize descriptions (case-insensitive)
        input_words = self._tokenize_description(input_description)
        matched_words = self._tokenize_description(matched_description)
        
        if not input_words or not matched_words:
            return 0
        
        # Count matching words (case-insensitive)
        input_words_set = set(word.upper() for word in input_words)
        matched_words_set = set(word.upper() for word in matched_words)
        
        # Count how many input words are found in matched description
        matching_words = len(input_words_set.intersection(matched_words_set))
        
        # Use TOTAL words in input description as denominator (not just unique words)
        total_words_in_input = len(input_words)  # All words in input, not just unique ones
        
        # Calculate percentage match
        match_percentage = matching_words / total_words_in_input if total_words_in_input > 0 else 0
        
        # Scale to 5-point system
        raw_score = match_percentage * 5
        
        # Apply custom rounding (>= 0.8 rounds up)
        final_score = self.custom_round(raw_score)
        
        return final_score
    
    def generate_confidence_code(self, part_score: int, description_score: int) -> str:
        """
        Generate a two-digit confidence code based on part and description scores.
        
        Args:
            part_score: Part number score (0-5)
            description_score: Description score (0-5)
            
        Returns:
            Two-digit confidence code as string
        """
        # Ensure scores are within valid range
        part_score = max(0, min(5, part_score))
        description_score = max(0, min(5, description_score))
        
        return f"{part_score}{description_score}"
    
    def calculate_all_scores(self, 
                           input_part: str, 
                           matched_part: str,
                           input_description: Optional[str] = None,
                           matched_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate all scores and metrics for a match.
        
        Args:
            input_part: The input part number
            matched_part: The matched part number from database
            input_description: Optional input description
            matched_description: Optional matched description from database
            
        Returns:
            Dictionary containing all scoring metrics
        """
        # Calculate part number score
        part_score = self.calculate_part_number_score(input_part, matched_part)
        
        # Calculate noise detection
        noise = self.detect_noise(input_part, matched_part)
        
        # Calculate description score
        description_score = 0
        if input_description and matched_description:
            description_score = self.calculate_description_score(input_description, matched_description)
        
        # Generate confidence code
        cocode = self.generate_confidence_code(part_score, description_score)
        
        # Convert noise from integer to boolean, and format for consistent output
        noise_bool = bool(noise)
        
        return {
            "part_number_score": part_score,
            "description_score": description_score,
            "noise": noise_bool,  # Boolean value
            "noise_detected": noise,  # Integer value (0 or 1) for frontend compatibility
            "noise_text": "yes" if noise_bool else "no",  # String representation
            "cocode": cocode
        }
    
    def _clean_for_comparison(self, part_number: str) -> str:
        """
        Clean part number for comparison by removing spaces and special characters.
        
        Args:
            part_number: The part number to clean
            
        Returns:
            Cleaned part number
        """
        if not part_number:
            return ""
        
        # Remove spaces and common separators, keep alphanumeric
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', part_number.strip())
        return cleaned
    
    def _tokenize_description(self, description: str) -> List[str]:
        """
        Tokenize description into words for comparison.
        Keep ALL words including common ones, as per user specification.
        
        Args:
            description: The description to tokenize
            
        Returns:
            List of words (case-insensitive, but preserves all words)
        """
        if not description:
            return []
        
        # Split by whitespace and common separators, filter out very short words
        words = re.findall(r'\b\w+\b', description.upper())
        
        # Keep all words including common ones (AND, BUT, etc.) as they count in the total
        # Only filter out single characters and empty strings
        words = [word for word in words if len(word) > 1]
        
        return words


# Global instance for easy access
scoring_system = PartNumberScoringSystem()
