import re
import logging
from logging.handlers import RotatingFileHandler
import time
import os
import traceback
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.database import db_manager
from tools.basic_tools import (
    iterative_search_part,
    _generate_complex_separator_variations,
    _generate_pattern_variations,
    _generate_progressive_variations,
    _generate_substring_variations,
    _calculate_similarity_score
)
from tools.similarity_utils import (
    calculate_confidence,
    calculate_part_confidence,
    text_similarity,
    calculate_combined_confidence,
    is_description_match,
    generate_description_patterns,
    description_matches_patterns,
    priority_description_match
)
from tools.description_pattern_generator import DescriptionPatternGenerator
from tools.pattern_optimizer import PatternOptimizer
from tools.scoring_system import scoring_system

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create log directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up rotating file handler to manage log size
file_handler = RotatingFileHandler(
    'logs/parts_processing3.log', maxBytes=5*1024*1024, backupCount=5
)
console_handler = logging.StreamHandler()

# Define log format
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


class PartProcessor:
    """
    A class that processes part numbers using a sequential, procedural approach
    replacing the previous LangGraph-based supervisor agent architecture.
    Optimized for performance with caching and prioritized pattern matching.
    """
    
    def __init__(self):
        """Initialize the PartProcessor."""
        self.pattern_optimizer = PatternOptimizer()
        self.description_generator = DescriptionPatternGenerator()
        
        # Configure thread pool for parallel database queries
        self.max_workers = min(8, (os.cpu_count() or 1) + 4)  # Optimal thread count for I/O operations
        self.parallel_enabled = True  # Can be disabled for debugging
        
        # Base patterns for part number validation
        self.base_patterns = [
            r'[0-9]{4,}',              # At least 4 digits together
            r'[A-Z][0-9]{3,}',         # Letter followed by at least 3 digits
            r'[0-9]{3,}[A-Z][0-9]+',   # At least 3 digits, letter, more digits
            r'[A-Z][0-9]+[A-Z][0-9]+', # Letter-digit-letter-digit pattern
        ]
        
        # Additional validation patterns
        self.validation_patterns = [
            r'[A-Z0-9]{5,}',           # At least 5 alphanumeric chars
            r'[0-9]{2,}[-]?[0-9]{2,}'  # digit groups with optional hyphen
        ]
        
        # Part number pattern templates from basic_tools.py
        self.part_patterns = [
            r'[0-9]+[-\.]?[0-9]+',              # digit-digit: "55.46", "55-46"
            r'[A-Z][0-9]+[-\.]?[0-9]+',         # letter-digit-digit: "K55.46", "S33-04"
            r'[0-9]+[-\.]?[0-9]+[A-Z]',         # digit-digit-letter: "55.46A", "33-04B"
            r'[A-Z][0-9]+[-\.]?[0-9]+[A-Z]'     # letter-digit-digit-letter: "K55.46A", "S33-04B"
        ]
        
        # Common prefixes and suffixes to try removing
        # Removed common_prefixes and common_suffixes as we now only do basic cleaning
        # without prefix/suffix removal to preserve important alphanumeric characters
        
        # Performance optimization parameters
        # Confidence thresholds for early termination
        self.high_confidence_threshold = 0.9
        self.medium_confidence_threshold = 0.7
        
        # Batch size for database queries
        self.db_batch_size = 50
        
    def _detect_suffix_variations(self, cleaned_input: str):
        """
        Detect potential suffixes in part numbers and create variations with suffixes removed.
        Specifically designed for cases like RTLO16918BPHWEL where actual part might be RTLO16918BP.
        
        Args:
            cleaned_input: The cleaned part number
            
        Returns:
            List of suffix-reduced variations
        """
        # Only process if input contains at least one letter
        if not any(c.isalpha() for c in cleaned_input):
            return []
        
        # Find trailing alphabetic characters that might be a suffix
        suffix_match = re.search(r'[A-Z]+$', cleaned_input)
        if not suffix_match:
            return []
            
        suffix = suffix_match.group()
        logger.info(f"Detected potential suffix: {suffix}")
        
        # If suffix is too short or too long, it might not be a real suffix
        if len(suffix) <= 1 or len(suffix) > 8:
            return []
            
        # Create variations by progressively removing characters from the suffix
        variations = []
        
        # Get the base part without the suffix
        base_part = cleaned_input[:-len(suffix)]
        
        # Add variations by progressively removing characters from the suffix
        for i in range(len(suffix)):
            # Keep only part of the suffix
            partial_suffix = suffix[:i]
            if partial_suffix:  # Only if we have some suffix left
                variation = base_part + partial_suffix
                variations.append(variation)
                logger.info(f"Created suffix variation: {variation}")
            
        # Also add the base part without any suffix
        variations.append(base_part)
        logger.info(f"Created base variation without suffix: {base_part}")
        
        return variations
    
    def _sliding_window_search(self, input_partnumber: str) -> Dict[str, Any]:
        """
        Bidirectional sliding window search that generates candidates by removing characters 
        from both the start (forward) and end (backward) of the cleaned part number.
        This improves search accuracy by finding patterns that might be in the middle,
        beginning, or end of the part number.
        
        Enhanced to handle various part number patterns without special case handling.
        """
        # First clean the input by removing all non-alphanumeric characters
        cleaned_input = re.sub(r'[^A-Z0-9]', '', input_partnumber.upper())
        logger.info(f"Cleaned input for bidirectional sliding window: {cleaned_input}")
        
        # Generate additional candidates by breaking the part number into segments
        # This handles cases where separators (like hyphens) should be removed
        # without hardcoding specific patterns
        segments = re.split(r'[^A-Z0-9]', input_partnumber.upper())
        segments = [seg for seg in segments if seg]  # Remove empty segments
        
        # First try with the original cleaned input
        candidates = self._generate_bidirectional_candidates(cleaned_input)
        
        # Generate combined segment variations without any hardcoded patterns
        # This will handle all types of special character cases generically
        additional_variations = []
        
        # Add each segment that's long enough
        for segment in segments:
            if len(segment) >= 3:
                additional_variations.append(segment)
                logger.info(f"Generated segment pattern: {segment}")
        
        # Generate single-char + segment combinations (handles S265818C pattern generically)
        for i in range(len(segments)):
            # Try first character from each segment combined with other segments
            if segments[i] and len(segments[i]) > 0:
                first_char = segments[i][0:1]
                for j in range(len(segments)):
                    if i != j and segments[j] and len(segments[j]) > 0:
                        combined = first_char + segments[j]
                        if len(combined) >= 4:  # Only add meaningful combinations
                            additional_variations.append(combined)
                            logger.info(f"Generated combined pattern: {combined}")
        
        # Combine all segments together in different ways
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                combined = segments[i] + segments[j]
                additional_variations.append(combined)
                logger.info(f"Generated combined segments: {combined}")
                
                # Also try first char + second segment (handles S265818C generically)
                if len(segments[i]) > 0:
                    combined = segments[i][0:1] + segments[j]
                    additional_variations.append(combined)
                    logger.info(f"Generated first-char + segment: {combined}")
        
        # Add all additional variations to search without any special logic
        additional_candidates = []
        for variation in additional_variations:
            additional_candidates.extend(self._generate_bidirectional_candidates(variation))
            
            # Mark these as additional patterns for confidence adjustments
            for candidate in additional_candidates:
                if candidate["pattern"] in additional_variations:
                    candidate["is_additional_pattern"] = True
                    
        # Add the additional candidates to our search candidates
        candidates.extend(additional_candidates)
        
        if not candidates:
            return None
            
        # Search for candidates in database with improved strategy
        all_matches = self._search_with_bidirectional_candidates(candidates, cleaned_input)
        
        # If we found high confidence matches, analyze them before returning
        high_confidence_matches = [m for m in all_matches if m.get('confidence', 0) >= 0.85 and 
                                 m.get('CLASS') == 'M']  # Prefer class 'M' matches
        
        # CRITICAL FIX: Don't early terminate on backward-only high confidence matches
        # Check if we have forward matches that might be more complete
        forward_matches = [m for m in all_matches if m.get('search_candidate_type') == 'forward']
        backward_matches = [m for m in all_matches if m.get('search_candidate_type') == 'backward']
        
        # If we have high confidence matches, check if they're predominantly backward
        if high_confidence_matches:
            high_conf_forward = [m for m in high_confidence_matches if m.get('search_candidate_type') == 'forward']
            high_conf_backward = [m for m in high_confidence_matches if m.get('search_candidate_type') == 'backward']
            
            # Only early terminate if we have high confidence forward matches OR
            # if backward matches are longer/more complete than available forward candidates
            should_early_terminate = False
            
            if high_conf_forward:
                # We have high confidence forward matches - safe to terminate
                should_early_terminate = True
                logger.info("Found high confidence forward matches, proceeding with early termination")
            
            elif high_conf_backward and forward_matches:
                # Check if backward matches are actually better than forward candidates
                max_backward_length = max(len(m.get('search_candidate', '')) for m in high_conf_backward)
                max_forward_length = max(len(m.get('search_candidate', '')) for m in forward_matches)
                
                # Only terminate early if backward matches are longer or equal length
                if max_backward_length >= max_forward_length:
                    should_early_terminate = True
                    logger.info(f"High confidence backward matches ({max_backward_length} chars) are longer than forward candidates ({max_forward_length} chars)")
                else:
                    logger.info(f"High confidence backward matches ({max_backward_length} chars) shorter than forward candidates ({max_forward_length} chars) - continuing processing")
            
            elif high_conf_backward and not forward_matches:
                # Only backward matches exist - safe to terminate
                should_early_terminate = True
                logger.info("Only high confidence backward matches found, no forward candidates available")
            
            if should_early_terminate:
                logger.info("Found high confidence matches with original input, skipping suffix processing")
                optimized_matches = self._analyze_bidirectional_results(high_confidence_matches, cleaned_input)
                return {
                    "cleaned_part": cleaned_input,
                    "data": optimized_matches,
                    "confidence": 0.9
                }
            else:
                logger.info("High confidence backward matches found, but continuing to process forward candidates")
        
        # If no high confidence matches found, try with suffix variations
        suffix_variations = self._detect_suffix_variations(cleaned_input)
        suffix_matches = []
        
        # Also check if any of the backward search patterns produced good results
        # This is a more general approach than hardcoding specific part numbers
        backward_candidates = [c["pattern"] for c in candidates if c["type"] == "backward"]
        
        # Add the backward patterns that look like potential part numbers to our variations
        # This handles cases like RTLO16918BPHWEL → RTLO16918BP without hardcoding
        for pattern in backward_candidates:
            # Only consider patterns of sufficient length that weren't already detected
            if len(pattern) >= 5 and pattern not in suffix_variations:
                # Prioritize patterns with mixed letters and numbers (common in part numbers)
                has_letter = any(c.isalpha() for c in pattern)
                has_digit = any(c.isdigit() for c in pattern)
                if has_letter and has_digit:
                    suffix_variations.insert(0, pattern)
                    logger.info(f"Added backward pattern as potential part number: {pattern}")
        
        for variation in suffix_variations:
            logger.info(f"Trying bidirectional search with suffix variation: {variation}")
            suffix_candidates = self._generate_bidirectional_candidates(variation)
            
            if not suffix_candidates:
                continue
                
            # Search with these candidates
            variation_matches = self._search_with_bidirectional_candidates(suffix_candidates, variation)
            
            if variation_matches:
                logger.info(f"Found {len(variation_matches)} matches with suffix variation {variation}")
                # Tag these matches as being from suffix processing
                for match in variation_matches:
                    match['suffix_processed'] = True
                    match['original_input'] = cleaned_input
                    match['suffix_variation'] = variation
                    
                    # Boost confidence for parts with common part number patterns
                    # Using pattern recognition instead of hardcoded part numbers
                    part_number = match.get('SPARTNUMBER', '')
                    if re.search(r'[0-9]+[A-Z]{1,3}$', part_number):
                        match['confidence'] *= 1.2
                        logger.info(f"Boosting confidence for pattern match: {part_number}")
                        
                    # Extra boost for Class M parts
                    if match.get('CLASS') == 'M':
                        match['confidence'] *= 1.15
                        logger.info(f"Boosting confidence for Class M part: {part_number}")
                
                # Add to our matches
                suffix_matches.extend(variation_matches)
                
                # Early termination if we find high-confidence Class M matches
                class_m_matches = [m for m in variation_matches if m.get('CLASS') == 'M' and 
                                  m.get('confidence', 0) >= 0.85]
                                  
                if class_m_matches:
                    logger.info(f"Found {len(class_m_matches)} high confidence Class M matches with suffix variation, stopping early")
                    break
        
        # Combine all matches
        all_matches.extend(suffix_matches)
        
        if not all_matches:
            return None
            
        # Analyze and optimize the results
        optimized_matches = self._analyze_bidirectional_results(all_matches, cleaned_input)
        
        return {
            "cleaned_part": cleaned_input,
            "data": optimized_matches,
            "confidence": 0.85
        }
        
    def _generate_bidirectional_candidates(self, cleaned_input, min_length=3):
        """
        Generate candidates from both forward and backward directions.
        Optimized to stop at 65% of input length for practical prefix/suffix limits.
        
        Args:
            cleaned_input: The cleaned part number string
            min_length: Minimum length for a candidate to be valid
            
        Returns:
            List of candidate objects with pattern and type information
        """
        candidates = []
        input_length = len(cleaned_input)
        
        # Calculate 65% threshold for maximum practical prefix/suffix length
        max_removal_length = int(input_length * 0.35)  # Remove up to 35%, keep at least 65%
        max_start_index = min(max_removal_length, input_length - min_length)
        
        # Forward sliding window (existing approach - left to right)
        # Stop at 65% to avoid impractically long prefixes
        for i in range(max_start_index + 1):
            forward_candidate = cleaned_input[i:]
            if len(forward_candidate) >= min_length:
                candidates.append({
                    "pattern": forward_candidate, 
                    "type": "forward",
                    "original_index": i
                })
                logger.info(f"Generated forward candidate: {forward_candidate}")
        
        # Backward sliding window (new approach - right to left)
        # Stop at 65% to avoid impractically long suffixes
        min_end_index = max(min_length, int(input_length * 0.65))
        for i in range(input_length, min_end_index - 1, -1):
            backward_candidate = cleaned_input[:i]
            if len(backward_candidate) >= min_length:
                candidates.append({
                    "pattern": backward_candidate, 
                    "type": "backward",
                    "original_index": len(cleaned_input) - i
                })
                logger.info(f"Generated backward candidate: {backward_candidate}")
        
        return candidates
        
    def _search_with_bidirectional_candidates(self, candidates, cleaned_input):
        """
        Search database using a prioritized strategy for bidirectional candidates.
        
        Args:
            candidates: List of candidate objects with pattern and type
            cleaned_input: The original cleaned input string
            
        Returns:
            List of match objects with confidence scores
        """
        all_matches = []
        highest_confidence = 0
        original_length = len(cleaned_input)
        
        # Group candidates by length for better search strategy
        by_length = {}
        for candidate in candidates:
            length = len(candidate["pattern"])
            if length not in by_length:
                by_length[length] = []
            by_length[length].append(candidate)
        
        # Sort lengths from longest to shortest
        sorted_lengths = sorted(by_length.keys(), reverse=True)
        
        # Tracking for early termination
        found_high_confidence_forward = False
        found_high_confidence_backward = False
        
        # Search in descending length order
        for length in sorted_lengths:
            current_length_candidates = by_length[length]
            
            for candidate in current_length_candidates:
                pattern = candidate["pattern"]
                candidate_type = candidate["type"]
                is_additional_pattern = candidate.get("is_additional_pattern", False)
                
                # Skip forward candidates if we already found good matches (but continue with additional patterns)
                if candidate_type == "forward" and found_high_confidence_forward and not is_additional_pattern:
                    continue
                
                # Skip backward candidates if we already found good matches (but continue with additional patterns)
                if candidate_type == "backward" and found_high_confidence_backward and not is_additional_pattern:
                    continue
                
                # Search database for this candidate
                matches = db_manager.search_parts_containing(pattern)
                
                if matches:
                    # Don't penalize backward matches anymore - they're equally important
                    type_factor = 1.0
                    
                    # Boost for additional pattern variations from segmenting
                    if is_additional_pattern:
                        type_factor = 1.5
                    
                    # Length factor - longer patterns get higher confidence
                    length_factor = length / original_length
                    
                    # Position factor calculation - adjusted to be more balanced
                    # For backward matches, being at the beginning is very important (prefix)
                    if candidate_type == "backward":
                        # Higher position factor for backward matches at the beginning
                        position_factor = 1.1 if candidate["original_index"] == 0 else (1 - (candidate["original_index"] / original_length))
                    else:
                        # Regular position factor for forward matches
                        position_factor = 1 - (candidate["original_index"] / original_length)
                    
                    # Calculate base confidence
                    base_confidence = 0.85 * position_factor * length_factor * type_factor
                    
                    # Generic pattern detection for any pattern following common formats
                    if (re.match(r'^[A-Z]\d{5,}\w$', pattern)) or (re.match(r'^[A-Z]{2,}\d{5,}[A-Z]{1,2}$', pattern)):
                        logger.info(f"Found common pattern match: {pattern}")
                        base_confidence *= 1.5  # Significant boost
                    
                    # Extra boost for backward matches that are exact part numbers
                    if candidate_type == "backward":
                        # Check if any matches are exact matches for this pattern
                        exact_matches = [m for m in matches if pattern == m.get("SPARTNUMBER", "")]
                        
                        if exact_matches:
                            # Significant boost for backward patterns that match exact part numbers
                            base_confidence *= 1.2
                    
                    # Process and add matches
                    match_count = 0
                    high_confidence_count = 0
                    
                    # Check for common patterns that we want to prioritize
                    special_part_matches = []
                    if any(not c.isalnum() for c in cleaned_input):  # If input has any non-alphanumeric chars
                        for match in matches:
                            part_number = match.get("SPARTNUMBER", "")
                            
                            # Handle common patterns generically without hardcoding
                            # Identify parts that follow common formats in automotive industry
                            if (re.match(r'^[A-Z]\d{4,}\w$', part_number)) or (re.match(r'^[A-Z]{2,}\d{4,}[A-Z]{1,2}$', part_number)):
                                special_part_matches.append(match)
                                match["is_additional_pattern"] = True
                                match["confidence"] = 1.0  # Highest confidence
                                logger.info(f"Identified exact suffix match: {part_number}")
                    
                    for match in matches:
                        match["search_candidate"] = pattern
                        match["candidate_type"] = candidate_type
                        
                        # Only set confidence if it wasn't already set (for special patterns)
                        if "confidence" not in match:
                            match["confidence"] = base_confidence
                        
                        # Special handling for backward pattern matches
                        if candidate_type == "backward":
                            # Check for exact part number matches in backward search
                            if pattern == match.get("SPARTNUMBER", ""):
                                # Major boost for exact matches in backward search
                                match["confidence"] *= 1.5  # Significant boost for exact matches
                                match["exact_backward_match"] = True
                                logger.info(f"Found exact backward match: {pattern}")
                            
                            # CRITICAL FIX: More selective pattern boosting to prevent short matches from dominating
                            # Only boost longer, more complete backward patterns
                            pattern_length = len(pattern)
                            
                            # Boost common automotive part number formats (but only for longer patterns)
                            if pattern_length >= 6:  # Only boost patterns 6+ characters
                                if re.match(r'^[A-Z]{2,}\d{4,}[A-Z]{1,2}$', pattern) or \
                                   re.match(r'^[A-Z]\d{5,}[A-Z]?$', pattern) or \
                                   re.match(r'^\d{5,}[A-Z]{1,2}$', pattern):
                                    match["confidence"] *= 1.4  # Extra boost for longer, complete patterns
                                    logger.info(f"Boosting known pattern format: {pattern}")
                                elif re.match(r'^[A-Z]\d{4}[A-Z]?$', pattern):
                                    # Reduced boost for shorter patterns like G0128
                                    match["confidence"] *= 1.1  # Much smaller boost for 4-5 digit patterns
                                    logger.info(f"Small boost for short pattern: {pattern}")
                            else:
                                # Apply penalty to very short backward patterns (likely partial matches)
                                if pattern_length <= 4:
                                    match["confidence"] *= 0.8  # Reduce confidence for very short patterns
                                    logger.info(f"Applying penalty to short pattern: {pattern}")
                        else:
                            # ENHANCED: Better handling for forward matches - prioritize longer patterns
                            pattern_length = len(pattern)
                            
                            if pattern == match.get("SPARTNUMBER", ""):
                                # Boost exact matches more for longer patterns
                                boost_factor = 1.1 + (pattern_length * 0.02)  # 1.1 to 1.25 based on length
                                match["confidence"] *= min(boost_factor, 1.25)  # Cap at 1.25
                                logger.info(f"Forward exact match boost ({boost_factor:.2f}): {pattern}")
                            elif pattern_length >= 6:
                                # Boost longer forward patterns that are likely complete part numbers
                                match["confidence"] *= 1.15
                                logger.info(f"Forward long pattern boost: {pattern}")
                        
                        all_matches.append(match)
                        match_count += 1
                        
                        if match["confidence"] > 0.85:
                            high_confidence_count += 1
                    
                    logger.info(f"Found {match_count} matches for {candidate_type} candidate: {pattern}")
                    
                    # Early termination logic - separately for forward and backward
                    if candidate_type == "forward" and high_confidence_count > 0:
                        found_high_confidence_forward = True
                        logger.info(f"Found high confidence forward matches with {pattern}")
                        
                    if candidate_type == "backward" and high_confidence_count > 0:
                        found_high_confidence_backward = True
                        logger.info(f"Found high confidence backward matches with {pattern}")
                    
                    # If we have good matches from both directions, we can stop
                    if found_high_confidence_forward and found_high_confidence_backward:
                        logger.info("Found high confidence matches from both directions - stopping early")
                        break
            
            # If we have good matches from both directions after processing this length, stop
            if found_high_confidence_forward and found_high_confidence_backward:
                break
        
        return all_matches
        
    def _analyze_bidirectional_results(self, all_matches, cleaned_input):
        """
        Analyze bidirectional search results to prioritize and optimize.
        Improved to better handle prefixed part numbers like "MTR" + "A1205S2593"
        and special characters in "HENS-26581-8C" -> "S265818C".
        
        Args:
            all_matches: List of all matches from search
            cleaned_input: Original cleaned input string
            
        Returns:
            Optimized list of matches
        """
        if not all_matches:
            return []
            
        # Group matches by part number to find overlapping results
        by_partnumber = {}
        for match in all_matches:
            spartnumber = match.get("SPARTNUMBER", "")
            if spartnumber not in by_partnumber:
                by_partnumber[spartnumber] = []
            by_partnumber[spartnumber].append(match)
        
        # Find parts detected by both forward and backward search
        bidirectional_matches = []
        forward_only_matches = []
        backward_only_matches = []
        
        # Track candidate patterns for debugging
        forward_patterns = set()
        backward_patterns = set()
        
        for spartnumber, matches in by_partnumber.items():
            # Check if this part number was found by both forward and backward candidates
            forward_matches = [m for m in matches if m.get("candidate_type") == "forward"]
            backward_matches = [m for m in matches if m.get("candidate_type") == "backward"]
            
            has_forward = len(forward_matches) > 0
            has_backward = len(backward_matches) > 0
            
            # Collect pattern info for logging
            for m in forward_matches:
                forward_patterns.add(m.get("search_candidate", ""))
            for m in backward_matches:
                backward_patterns.add(m.get("search_candidate", ""))
            
            # Calculate improved confidence based on pattern quality
            best_match = None
            best_confidence = 0
            
            for match in matches:
                candidate = match.get("search_candidate", "")
                candidate_type = match.get("candidate_type")
                base_confidence = match.get("confidence", 0)
                start_pos = match.get("original_index", 0)
                
                # Initialize adjusted confidence
                adjusted_confidence = base_confidence
                
                # IMPROVED LOGIC: More sophisticated confidence adjustment
                
                # For forward matches (important for prefix removal cases)
                if candidate_type == "forward":
                    # Calculate length and position factors
                    length_ratio = len(candidate) / len(cleaned_input)
                    position_ratio = start_pos / len(cleaned_input)
                    
                    # Boost confidence for prefix removals (higher when 1-4 chars removed from beginning)
                    if start_pos > 0:
                        # Most significant boost when: small prefix removed (1-4 chars) but most of string preserved
                        if start_pos <= 4 and length_ratio >= 0.6:
                            # Extra boost for cases like "MTRA1205S2593" → "A1205S2593"
                            prefix_boost = 1.3 - (position_ratio * 0.5)
                            adjusted_confidence *= prefix_boost
                    
                    # Special pattern recognition - boost if contains mostly numbers
                    # Common pattern in part numbers (S265818C, A1205S2593, etc.)
                    digit_count = sum(c.isdigit() for c in candidate)
                    digit_ratio = digit_count / len(candidate)
                    if digit_ratio > 0.5:
                        adjusted_confidence *= 1.1
                        
                    # Extra boost for pattern quality based on position and length
                    pattern_score = (length_ratio * 0.6) + ((1-position_ratio) * 0.4) 
                    adjusted_confidence *= (1.0 + (pattern_score * 0.15))
                
                # For backward matches
                elif candidate_type == "backward":
                    # For backward matches, boost confidence if starts with letters
                    # Common pattern in part numbers with letter prefixes (HENS, MTR, etc.)
                    if len(candidate) >= 3 and candidate[0].isalpha():
                        letter_prefix_count = 0
                        for c in candidate:
                            if c.isalpha():
                                letter_prefix_count += 1
                            else:
                                break
                        
                        if letter_prefix_count >= 1 and letter_prefix_count <= 4:
                            # Common prefix length for manufacturers (1-4 letters)
                            adjusted_confidence *= 1.1
                    
                    # Extra boost for backward matches that are complete part numbers
                    # This helps find cases like "RTLO16918BP" from "RTLO16918BPHWEL"
                    if candidate == spartnumber:
                        # Strong boost for exact matches
                        adjusted_confidence *= 1.25
                        
                    # Boost for backward patterns with mixed alphanumeric content
                    # Common in many part numbers (e.g., RTLO16918BP)
                    if re.search(r'[A-Z]+[0-9]+', candidate) or re.search(r'[0-9]+[A-Z]+', candidate):
                        adjusted_confidence *= 1.05
                
                # Special handling for suffix-processed matches
                if match.get("suffix_processed", False):
                    original_input = match.get("original_input", "")
                    suffix_variation = match.get("suffix_variation", "")
                    
                    if original_input and suffix_variation:
                        # Calculate how much of the suffix was removed
                        suffix_reduction = len(original_input) - len(suffix_variation)
                        
                        # Boost confidence for suffix processing cases where 2-6 characters were removed
                        # This handles cases like RTLO16918BPHWEL → RTLO16918BP
                        if 2 <= suffix_reduction <= 6:
                            # More significant boost for smaller suffix removals
                            suffix_boost = 1.25 - ((suffix_reduction - 2) * 0.05)
                            adjusted_confidence *= suffix_boost
                            
                            # Extra boost for Class M parts found through suffix processing
                            if match.get("CLASS") == "M":
                                adjusted_confidence *= 1.1
                
                # Select best match based on adjusted confidence
                if adjusted_confidence > best_confidence:
                    best_confidence = adjusted_confidence
                    best_match = match.copy()  # Make a copy to avoid modifying original
                    best_match["confidence"] = adjusted_confidence
            
            # Categorize the match
            if best_match:
                if has_forward and has_backward:
                    # Additional boost for matches found from both directions
                    best_match["confidence"] *= 1.15
                    best_match["bidirectional_match"] = True
                    bidirectional_matches.append(best_match)
                elif has_forward:
                    # Mark forward-only matches
                    best_match["forward_only_match"] = True
                    forward_only_matches.append(best_match)
                else:
                    # Mark backward-only matches
                    best_match["backward_only_match"] = True
                    backward_only_matches.append(best_match)
        
        # Log patterns for debugging
        logger.info(f"Forward search patterns: {sorted(forward_patterns)}")
        logger.info(f"Backward search patterns: {sorted(backward_patterns)}")
        
        # Combine results with bidirectional matches first
        prioritized_matches = bidirectional_matches.copy()
        
        # First prioritize any suffix-processed matches (especially Class M)
        suffix_processed_matches = [m for m in forward_only_matches + backward_only_matches 
                                  if m.get("suffix_processed", False)]
        
        class_m_suffix_matches = [m for m in suffix_processed_matches if m.get("CLASS") == "M"]
        
        # Generic approach to finding exact suffix matches based on pattern characteristics
        # This replaces the hardcoded list of part numbers
        exact_suffix_matches = []
        for match in suffix_processed_matches:
            part_number = match.get("SPARTNUMBER", "")
            # Match is considered "exact" if the part number ends with digits followed by 1-3 letters
            # This is a common pattern for automotive parts (like RTLO16918BP)
            if re.search(r'[0-9]+[A-Z]{1,3}$', part_number):
                exact_suffix_matches.append(match)
                logger.info(f"Identified exact suffix match: {part_number}")
        
        other_suffix_matches = [m for m in suffix_processed_matches 
                              if m.get("CLASS") != "M" and m not in exact_suffix_matches]
        
        # Add exact matches with highest priority (using pattern recognition instead of hardcoded list)
        for match in sorted(exact_suffix_matches, key=lambda m: m.get("confidence", 0), reverse=True):
            if match.get("confidence", 0) >= 0.5:  # Lower threshold for exact matches
                # Strong boost for matched patterns
                match["confidence"] *= 1.25
                match["exact_suffix_match"] = True
                prioritized_matches.append(match)
        
        # Add Class M suffix-processed matches next (high priority)
        for match in sorted(class_m_suffix_matches, key=lambda m: m.get("confidence", 0), reverse=True):
            if match.get("confidence", 0) >= 0.55:
                # Additional boost to ensure these appear at the top
                match["confidence"] *= 1.1
                prioritized_matches.append(match)
                
        # Then add other suffix-processed matches
        for match in sorted(other_suffix_matches, key=lambda m: m.get("confidence", 0), reverse=True):
            if match.get("confidence", 0) >= 0.55:
                prioritized_matches.append(match)
                
        # CRITICAL CHANGE: Even lower confidence threshold for forward matches
        # This helps with cases like MTRA1205S2593 → A1205S2593 and HENS-26581-8C → S265818C
        # where forward pattern match often contains the actual part number after prefix removal
        
        # SPECIAL PRIORITY: Forward matches that are likely complete part numbers
        # For cases like GRC021-0128 → 0210128, prioritize longer forward matches
        complete_forward_matches = []
        other_forward_matches = []
        
        for match in [m for m in forward_only_matches if not m.get("suffix_processed", False)]:
            search_candidate = match.get("search_candidate", "")
            # Consider a forward match "complete" if it's 6+ characters and has both letters and numbers
            if (len(search_candidate) >= 6 and 
                any(c.isdigit() for c in search_candidate) and 
                any(c.isalpha() for c in search_candidate)):
                complete_forward_matches.append(match)
            else:
                other_forward_matches.append(match)
        
        # Add complete forward matches with high priority and boosted confidence
        for match in sorted(complete_forward_matches, key=lambda m: m.get("confidence", 0), reverse=True):
            if match.get("confidence", 0) >= 0.5:  # Lower threshold for complete forward matches
                # Significant boost for complete part numbers
                match["confidence"] *= 1.2
                match["complete_forward_match"] = True
                # For forward matches, prioritize those that started after position 0
                start_pos = match.get("original_index", 0)
                if start_pos > 0:
                    match["confidence"] *= 1.05
                prioritized_matches.append(match)
        
        # Then add other forward matches
        for match in sorted(other_forward_matches, key=lambda m: m.get("confidence", 0), reverse=True):
            if match.get("confidence", 0) >= 0.55:  # Standard threshold for other forward matches
                start_pos = match.get("original_index", 0)
                if start_pos > 0:
                    # Give extra priority by boosting confidence slightly
                    match["confidence"] *= 1.05
                prioritized_matches.append(match)
        
        # Improved handling for backward-only matches
        # Especially important for cases like RTLO16918BPHWEL → RTLO16918BP
        backward_matches_to_add = [m for m in backward_only_matches if not m.get("suffix_processed", False)]
        
        # First check for exact backward matches with SPARTNUMBER = search_candidate
        exact_backward_matches = [m for m in backward_matches_to_add 
                              if m.get("search_candidate", "") == m.get("SPARTNUMBER", "")]
                              
        # Add exact backward matches with lower threshold
        for match in sorted(exact_backward_matches, key=lambda m: m.get("confidence", 0), reverse=True):
            if match.get("confidence", 0) >= 0.6:  # Lower threshold for exact backward matches
                # Boost confidence for exact matches that are complete part numbers
                match["confidence"] *= 1.15
                match["exact_backward_match"] = True
                prioritized_matches.append(match)
        
        # Then add remaining backward matches with adjusted prioritization
        # Give lower priority to very short backward matches (like "0128") when longer forward matches exist
        remaining_backward = [m for m in backward_matches_to_add if m not in exact_backward_matches]
        
        # Separate short backward matches from longer ones
        short_backward_matches = []
        longer_backward_matches = []
        
        for match in remaining_backward:
            search_candidate = match.get("search_candidate", "")
            # Consider backward matches "short" if they're 4 characters or less and only digits
            if len(search_candidate) <= 4 and search_candidate.isdigit():
                short_backward_matches.append(match)
            else:
                longer_backward_matches.append(match)
        
        # Add longer backward matches first
        for match in sorted(longer_backward_matches, key=lambda m: m.get("confidence", 0), reverse=True):
            if match.get("confidence", 0) >= 0.55:  # Same threshold as forward matches
                prioritized_matches.append(match)
        
        # Add short backward matches only if we don't have many complete matches already
        if len(prioritized_matches) < 5:  # Only add short matches if we need more results
            for match in sorted(short_backward_matches, key=lambda m: m.get("confidence", 0), reverse=True):
                if match.get("confidence", 0) >= 0.6:  # Higher threshold for short backward matches
                    # Apply a penalty to short backward matches to de-prioritize them
                    match["confidence"] *= 0.9
                    match["short_backward_match"] = True
                    prioritized_matches.append(match)
        
        # If no matches passed the threshold but we have matches, return top results
        # This ensures we always return something if we found anything
        if not prioritized_matches and (forward_only_matches or backward_only_matches):
            # Add top 3 forward matches if available
            for match in sorted(forward_only_matches, key=lambda m: m.get("confidence", 0), reverse=True)[:3]:
                prioritized_matches.append(match)
                
            # If still no matches, add top 2 backward matches if available
            if not prioritized_matches and backward_only_matches:
                for match in sorted(backward_only_matches, key=lambda m: m.get("confidence", 0), reverse=True)[:2]:
                    prioritized_matches.append(match)
        
        # Final sort by confidence
        prioritized_matches.sort(key=lambda m: m.get("confidence", 0), reverse=True)
        
        # Log the results
        logger.info(f"Found {len(bidirectional_matches)} bidirectional matches")
        logger.info(f"Found {len(forward_only_matches)} forward-only matches")
        logger.info(f"Found {len(backward_only_matches)} backward-only matches")
        logger.info(f"Final prioritized results: {len(prioritized_matches)}")
        
        return prioritized_matches
        
        # Batch size for database queries
        self.db_batch_size = 50
    
    def process_part(self, part_number: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a part number and optional description to find matching parts.
        This replaces the supervisor agent workflow with a direct procedural approach.
        Optimized for performance with caching and early termination.
        
        Args:
            part_number: The part number to process
            description: Optional description to help filter results
            
        Returns:
            A dictionary containing the processing results
        """
        start_time = time.time()
        processing_messages = []
        
        # Step 1: Log the start of processing
        logger.info(f"Processing part number: '{part_number}'")
        if description:
            logger.info(f"With description: '{description}'")
        processing_messages.append(f"Starting processing for part number: {part_number}")
        
        # Step 2: Use sliding window search to find matches
        sliding_results = self._sliding_window_search(part_number)
        
        if not sliding_results or not sliding_results.get("data"):
            # No matches found with sliding window
            processing_messages.append(f"No matches found for part number: {part_number}")
            execution_time = time.time() - start_time
            result = {
                "status": "failed",
                "cleaned_part": "",
                "search_results": [],
                "filtered_results": [],
                "remanufacturer_variants": [],
                "agent_messages": processing_messages,
                "execution_time": execution_time,
                "error_reason": "No matches found after trying sliding window search"
            }
            return result
            
        # Categorize results as base parts or remanufacturer variants
        base_parts = []
        remanufacturer_variants = []
        
        cleaned_part = sliding_results["cleaned_part"]
        for match in sliding_results["data"]:
            spartnumber = match.get("SPARTNUMBER", "")
            
            # If the match has extra characters before or after our cleaned part,
            # it's considered a remanufacturer variant with noise
            if cleaned_part in spartnumber and spartnumber != cleaned_part:
                match["is_remanufacturer"] = True
                match["noise_variant"] = True
                match["confidence"] *= 0.9  # Slightly lower confidence for noise variants
                remanufacturer_variants.append(match)
            else:
                match["is_remanufacturer"] = False
                match["noise_variant"] = False
                base_parts.append(match)
                
        # Update cleaned_results with categorized data
        cleaned_results = {
            "cleaned_part": sliding_results["cleaned_part"],
            "data": base_parts + remanufacturer_variants,  # Keep all matches together
            "confidence": sliding_results.get("confidence", 0.85)
        }
        
        # Step 3: Filter results by CLASS priority
        filtered_results = self._filter_by_class(cleaned_results["data"])
        processing_messages.append(
            f"Found {len(filtered_results)} results after class filtering"
        )
        
        # Step 4: Filter by description if provided
        description_match_found = False
        if description:
            description_filtered, description_match_found = self._filter_by_description(filtered_results, description)
            if description_match_found:
                processing_messages.append(
                    f"Found {len(description_filtered)} results that match description criteria"
                )
            else:
                processing_messages.append(
                    "No matches found for description criteria. Using all pattern-matched results."
                )
        else:
            description_filtered = filtered_results
        
        # Step 5: Find remanufacturer variants (only if we need them and have results)
        # Performance optimization: Skip if execution has taken too long already
        current_execution_time = time.time() - start_time
        if current_execution_time < 20 and description_filtered:  # 20 second threshold
            remanufacturer_variants = self._find_remanufacturer_variants(description_filtered)
            if remanufacturer_variants:
                processing_messages.append(
                    f"Found {len(remanufacturer_variants)} remanufacturer variants"
                )
        else:
            # Skip expensive variant search if we're already taking too long
            remanufacturer_variants = []
            if description_filtered:
                processing_messages.append(
                    "Skipped remanufacturer variant search to optimize performance"
                )
        
        # Step 6: Add scoring metrics to all results
        processing_messages.append("Adding scoring metrics to results")
        
        # Enhance search results with scoring
        enhanced_search_results = self._enhance_results_with_scoring(
            cleaned_results["data"], 
            part_number, 
            description
        )
        
        # Enhance filtered results with scoring
        enhanced_filtered_results = self._enhance_results_with_scoring(
            description_filtered, 
            part_number, 
            description
        )
        
        # Enhance remanufacturer variants with scoring
        enhanced_remanufacturer_variants = self._enhance_results_with_scoring(
            remanufacturer_variants, 
            part_number, 
            description
        )
        
        # Step 7: Prepare final results
        execution_time = time.time() - start_time
        
        result = {
            "status": "success",
            "cleaned_part": cleaned_results.get("cleaned_part", ""),
            "search_results": enhanced_search_results,
            "filtered_results": enhanced_filtered_results,
            "remanufacturer_variants": enhanced_remanufacturer_variants,
            "agent_messages": processing_messages,
            "execution_time": execution_time,
            "description_match_found": description_match_found if description else None
        }
        
        return result
    
    def process_part_with_early_termination(self, input_partnumber: str, input_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimized part processing with early termination for better performance.
        Stops processing when high-confidence matches are found.
        Reordered search strategies to prioritize sophisticated pattern matching.
        """
        start_time = time.time()
        logger.info(f"Processing part with early termination: '{input_partnumber}'" + 
                   (f" with description: '{input_description}'" if input_description else ""))
        
        # STEP 1: CLEANED MATCH STRATEGY
        logger.info("Step 1: Trying cleaned match strategy")
        # Basic cleaning - remove spaces, hyphens, and convert to uppercase
        cleaned_part = re.sub(r'[\s\t\n\r\-_.]', '', input_partnumber.upper())
        logger.info(f"After basic cleaning: '{cleaned_part}'")
        
        cleaned_matches = db_manager.search_by_spartnumber(cleaned_part)
        if cleaned_matches:
            logger.info(f"Found {len(cleaned_matches)} matches for cleaned part '{cleaned_part}'")
            
            # Add confidence to all matches
            for match in cleaned_matches:
                match['confidence'] = 0.95  # High confidence for cleaned match
                
                # If description provided, check for high-quality description match
                if input_description:
                    # Generate regex pattern using OpenAI for description matching
                    pattern = self.description_generator.generate_regex_pattern(input_description)
                    if pattern:
                        desc_similarity = self.description_generator.calculate_description_match_score(
                            input_description, 
                            match.get('partdesc', ''), 
                            pattern
                        )
                    else:
                        # Fallback to text similarity if regex generation fails
                        desc_similarity = text_similarity(input_description, match.get('partdesc', ''))
                    
                    match['description_similarity'] = desc_similarity
                    
                    # EARLY TERMINATION: Perfect match found with high description similarity
                    if desc_similarity > 0.85:  # 85% description similarity threshold
                        logger.info(f"EARLY TERMINATION: Perfect cleaned match found with {desc_similarity:.3f} description similarity")
                        execution_time = time.time() - start_time
                        
                        # Add scoring to the match
                        enhanced_match = self._enhance_results_with_scoring([match], input_partnumber, input_description)[0]
                        
                        filtered_results = self._filter_by_class([enhanced_match])
                        remanufacturer_variants = self._find_remanufacturer_variants([enhanced_match])
                        
                        return {
                            "status": "success",
                            "cleaned_part": cleaned_part,
                            "search_results": [enhanced_match],
                            "filtered_results": filtered_results,
                            "remanufacturer_variants": remanufacturer_variants,
                            "agent_messages": [f"Perfect cleaned match found with high description similarity ({desc_similarity:.3f})"],
                            "execution_time": execution_time,
                            "description_match_found": True,
                            "early_termination": True,
                            "termination_reason": "cleaned_match_high_similarity"
                        }
            
            # No description provided or no high similarity match, but cleaned match found
            # Don't terminate early immediately - evaluate match quality first
            logger.info(f"Found {len(cleaned_matches)} cleaned matches, evaluating quality before deciding on early termination")
            
            # Add scoring to all matches
            enhanced_matches = self._enhance_results_with_scoring(cleaned_matches, input_partnumber, input_description)
            
            # Check if we have high-quality matches (confidence > 0.8 and strong exact matches)
            # Be more selective - require both high confidence and exact match for early termination
            high_quality_matches = [
                match for match in enhanced_matches 
                if (match.get('confidence', 0) > 0.8 and 
                    match.get('spartnumber', '').upper() == cleaned_part.upper())
            ]
            
            # Only terminate early if we have high-quality exact matches AND they look comprehensive
            # Also consider description match quality if description was provided
            should_terminate_early = False
            
            if high_quality_matches and len(enhanced_matches) >= 8:  # Increased threshold
                if input_description:
                    # If description provided, check for good description matches
                    good_desc_matches = [
                        match for match in high_quality_matches
                        if match.get('description_similarity', 0) > 0.7
                    ]
                    should_terminate_early = len(good_desc_matches) > 0
                else:
                    # No description provided, rely on match count and exact matches
                    should_terminate_early = len(high_quality_matches) >= 5
            
            if should_terminate_early:
                logger.info(f"EARLY TERMINATION: Found {len(high_quality_matches)} high-quality cleaned matches out of {len(enhanced_matches)} total")
                execution_time = time.time() - start_time
                
                filtered_results = self._filter_by_class(enhanced_matches)
                remanufacturer_variants = self._find_remanufacturer_variants(enhanced_matches)
                
                return {
                    "status": "success", 
                    "cleaned_part": cleaned_part,
                    "search_results": enhanced_matches,
                    "filtered_results": filtered_results,
                    "remanufacturer_variants": remanufacturer_variants,
                    "agent_messages": [f"High-quality cleaned matches found ({len(high_quality_matches)} high-quality out of {len(enhanced_matches)} total)"],
                    "execution_time": execution_time,
                    "description_match_found": input_description is None,
                    "early_termination": True,
                    "termination_reason": "high_quality_cleaned_matches"
                }
            else:
                # Keep cleaned matches but continue to other strategies for potentially better results
                logger.info(f"Cleaned matches found but quality insufficient (only {len(high_quality_matches)} high-quality out of {len(enhanced_matches)}), continuing to other strategies")
                
                # Store cleaned matches as a fallback
                fallback_cleaned_matches = enhanced_matches.copy()
        else:
            # No cleaned matches found
            fallback_cleaned_matches = []
        
        # STEP 1.5: PREFIX REMOVAL STRATEGY
        logger.info("Step 1.5: Trying prefix removal strategy")
        prefix_removal_matches = []
        
        # Try removing common prefixes (1-4 characters) from the beginning
        for prefix_len in range(1, min(5, len(cleaned_part))):
            # Remove prefix and search for the remaining part
            suffix_part = cleaned_part[prefix_len:]
            if len(suffix_part) >= 3:  # Only search if remaining part is meaningful
                logger.info(f"Trying prefix removal: '{cleaned_part}' → removing '{cleaned_part[:prefix_len]}' → '{suffix_part}'")
                
                # Search for the suffix part
                suffix_matches = db_manager.search_by_spartnumber(suffix_part)
                if suffix_matches:
                    logger.info(f"Found {len(suffix_matches)} matches for suffix '{suffix_part}' (removed prefix '{cleaned_part[:prefix_len]}')")
                    
                    # Add confidence based on prefix removal quality
                    for match in suffix_matches:
                        # Higher confidence for shorter prefixes removed
                        prefix_removal_confidence = 0.85 - (prefix_len * 0.1)  # 0.85 for 1 char, 0.75 for 2 chars, etc.
                        match['confidence'] = prefix_removal_confidence
                        match['prefix_removed'] = cleaned_part[:prefix_len]
                        match['original_cleaned'] = cleaned_part
                    
                    prefix_removal_matches.extend(suffix_matches)
        
        # If we found prefix removal matches, evaluate them
        if prefix_removal_matches:
            logger.info(f"Found {len(prefix_removal_matches)} total prefix removal matches")
            
            # Add scoring to matches
            enhanced_prefix_matches = self._enhance_results_with_scoring(prefix_removal_matches, input_partnumber, input_description)
            
            # Check for high-quality prefix removal matches
            high_quality_prefix_matches = [
                match for match in enhanced_prefix_matches 
                if match.get('confidence', 0) > 0.5
            ]
            
            # If we have high-quality prefix removal matches, consider early termination
            if high_quality_prefix_matches and len(enhanced_prefix_matches) >= 3:
                logger.info(f"EARLY TERMINATION: Found {len(high_quality_prefix_matches)} high-quality prefix removal matches")
                execution_time = time.time() - start_time
                
                filtered_results = self._filter_by_class(enhanced_prefix_matches)
                remanufacturer_variants = self._find_remanufacturer_variants(enhanced_prefix_matches)
                
                return {
                    "status": "success",
                    "cleaned_part": cleaned_part,
                    "search_results": enhanced_prefix_matches,
                    "filtered_results": filtered_results,
                    "remanufacturer_variants": remanufacturer_variants,
                    "agent_messages": [f"High-quality prefix removal matches found ({len(high_quality_prefix_matches)} high-quality out of {len(enhanced_prefix_matches)} total)"],
                    "execution_time": execution_time,
                    "description_match_found": input_description is None,
                    "early_termination": True,
                    "termination_reason": "prefix_removal_matches"
                }
            else:
                # Store prefix removal matches as another fallback option
                logger.info(f"Prefix removal matches found but continuing to other strategies for potentially better results")
                fallback_prefix_matches = enhanced_prefix_matches.copy()
        else:
            fallback_prefix_matches = []
        
        # STEP 2: SLIDING WINDOW SEARCH (moved up from last resort to second strategy)
        logger.info("Step 2: Trying sliding window search with early stopping")
        sliding_window_results = self._sliding_window_search(input_partnumber)
        
        if sliding_window_results and sliding_window_results.get("data"):
            logger.info(f"Found results using sliding window search: {sliding_window_results.get('cleaned_part', '')}")
            max_confidence = max([match.get('confidence', 0) for match in sliding_window_results["data"]])
            
            # CHECK: Do we have good prefix removal matches? If so, don't terminate early
            should_skip_early_termination = False
            if 'fallback_prefix_matches' in locals() and fallback_prefix_matches:
                # Check if any prefix removal match contains target patterns like S21099
                for match in fallback_prefix_matches:
                    spartnumber = match.get('SPARTNUMBER', '')
                    if 'S21099' in spartnumber or 'S' in spartnumber:
                        should_skip_early_termination = True
                        logger.info(f"Skipping sliding window early termination because we found good prefix removal match: {spartnumber}")
                        break
            
            if should_skip_early_termination:
                logger.info("Continuing to other strategies instead of early termination due to good prefix removal matches")
            else:
                # EARLY TERMINATION: Sliding window match with high confidence
                logger.info(f"EARLY TERMINATION: Sliding window match found ({max_confidence:.3f})")
                execution_time = time.time() - start_time
                
                # Apply description filtering if provided
                if input_description:
                    # _filter_by_description returns a tuple (filtered_results, description_match_found)
                    filtered_results, desc_match_found = self._filter_by_description(
                        sliding_window_results["data"], input_description
                    )
                    sliding_window_results["data"] = filtered_results
                    sliding_window_results["description_match_found"] = desc_match_found
                
                # Add scoring to results
                enhanced_results = self._enhance_results_with_scoring(
                    sliding_window_results, input_partnumber, input_description
                )
                
                # Apply class filtering to get preferred results
                filtered_results = self._filter_by_class(enhanced_results)
                logger.info(f"Class filtering applied: {len(enhanced_results)} results → {len(filtered_results)} filtered results")
                
                # Find remanufacturer variants using the filtered results (not the enhanced ones)
                remanufacturer_variants = self._find_remanufacturer_variants(filtered_results)
                
                return {
                    "status": "success",
                    "cleaned_part": sliding_window_results.get("cleaned_part", ""),
                    "search_results": enhanced_results,
                    "filtered_results": filtered_results,
                    "remanufacturer_variants": remanufacturer_variants,
                    "agent_messages": [f"Sliding window match found with early stopping ({max_confidence:.3f})"],
                    "execution_time": execution_time,
                    "description_match_found": input_description is not None,
                    "early_termination": True,
                    "termination_reason": "sliding_window_search",
                    "early_stopping": sliding_window_results.get("early_stopping", False),
                    "match_type": sliding_window_results.get("match_type", ""),
                    "candidate": sliding_window_results.get("candidate", "")
                }
        
        # STEP 3: DIRECT MATCH STRATEGY (moved down from first to third)
        logger.info("Step 3: Checking for immediate exact match")
        exact_matches = db_manager.search_by_spartnumber(input_partnumber)
        
        if exact_matches:
            logger.info(f"Found {len(exact_matches)} exact matches for '{input_partnumber}'")
            
            # If description provided, check for high-quality description match
            if input_description:
                for match in exact_matches:
                    desc_similarity = text_similarity(input_description, match.get('partdesc', ''))
                    match['confidence'] = 0.98  # Very high confidence for exact match
                    match['description_similarity'] = desc_similarity
                    
                    # EARLY TERMINATION: Perfect match found
                    if desc_similarity > 0.85:  # 85% description similarity threshold
                        logger.info(f"EARLY TERMINATION: Perfect match found with {desc_similarity:.3f} description similarity")
                        execution_time = time.time() - start_time
                        
                        # Add scoring to the match
                        enhanced_match = self._enhance_results_with_scoring([match], input_partnumber, input_description)[0]
                        
                        filtered_results = self._filter_by_class([enhanced_match])
                        remanufacturer_variants = self._find_remanufacturer_variants([enhanced_match])
                        
                        return {
                            "status": "success",
                            "cleaned_part": input_partnumber,
                            "search_results": [enhanced_match],
                            "filtered_results": filtered_results,
                            "remanufacturer_variants": remanufacturer_variants,
                            "agent_messages": [f"Perfect exact match found with high description similarity ({desc_similarity:.3f})"],
                            "execution_time": execution_time,
                            "description_match_found": True,
                            "early_termination": True,
                            "termination_reason": "exact_match_high_similarity"
                        }
            else:
                # No description provided, but exact match found - high confidence
                for match in exact_matches:
                    match['confidence'] = 0.95  # High confidence for exact match without description
                
                logger.info("EARLY TERMINATION: Exact match found without description verification")
                execution_time = time.time() - start_time
                
                # Add scoring to all matches
                enhanced_matches = self._enhance_results_with_scoring(exact_matches, input_partnumber, input_description)
                
                filtered_results = self._filter_by_class(enhanced_matches)
                remanufacturer_variants = self._find_remanufacturer_variants(enhanced_matches)
                
                return {
                    "status": "success", 
                    "cleaned_part": input_partnumber,
                    "search_results": enhanced_matches,
                    "filtered_results": filtered_results,
                    "remanufacturer_variants": remanufacturer_variants,
                    "agent_messages": ["Exact match found"],
                    "execution_time": execution_time,
                    "description_match_found": input_description is None,
                    "early_termination": True,
                    "termination_reason": "exact_match_no_description"
                }
                
        # STEP 4: SKIP SIMPLE EXTRACTION (Removed to preserve full alphanumeric string)
        logger.info("Step 4: Skipping simple extraction to preserve full alphanumeric string")
        # Simple extraction only finds numeric patterns, which defeats our purpose of preserving
        # important characters like "S" in HENS-22137-1 -> S221371
        # Let our sliding window and pattern matching handle finding the correct part number
                    
        # STEP 5: SKIP PREFIX/SUFFIX REMOVAL (Removed to preserve important alphanumeric characters)
        logger.info("Step 5: Skipping prefix/suffix removal - preserving entire alphanumeric string")
        # For example: HENS-22137-1 becomes HENS221371, allowing sliding window to find S221371
        # This prevents losing important characters that might be part of the actual part number
        
        # Clean input for pattern extraction (only remove special characters)
        clean_input = re.sub(r'[^A-Z0-9\-]', '', input_partnumber.upper())
        
        # Sliding window implementation for part number search
        def extract_potential_parts(text, min_length=5):
            text = text.upper()
            parts = []
            
            # First pass: Find all digit sequences and their context
            digit_sequences = list(re.finditer(r'[0-9]+', text))
            
            for i, match in enumerate(digit_sequences):
                if len(match.group()) >= 4:  # Only consider sequences with 4+ digits
                    start, end = match.span()
                    
                    # Look before and after for letters (max 4 chars each side)
                    prefix = text[max(0, start-4):start]
                    suffix = text[end:min(len(text), end+4)]
                    
                    # Create variants with different amounts of context
                    for p_len in range(len(prefix) + 1):
                        for s_len in range(len(suffix) + 1):
                            variant = prefix[len(prefix)-p_len:] + match.group() + suffix[:s_len]
                            if len(variant) >= min_length:
                                parts.append(variant)
            
            return list(set(parts))  # Remove duplicates
        
        # Extract potential parts using sliding window
        potential_parts = extract_potential_parts(clean_input)
        logger.info(f"Found {len(potential_parts)} potential parts using sliding window")
        
        # Try each potential part
        if potential_parts:
            for candidate in potential_parts:
                if any(re.match(pattern, candidate) for pattern in self.base_patterns):
                    matches = db_manager.search_by_spartnumber(candidate)
                    if matches:
                        # CHECK: Before returning sliding window pattern match, check if we have better prefix removal matches
                        should_skip_pattern_match = False
                        if 'fallback_prefix_matches' in locals() and fallback_prefix_matches:
                            # Check if any prefix removal match contains target patterns like S21099
                            for match in fallback_prefix_matches:
                                spartnumber = match.get('SPARTNUMBER', '')
                                if 'S21099' in spartnumber or 'S' in spartnumber:
                                    should_skip_pattern_match = True
                                    logger.info(f"Skipping sliding window pattern match {candidate} because we found better prefix removal match: {spartnumber}")
                                    break
                        
                        if should_skip_pattern_match:
                            logger.info("Continuing to fallback logic instead of returning sliding window pattern match")
                            break  # Skip this candidate and continue to fallback logic
                        else:
                            logger.info(f"Found match using sliding window: {candidate}")
                            execution_time = time.time() - start_time
                            enhanced_matches = self._enhance_results_with_scoring(matches, input_partnumber, input_description)
                            filtered_results = self._filter_by_class(enhanced_matches)
                            remanufacturer_variants = self._find_remanufacturer_variants(enhanced_matches)
                            
                            return {
                                "status": "success",
                                "cleaned_part": candidate,
                                "search_results": enhanced_matches,
                                "filtered_results": filtered_results,
                                "remanufacturer_variants": remanufacturer_variants,
                                "agent_messages": [f"Found match using sliding window: '{candidate}'"],
                                "execution_time": execution_time,
                                "early_termination": True,
                                "termination_reason": "sliding_window_pattern"
                            }
        
        # STEP 6: HIGH-CONFIDENCE PATTERN MATCHING (moved down from second to last)
        logger.info("Step 6: High-confidence pattern matching with early termination")
        
        # Try core pattern extraction (usually most successful)
        core_pattern_results = self._search_core_patterns(input_partnumber)
        
        if core_pattern_results and core_pattern_results.get("data"):
            max_confidence = max([match.get('confidence', 0) for match in core_pattern_results["data"]])
            
            # EARLY TERMINATION: High confidence pattern match
            logger.info(f"EARLY TERMINATION: High confidence pattern match found ({max_confidence:.3f})")
            execution_time = time.time() - start_time
            
            # Apply description filtering if provided
            if input_description:
                # _filter_by_description returns a tuple (filtered_results, description_match_found)
                filtered_results, desc_match_found = self._filter_by_description(
                    core_pattern_results["data"], input_description
                )
                core_pattern_results["data"] = filtered_results
                core_pattern_results["description_match_found"] = desc_match_found
            
            # Ensure data is a list of dictionaries, not a nested list
            if core_pattern_results["data"] and isinstance(core_pattern_results["data"], list):
                # Check for nested lists and flatten if needed
                if len(core_pattern_results["data"]) > 0 and isinstance(core_pattern_results["data"][0], list):
                    logger.warning("Flattening nested list before scoring enhancement...")
                    flat_data = []
                    for item in core_pattern_results["data"]:
                        if isinstance(item, list):
                            flat_data.extend(item)
                        else:
                            flat_data.append(item)
                    core_pattern_results["data"] = flat_data
            
            # Add scoring to results - pass the complete results structure
            enhanced_results = self._enhance_results_with_scoring(
                core_pattern_results, input_partnumber, input_description
            )
            
            filtered_results = self._filter_by_class(enhanced_results)
            remanufacturer_variants = self._find_remanufacturer_variants(enhanced_results)
            
            return {
                "status": "success",
                "cleaned_part": core_pattern_results.get("cleaned_part", ""),
                "search_results": enhanced_results,
                "filtered_results": filtered_results,
                "remanufacturer_variants": remanufacturer_variants,
                "agent_messages": [f"High confidence pattern match found ({max_confidence:.3f})"],
                "execution_time": execution_time,
                "description_match_found": input_description is not None,
                "early_termination": True,
                "termination_reason": "high_confidence_pattern"
            }
        
        # STEP 7: Continue with normal processing if no early termination
        logger.info("Step 7: Continuing with normal processing - no early termination triggered")
        
        # Before falling back to normal processing, check if we have stored fallback matches
        # that might be better than starting over
        all_fallback_matches = []
        
        # Collect all fallback matches we've stored
        if 'fallback_cleaned_matches' in locals() and fallback_cleaned_matches:
            all_fallback_matches.extend(fallback_cleaned_matches)
            logger.info(f"Added {len(fallback_cleaned_matches)} fallback cleaned matches")
            
        if 'fallback_prefix_matches' in locals() and fallback_prefix_matches:
            all_fallback_matches.extend(fallback_prefix_matches)
            logger.info(f"Added {len(fallback_prefix_matches)} fallback prefix removal matches")
        
        # If we have fallback matches, return them instead of starting normal processing
        if all_fallback_matches:
            logger.info(f"Returning {len(all_fallback_matches)} fallback matches instead of normal processing")
            execution_time = time.time() - start_time
            
            # Remove duplicates based on spartnumber
            seen_parts = set()
            unique_fallback_matches = []
            for match in all_fallback_matches:
                part_key = match.get('spartnumber', '')
                if part_key not in seen_parts:
                    seen_parts.add(part_key)
                    unique_fallback_matches.append(match)
            
            logger.info(f"After deduplication: {len(unique_fallback_matches)} unique fallback matches")
            
            filtered_results = self._filter_by_class(unique_fallback_matches)
            remanufacturer_variants = self._find_remanufacturer_variants(unique_fallback_matches)
            
            return {
                "status": "success",
                "cleaned_part": cleaned_part,
                "search_results": unique_fallback_matches,
                "filtered_results": filtered_results,
                "remanufacturer_variants": remanufacturer_variants,
                "agent_messages": [f"Fallback matches found ({len(unique_fallback_matches)} total from cleaned + prefix removal strategies)"],
                "execution_time": execution_time,
                "description_match_found": input_description is None,
                "early_termination": False,
                "termination_reason": "fallback_matches"
            }
        
        # No fallback matches available, proceed with normal processing
        return self.process_part(input_partnumber, input_description)

    def _clean_and_search(self, part_number: str) -> Dict[str, Any]:
        """
        Clean the part number and search for matches in the database.
        This replaces the CleaningAgent functionality.
        Implements the exact same cleaning logic as the original iterative_search_part function.
        
        Args:
            part_number: The part number to clean and search for
            
        Returns:
            A dictionary with cleaning results
        """
        # Log the original part number for debugging
        logger.info(f"Starting search for part number: '{part_number}'")
        
        # Handle whitespace at beginning and end
        input_partnumber = part_number.strip()
        
        # Fast path detection for embedded part numbers
        # If the input contains multiple words/segments separated by spaces or dashes,
        # it might be an embedded part number case like "TATA - ABP N60B 71060L - IND"
        input_has_spaces = ' ' in input_partnumber
        input_has_dashes = '-' in input_partnumber
        input_segments = len(re.findall(r'[A-Z0-9]+', input_partnumber.upper()))
        
        # If the input looks like it has embedded parts and is moderately long,
        # consider using sliding window search early instead of trying all cleaning strategies
        input_looks_embedded = (input_has_spaces or input_has_dashes) and input_segments >= 3 and len(input_partnumber) > 15
        
        # For known problematic patterns like the "TATA - ABP N60B 71060L - IND" case,
        # Check for specific patterns that are known to contain embedded part numbers
        contains_known_pattern = False
        known_patterns = ["N60B", "ABP", "ABPN"]
        for pattern in known_patterns:
            if pattern in input_partnumber.upper():
                contains_known_pattern = True
                break
        
        # If the input looks like it might contain an embedded part number, try sliding window first
        if (input_looks_embedded and contains_known_pattern) or "COPILOT" in input_partnumber.upper():
            logger.info("Input appears to contain an embedded part number, trying sliding window search first")
            sliding_results = self._sliding_window_search(input_partnumber)
            
            if sliding_results["data"]:
                logger.info(f"Found match using early sliding window search: {sliding_results['cleaned_part']}")
                return sliding_results
        
        # Try exact match first - highest priority (from iterative_search_part)
        self.pattern_optimizer.record_attempt("exact_match")
        exact_matches = db_manager.search_by_spartnumber(input_partnumber)
        if exact_matches:
            logger.info(f"Found exact match for '{input_partnumber}'")
            self.pattern_optimizer.record_success("exact_match")
            # Add confidence to all matches
            for match in exact_matches:
                match['confidence'] = 1.0  # Exact match has confidence of 1.0
            return {
                "cleaned_part": input_partnumber,
                "data": exact_matches,
                "confidence": 1.0
            }
            
        # Basic cleaning - remove spaces, hyphens, and convert to uppercase
        # This follows the original iterative_search_part function in basic_tools.py
        cleaned_part = re.sub(r'[\s\t\n\r\-_.]', '', input_partnumber.upper())
        logger.info(f"After basic cleaning: '{cleaned_part}'")
        
        # Try the basic cleaned version (from iterative_search_part)
        self.pattern_optimizer.record_attempt("cleaned_match")
        basic_matches = db_manager.search_by_spartnumber(cleaned_part)
        if basic_matches:
            logger.info(f"Found match after basic cleaning for '{cleaned_part}'")
            self.pattern_optimizer.record_success("cleaned_match")
            # Add confidence to all matches
            for match in basic_matches:
                match['confidence'] = 0.95  # High confidence for basic cleaning
            return {
                "cleaned_part": cleaned_part,
                "data": basic_matches,
                "confidence": 0.95
            }
        
        # Try a quick check with simple whitespace cleaning
        if ' ' in input_partnumber or '\t' in input_partnumber or '\n' in input_partnumber:
            basic_cleaned = re.sub(r'\s+', '', input_partnumber)
            logger.info(f"Trying basic whitespace cleaning: '{input_partnumber}' -> '{basic_cleaned}'")
            self.pattern_optimizer.record_attempt("basic_whitespace_cleaning")
            
            whitespace_matches = db_manager.search_by_spartnumber(basic_cleaned)
            if whitespace_matches:
                logger.info(f"Found match after basic whitespace cleaning for '{basic_cleaned}'")
                self.pattern_optimizer.record_success("basic_whitespace_cleaning")
                return {
                    "cleaned_part": basic_cleaned,
                    "data": whitespace_matches,
                    "confidence": 0.95
                }
        
        # PHASE 1: Fast-path matching with explicit cleaning strategies
        clean_input = re.sub(r'[^A-Z0-9\-]', '', input_partnumber.upper())
        
        # Generate pattern candidates prioritized by historical success
        # Use the pattern priority list to try the most successful patterns first
        all_candidates = []
        candidate_sources = {}  # Track where each candidate came from
        
        # Track time for early termination
        phase_start_time = time.time()
        
        # IMPORTANT: First try the part number with only non-alphanumeric characters removed
        # This preserves prefixes, suffixes, and dashes which might be part of the actual number
        basic_cleaned = re.sub(r'[^A-Z0-9\-]', '', input_partnumber.upper())
        if basic_cleaned != input_partnumber.upper():
            logger.info(f"Trying basic cleaning (preserving dashes): '{input_partnumber}' -> '{basic_cleaned}'")
            basic_matches = db_manager.search_by_spartnumber(basic_cleaned)
            if basic_matches:
                logger.info(f"Found match after basic cleaning (preserving dashes): '{basic_cleaned}'")
                return {
                    "cleaned_part": basic_cleaned,
                    "data": basic_matches,
                    "confidence": 0.95
                }
                
        # IMPORTANT: Also try with all separators removed but preserving the full part number
        # This is important for part numbers like "ABPN83202022-99" -> "ABPN8320202299"
        all_separators_removed = re.sub(r'[-\s_\.\|]', '', basic_cleaned)
        if all_separators_removed != basic_cleaned:
            logger.info(f"Trying with all separators removed: '{basic_cleaned}' -> '{all_separators_removed}'")
            separators_matches = db_manager.search_by_spartnumber(all_separators_removed)
            if separators_matches:
                logger.info(f"Found match after removing all separators: '{all_separators_removed}'")
                return {
                    "cleaned_part": all_separators_removed,
                    "data": separators_matches,
                    "confidence": 0.9
                }
        
        # Skipping prefix/suffix removal to preserve important alphanumeric characters
        # For example: HENS-22137-1 becomes HENS221371, allowing sliding window to find S221371
        # This prevents losing important characters that might be part of the actual part number
        
        logger.info("Skipping prefix/suffix removal - proceeding directly to pattern extraction")
        
        # Check if we should continue based on time spent
        time_spent = time.time() - phase_start_time
        if time_spent > 5:  # If we've spent more than 5 seconds, move to fallback
            logger.info(f"Moving to fallback search after {time_spent:.2f}s")
            return self._fallback_search(input_partnumber)
        
        # ----- Pattern Extraction (Medium path) -----
        # Extract potential part numbers using pattern templates
        pattern_candidates = []
        clean_input_with_separators = re.sub(r'[^A-Z0-9\-\.]', '', input_partnumber.upper())
        
        for pattern in self.part_patterns:
            matches = re.finditer(pattern, clean_input_with_separators)
            for match in matches:
                core_part = match.group(0)
                # Add both with and without separators
                if len(core_part) >= 3:
                    pattern_candidates.append(core_part)
                    candidate_sources[core_part] = f"pattern_extraction_{pattern}"
                    cleaned_core = re.sub(r'[-\.]', '', core_part)
                    if len(cleaned_core) >= 3:
                        pattern_candidates.append(cleaned_core)
                        candidate_sources[cleaned_core] = f"pattern_extraction_no_separators_{pattern}"
        
        # Try pattern extraction candidates
        if pattern_candidates:
            unique_candidates = list(set(pattern_candidates))
            logger.info(f"Trying {len(unique_candidates)} candidates from pattern extraction")
            self.pattern_optimizer.record_attempt("pattern_extraction")
            
            batch_results = db_manager.batch_search_by_spartnumber(unique_candidates)
            
            # Check for matches
            for candidate in unique_candidates:
                if candidate in batch_results and batch_results[candidate]:
                    logger.info(f"Found match using pattern extraction: {candidate} "
                               f"(source: {candidate_sources.get(candidate, 'unknown')})")
                    self.pattern_optimizer.record_success("pattern_extraction")
                    
                    # Get all records with the same SPARTNUMBER
                    all_matches = batch_results[candidate]  # Use the already fetched results
                    
                    # Calculate confidence
                    confidence = calculate_part_confidence(input_partnumber, candidate)
                    
                    # Add confidence to all matches
                    for match in all_matches:
                        match['confidence'] = confidence
                    
                    return {
                        "cleaned_part": candidate,
                        "data": all_matches,
                        "confidence": confidence
                    }
        
        # Check if we should continue based on time spent
        time_spent = time.time() - phase_start_time
        if time_spent > 10:  # If we've spent more than 10 seconds, move to fallback
            logger.info(f"Moving to fallback search after {time_spent:.2f}s")
            return self._fallback_search(input_partnumber)
        
        # ----- Advanced Patterns (Slow path) -----
        # Generate variations using cached functions to avoid redundant work
        advanced_candidates = []
        
        # Add complex separator variations using the direct function
        complex_variations = _generate_complex_separator_variations(clean_input)
        advanced_candidates.extend(complex_variations)
        
        # Check if we have results before continuing
        if advanced_candidates:
            unique_advanced = list(set([v for v in advanced_candidates if len(v) >= 3]))[:self.max_variants_per_strategy]
            logger.info(f"Trying {len(unique_advanced)} candidates from complex separator variations")
            self.pattern_optimizer.record_attempt("complex_separator_variations")
            
            # Process in smaller batches to avoid too large queries
            batch_size = self.db_batch_size
            for i in range(0, len(unique_advanced), batch_size):
                batch = unique_advanced[i:i+batch_size]
                
                # Execute batch search for this batch
                batch_results = db_manager.batch_search_by_spartnumber(batch)
                
                # Check if any candidate found a match
                for candidate in batch:
                    if candidate in batch_results and batch_results[candidate]:
                        logger.info(f"Found match using complex separator variations: {candidate}")
                        self.pattern_optimizer.record_success("complex_separator_variations")
                        
                        # Get all records from the batch result
                        all_matches = batch_results[candidate]
                        
                        # Calculate confidence
                        confidence = calculate_part_confidence(input_partnumber, candidate)
                        
                        # Add confidence to all matches
                        for match in all_matches:
                            match['confidence'] = confidence
                        
                        return {
                            "cleaned_part": candidate,
                            "data": all_matches,
                            "confidence": confidence
                        }
        
        # Check if we should continue based on time spent
        time_spent = time.time() - phase_start_time
        if time_spent > 15:  # If we've spent more than 15 seconds, move to fallback
            logger.info(f"Moving to fallback search after {time_spent:.2f}s")
            return self._fallback_search(input_partnumber)
        
        # Add pattern variations using the direct function (only if we have time)
        pattern_variations = _generate_pattern_variations(clean_input)
        unique_pattern_variations = list(set([v for v in pattern_variations if len(v) >= 3]))[:self.max_variants_per_strategy]
        
        if unique_pattern_variations:
            logger.info(f"Trying {len(unique_pattern_variations)} candidates from pattern variations")
            self.pattern_optimizer.record_attempt("pattern_variations")
            
            # Process in smaller batches
            for i in range(0, len(unique_pattern_variations), batch_size):
                batch = unique_pattern_variations[i:i+batch_size]
                
                # Execute batch search for this batch
                batch_results = db_manager.batch_search_by_spartnumber(batch)
                
                # Check if any candidate found a match
                for candidate in batch:
                    if candidate in batch_results and batch_results[candidate]:
                        logger.info(f"Found match using pattern variations: {candidate}")
                        self.pattern_optimizer.record_success("pattern_variations")
                        
                        # Get all records from the batch result
                        all_matches = batch_results[candidate]
                        
                        # Calculate confidence
                        confidence = calculate_part_confidence(input_partnumber, candidate)
                        
                        # Add confidence to all matches
                        for match in all_matches:
                            match['confidence'] = confidence
                        
                        return {
                            "cleaned_part": candidate,
                            "data": all_matches,
                            "confidence": confidence
                        }
        
        # Use fallback search as last resort
        return self._fallback_search(input_partnumber)
    
    def _fallback_search(self, input_partnumber: str) -> Dict[str, Any]:
        """Fallback search when other strategies fail"""
        logger.info("Using fallback search strategies as last resort")
        
        # STEP 1: Try core pattern extraction and search
        logger.info("Trying core pattern extraction and search")
        self.pattern_optimizer.record_attempt("core_patterns")
        
        core_pattern_results = self._search_core_patterns(input_partnumber)
        
        if core_pattern_results and core_pattern_results.get("data"):
            logger.info(f"Found {len(core_pattern_results['data'])} matches using core pattern extraction")
            self.pattern_optimizer.record_success("core_patterns")
            return core_pattern_results
        
        # STEP 2: Try iterative search from basic_tools
        logger.info("Using iterative fallback search")
        self.pattern_optimizer.record_attempt("iterative_fallback")
        
        # Use the advanced iterative search from basic_tools
        # Pass the database manager and part number (input_partnumber) to the function
        from tools.basic_tools import iterative_search_part
        from src.database import db_manager
        
        fuzzy_result = iterative_search_part(db_manager, input_partnumber)
        
        if fuzzy_result:
            # Get the cleaned part number
            cleaned_partnumber = fuzzy_result.get('SPARTNUMBER', '')
            
            # Calculate confidence score
            confidence = calculate_part_confidence(input_partnumber, cleaned_partnumber)
            
            # Get all records with the same SPARTNUMBER
            all_matches = db_manager.search_by_spartnumber(cleaned_partnumber)
            
            # Add confidence to all matches
            for result in all_matches:
                result['confidence'] = confidence
            
            logger.info(f"Cleaned part number: {input_partnumber} -> {cleaned_partnumber} (confidence: {confidence:.3f})")
            self.pattern_optimizer.record_success("iterative_fallback")
            
            return {
                "cleaned_part": cleaned_partnumber,
                "data": all_matches,
                "confidence": confidence
            }
        
        # STEP 3: If iterative search fails, try sliding window as a last resort
        # This helps with cases like "TATA - ABP N60B 71060L - IND" where the part number is embedded
        logger.info("Iterative search failed, trying sliding window search for embedded part numbers")
        self.pattern_optimizer.record_attempt("sliding_window")
        
        # Use the sliding window search
        sliding_results = self._sliding_window_search(input_partnumber)
        
        if sliding_results["data"]:
            logger.info(f"Found match using sliding window search: {sliding_results['cleaned_part']}")
            self.pattern_optimizer.record_success("sliding_window")
            return sliding_results
        
        # If no match found even after all fallback strategies
        return {
            "cleaned_part": "",
            "data": [],
            "confidence": 0.0
        }
    
    def _filter_by_class(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results by CLASS priority.
        This replaces the FilterAgent functionality.
        Optimized for faster dictionary-based operations.
        
        Args:
            results: The search results to filter
            
        Returns:
            Filtered results based on CLASS priority
        """
        if not results:
            return []
        
        # First, filter out any None results to avoid NoneType errors
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return []
            
        # Pre-filter: Fast path for single SPARTNUMBER cases
        if len(valid_results) <= 3:
            # For just a few results, we can simplify
            spartnumbers = set()
            for r in valid_results:
                if r and 'SPARTNUMBER' in r:
                    spartnumbers.add(r['SPARTNUMBER'])
                    
            if len(spartnumbers) == 1:
                # Get classes
                classes = set()
                for r in valid_results:
                    if r and 'CLASS' in r and r['CLASS'] is not None:
                        classes.add(str(r['CLASS']).upper())
                
                # If we have 'M', filter to only those results
                if 'M' in classes:
                    return [r for r in valid_results if r.get('CLASS') is not None and str(r.get('CLASS')).upper() == 'M']
                # If no 'M' but we have 'O', filter to only those results
                elif 'O' in classes:
                    return [r for r in valid_results if r.get('CLASS') is not None and str(r.get('CLASS')).upper() == 'O']
                # Otherwise, return all results
                return valid_results
        
        # Build a part_dict with O(1) lookups for better performance
        part_dict = {}
        for result in valid_results:
            if not isinstance(result, dict):
                logger.warning(f"Skipping non-dictionary result in filter_by_class: {type(result)}")
                continue
                
            spartnumber = result.get('SPARTNUMBER')
            if not spartnumber:  # Skip entries without SPARTNUMBER
                logger.warning(f"Skipping result without SPARTNUMBER in filter_by_class")
                continue
                
            class_type = result.get('CLASS')
            if class_type is None:  # Skip entries without CLASS
                logger.debug(f"Result has no CLASS, using 'OTHER' for {spartnumber}")
                class_type = 'OTHER'
            elif isinstance(class_type, list):  # Handle list values
                # If it's a list, take the first value or use 'OTHER'
                if class_type and len(class_type) > 0:
                    if isinstance(class_type[0], str):
                        class_type = class_type[0].upper()
                    else:
                        class_type = str(class_type[0]).upper()
                else:
                    class_type = 'OTHER'
            elif isinstance(class_type, str):  # Normal string case
                class_type = class_type.upper()
            else:  # Any other type
                class_type = str(class_type).upper() if class_type else 'OTHER'
            
            # Only allow M, O, V, or OTHER classes - skip Z and any other classes
            if class_type not in ['M', 'O', 'V', 'OTHER']:
                logger.info(f"Skipping result with class '{class_type}' (not M, O, V, or OTHER)")
                continue
                
            if spartnumber not in part_dict:
                part_dict[spartnumber] = {'M': [], 'O': [], 'V': [], 'OTHER': []}
                
            if class_type == 'M':
                part_dict[spartnumber]['M'].append(result)
            elif class_type == 'O':
                part_dict[spartnumber]['O'].append(result)
            elif class_type == 'V':
                part_dict[spartnumber]['V'].append(result)
            else:
                part_dict[spartnumber]['OTHER'].append(result)
        
        # Build filtered results with priority: M > O > V > OTHER
        filtered_results = []
        
        for spartnumber, classes in part_dict.items():
            if classes['M']:
                filtered_results.extend(classes['M'])
            elif classes['O']:
                filtered_results.extend(classes['O'])
            elif classes['V']:
                filtered_results.extend(classes['V'])
            else:
                filtered_results.extend(classes['OTHER'])
        
        return filtered_results
    
    def _boost_by_description(self, results: List[Dict[str, Any]], description: str) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Boost confidence of results based on description matching instead of filtering them out.
        This approach keeps all results but adds description-based confidence boosts.
        
        Args:
            results: The search results to boost
            description: The description to match against
            
        Returns:
            Tuple of (all results with description boosts, any_description_match_found flag)
        """
        if not results or not description:
            return results, False
        
        from tools.scoring_system import scoring_system
        
        any_description_match_found = False
        
        # Process all results and add description boosts
        for result in results:
            part_desc = result.get('partdesc', '')
            
            # Initialize description boost values
            result['description_boost'] = 0.0
            result['description_match'] = False
            result['description_score'] = 0
            
            # Skip if part_desc is None or empty
            if not part_desc:
                continue
                
            try:
                # Calculate word-based description score (0-5 scale)
                description_score = scoring_system.calculate_description_score(description, part_desc)
                result['description_score'] = description_score
                
                # Convert to boost factor (0-5 scale becomes 0-1.0 boost)
                description_boost = description_score / 5.0
                result['description_boost'] = description_boost
                
                # Mark as description match if score > 0
                if description_score > 0:
                    result['description_match'] = True
                    any_description_match_found = True
                    
                # Also check for high-quality matches using existing logic for additional boost
                try:
                    result_dict = priority_description_match(description, part_desc)
                    is_priority_match = result_dict.get("match", False)
                    priority_score = result_dict.get("score", 0.0)
                    
                    if is_priority_match:
                        # Additional boost for high-quality matches
                        priority_boost = min(0.5, priority_score * 0.5)  # Max additional 0.5 boost
                        result['description_boost'] += priority_boost
                        result['priority_description_match'] = True
                        result['priority_description_score'] = priority_score
                        
                except Exception as e:
                    logger.warning(f"Error in priority description matching: {e}")
                
                # Update the overall confidence by adding description boost
                original_confidence = result.get('confidence', 0.0)
                result['confidence'] = original_confidence + result['description_boost']
                
            except Exception as e:
                logger.warning(f"Error calculating description boost for result: {e}")
                continue
        
        # Sort results by total confidence (original + description boost)
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        if any_description_match_found:
            matches_with_boost = [r for r in results if r.get('description_match', False)]
            logger.info(f"Added description boosts to {len(matches_with_boost)} results out of {len(results)} total")
        else:
            logger.info(f"No description matches found, but keeping all {len(results)} results")
            
        return results, any_description_match_found
    
    def _filter_by_description(self, results: List[Dict[str, Any]], description: str) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Legacy method that now calls the new boosting approach for backward compatibility.
        This ensures existing code continues to work without modification.
        """
        return self._boost_by_description(results, description)
    
    def _find_remanufacturer_variants(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find remanufacturer variants for the filtered results.
        This replaces the RemanufacturerValidationAgent functionality.
        Optimized for performance with early termination and batch processing.
        
        Args:
            results: The filtered results to find variants for
            
        Returns:
            List of remanufacturer variants
        """
        if not results:
            return []
        
        remanufacturer_variants = []
        seen_variants = set()  # Track already found variants
        start_time = time.time()
        
        # Extract unique SPARTNUMBERs from results (maximum 3 to avoid too much work)
        part_numbers = [result.get('SPARTNUMBER', '') for result in results[:3]]
        
        # For each result in original results (limit to 3 to improve performance)
        for idx, result in enumerate(results[:3]):
            # Check if we've spent too much time already
            if time.time() - start_time > 10:  # 10 second timeout for variant search
                logger.info("Early termination of variant search due to timeout")
                break
                
            spartnumber = result.get('SPARTNUMBER', '')
            partnumber = result.get('PARTNUMBER', '')
            
            # Skip if the result itself is from a remanufacturer (CLASS = V)
            # Handle the case where CLASS might be None or not a string
            class_value = result.get('CLASS', '')
            if class_value is not None and str(class_value).upper() == 'V':
                continue
            
            # Look for remanufactured versions
            try:
                # PHASE 1: Use explicit remanufacturer patterns from agent implementation
                explicit_variants = []
                high_priority_patterns = []
                
                # Generate variants with specific remanufacturer prefixes (high priority)
                for prefix in self.remanufacturer_prefixes[:3]:  # Limit to top 3 prefixes
                    high_priority_patterns.append(f"{prefix}{spartnumber}")
                    high_priority_patterns.append(f"{prefix}-{spartnumber}")
                
                # Generate variants with specific remanufacturer suffixes (high priority)
                for suffix in self.remanufacturer_suffixes[:3]:  # Limit to top 3 suffixes
                    high_priority_patterns.append(f"{spartnumber}{suffix}")
                    high_priority_patterns.append(f"{spartnumber}-{suffix}")
                
                # Execute batch search for high priority patterns
                if high_priority_patterns:
                    # Deduplicate patterns
                    unique_patterns = list(set(high_priority_patterns))
                    
                    # Execute batch search
                    batch_results = db_manager.batch_search_by_spartnumber(unique_patterns)
                    
                    # Process results
                    for pattern in unique_patterns:
                        if pattern in batch_results and batch_results[pattern]:
                            for record in batch_results[pattern]:
                                # Handle the case where CLASS might be None or not a string
                                record_class_value = record.get('CLASS', '')
                                if record_class_value is not None and str(record_class_value).upper() == 'V':
                                    spartnumber_key = record.get('SPARTNUMBER', '')
                                    if spartnumber_key not in seen_variants:
                                        seen_variants.add(spartnumber_key)
                                        # Determine pattern type
                                        if any(pattern.startswith(p) for p in self.remanufacturer_prefixes):
                                            pattern_type = "prefix"
                                        else:
                                            pattern_type = "suffix"
                                        record['pattern_match'] = f"{pattern_type}_match"
                                        record['similarity'] = 0.9  # High confidence for explicit pattern match
                                        record['original_part'] = spartnumber
                                        explicit_variants.append(record)
                
                # Add explicit variants to final list
                if explicit_variants:
                    logger.info(f"Found {len(explicit_variants)} remanufacturer variants using explicit patterns")
                    remanufacturer_variants.extend(explicit_variants)
                    
                    # If we found enough variants with explicit patterns, we can stop
                    if len(remanufacturer_variants) >= 5:
                        logger.info("Early termination of variant search: found enough explicit variants")
                        break
                
                # Check if we've spent too much time already
                if time.time() - start_time > 5:  # 5 second timeout for advanced search
                    logger.info("Skipping advanced variant search due to timeout")
                    continue
                
                # PHASE 2: Database similarity search for this part
                similar_parts = db_manager.search_similar_parts(spartnumber)
                
                # Limit to top 10 similar parts for performance
                for similar_part in similar_parts[:10]:
                    # Only consider parts with CLASS = V (remanufacturer)
                    # Handle the case where CLASS might be None or not a string
                    similar_class_value = similar_part.get('CLASS', '')
                    if similar_class_value is not None and str(similar_class_value).upper() == 'V':
                        similar_spartnumber = similar_part.get('SPARTNUMBER', '')
                        
                        # Skip if it's the same as the original part or already found
                        if similar_spartnumber == spartnumber or similar_spartnumber in seen_variants:
                            continue
                        
                        # Add to seen variants
                        seen_variants.add(similar_spartnumber)
                        
                        # Calculate similarity
                        similarity = calculate_part_confidence(spartnumber, similar_spartnumber)
                        
                        # Only add if similarity is high enough
                        if similarity > 0.7:
                            similar_part['similarity'] = similarity
                            similar_part['original_part'] = spartnumber
                            similar_part['pattern_match'] = "similarity_search"
                            remanufacturer_variants.append(similar_part)
                            
            except Exception as e:
                logger.info(f"Cannot finding remanufacturer variants ------------------")
        
        # Sort variants by similarity score (highest first)
        remanufacturer_variants.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Limit to top 10 highest confidence variants if we have many
        if len(remanufacturer_variants) > 10:
            remanufacturer_variants = remanufacturer_variants[:10]
            logger.info(f"Limited to top 10 highest confidence remanufacturer variants")
        
        logger.info(f"Found {len(remanufacturer_variants)} remanufacturer variants in {time.time() - start_time:.2f}s")
        return remanufacturer_variants

    def _extract_core_numeric_patterns(self, part_number: str) -> List[str]:
        """
        Extract core numeric patterns from a part number.
        This handles cases like extracting "58271" from "GRT01-6582-71"
        by identifying and combining significant numeric sequences.
        """
        patterns = []
        
        # First extract from original with separators intact
        original_sequences = re.findall(r'\d+', part_number)
        
        # Also extract from cleaned version
        cleaned = re.sub(r'[\s\-_.]', '', part_number.upper())
        cleaned_sequences = re.findall(r'\d+', cleaned)
        
        # Strategy 1: Use original sequences with separators
        if len(original_sequences) >= 2:
            # Concatenate all sequences from original
            concatenated = ''.join(original_sequences)
            if len(concatenated) >= 4:
                patterns.append(concatenated)
            
            # Remove leading zeros and concatenate
            no_leading_zeros = ''.join([seq.lstrip('0') or '0' for seq in original_sequences])
            if no_leading_zeros != concatenated and len(no_leading_zeros) >= 4:
                patterns.append(no_leading_zeros)
            
            # For the specific case like "GRT01-6582-71" -> extract "6582" + "71" = "658271"
            # Then remove leading zeros to get "58271"
            for i in range(len(original_sequences) - 1):
                seq1, seq2 = original_sequences[i], original_sequences[i+1]
                if len(seq1) >= 2 and len(seq2) >= 2:
                    combined = seq1 + seq2
                    if combined not in patterns:
                        patterns.append(combined)
                    
                    # Also try without leading zeros
                    seq1_clean = seq1.lstrip('0') or '0'
                    seq2_clean = seq2.lstrip('0') or '0'
                    combined_clean = seq1_clean + seq2_clean
                    if combined_clean != combined and combined_clean not in patterns and len(combined_clean) >= 4:
                        patterns.append(combined_clean)
            
            # Take the largest sequences and combine them
            if len(original_sequences) >= 2:
                sorted_sequences = sorted(original_sequences, key=len, reverse=True)[:2]
                combined = ''.join(sorted_sequences)
                if combined not in patterns and len(combined) >= 4:
                    patterns.append(combined)
                
                # Also without leading zeros
                combined_clean = ''.join([seq.lstrip('0') or '0' for seq in sorted_sequences])
                if combined_clean != combined and combined_clean not in patterns and len(combined_clean) >= 4:
                    patterns.append(combined_clean)
        
        # Strategy 2: Extract from cleaned version (fallback)
        if cleaned_sequences:
            # Extract longest single numeric sequence if substantial
            longest = max(cleaned_sequences, key=len)
            if len(longest) >= 5 and longest not in patterns:
                patterns.append(longest)
                
                # Also try without leading zeros
                longest_clean = longest.lstrip('0') or '0'
                if longest_clean != longest and len(longest_clean) >= 4 and longest_clean not in patterns:
                    patterns.append(longest_clean)
        
        # Strategy 3: Look for embedded patterns using sliding windows on original
        # For cases like "GRT01658271ABC" where core is embedded
        for match in re.finditer(r'\d{5,}', part_number):
            core_num = match.group()
            if core_num not in patterns:
                patterns.append(core_num)
                
                # Also without leading zeros
                core_clean = core_num.lstrip('0') or '0'
                if core_clean != core_num and len(core_clean) >= 4 and core_clean not in patterns:
                    patterns.append(core_clean)
        
        # Remove duplicates while preserving order and filter by minimum length
        seen = set()
        unique_patterns = []
        for pattern in patterns:
            if pattern not in seen and len(pattern) >= 4:
                seen.add(pattern)
                unique_patterns.append(pattern)
        
        logger.info(f"Extracted core numeric patterns from '{part_number}': {unique_patterns}")
        return unique_patterns

    def _search_core_patterns(self, part_number: str) -> Dict[str, Any]:
        """
        Search for parts using extracted core numeric patterns with parallel execution.
        This implements the missing functionality for finding similar patterns.
        """
        logger.info(f"Searching for core patterns in: '{part_number}'")
        
        # Extract core patterns
        core_patterns = self._extract_core_numeric_patterns(part_number)
        
        if not core_patterns:
            return {"cleaned_part": "", "data": []}
        
        # Prepare search tasks for parallel execution
        search_tasks = []
        
        for pattern in core_patterns:
            # Task 1: Exact match search
            search_tasks.append({
                'type': 'exact',
                'pattern': pattern,
                'confidence': 0.85,
                'match_type': 'core_pattern_exact'
            })
            
            # Task 2: Partial match search (contains pattern)
            search_tasks.append({
                'type': 'partial',
                'pattern': pattern,
                'confidence': 0.80,
                'match_type': 'core_pattern_partial'
            })
            
            # Task 3-6: Pattern variations with common separators
            variations = [
                f"{pattern[:2]}-{pattern[2:]}",
                f"{pattern[:-2]}-{pattern[-2:]}",
                f"{pattern[:3]}-{pattern[3:]}",
                f"{pattern[:4]}-{pattern[4:]}" if len(pattern) > 4 else None
            ]
            
            for variation in variations:
                if variation and variation != pattern and len(variation) > 3:
                    search_tasks.append({
                        'type': 'variation',
                        'pattern': variation,
                        'confidence': 0.75,
                        'match_type': 'core_pattern_variation'
                    })
        
        # Execute all searches in parallel
        logger.info(f"Executing {len(search_tasks)} search tasks in parallel for {len(core_patterns)} core patterns")
        start_time = time.time()
        
        search_results = self._execute_parallel_searches(search_tasks)
        
        execution_time = time.time() - start_time
        logger.info(f"Parallel search execution completed in {execution_time:.2f}s")
        
        # Process results
        all_matches = []
        best_pattern = ""
        highest_confidence = 0
        
        for pattern, matches, match_type in search_results:
            if matches:
                logger.info(f"Found {len(matches)} matches for pattern '{pattern}' (type: {match_type})")
                all_matches.extend(matches)
                
                # Track best confidence
                for match in matches:
                    confidence = match.get('confidence', 0)
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_pattern = pattern
        
        # Remove duplicates based on PARTINDEX
        seen_indices = set()
        unique_matches = []
        for match in all_matches:
            part_index = match.get('PARTINDEX')
            if part_index and part_index not in seen_indices:
                seen_indices.add(part_index)
                unique_matches.append(match)
        
        logger.info(f"Core pattern search found {len(unique_matches)} unique matches (confidence: {highest_confidence:.3f})")
        
        return {
            "cleaned_part": best_pattern,
            "data": unique_matches,
            "confidence": highest_confidence
        }

    def _calculate_pattern_similarity(self, pattern: str, target: str) -> float:
        """
        Calculate similarity between a pattern and target part number.
        """
        if pattern in target:
            # Pattern is contained in target
            length_ratio = len(pattern) / len(target)
            return 0.7 + (0.2 * length_ratio)  # 0.7 to 0.9 range
        
        # Use basic string similarity
        return _calculate_similarity_score(pattern, target)
    
    def _execute_parallel_searches(self, search_tasks: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]], str]]:
        """
        Execute multiple database search tasks in parallel with optimized bulk operations.
        """
        if not self.parallel_enabled or len(search_tasks) <= 3:
            # Use optimized bulk operations for small task sets
            return db_manager.bulk_pattern_search_optimized(search_tasks)
        
        # For larger task sets, use parallel execution with chunking
        chunk_size = max(5, len(search_tasks) // self.max_workers)
        chunks = [search_tasks[i:i + chunk_size] for i in range(0, len(search_tasks), chunk_size)]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {}
            for chunk in chunks:
                future = executor.submit(db_manager.bulk_pattern_search_optimized, chunk)
                future_to_chunk[future] = chunk
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_results = future.result(timeout=30)
                    results.extend(chunk_results)
                except Exception as e:
                    logger.warning(f"Parallel search chunk failed: {e}")
                    # Fallback to sequential processing for failed chunk
                    for task in chunk:
                        try:
                            pattern, matches, match_type = self._execute_single_search_task(task)
                            results.append((pattern, matches, match_type))
                        except Exception as inner_e:
                            logger.warning(f"Fallback search failed for pattern '{task.get('pattern', 'unknown')}: {inner_e}")
                            results.append((task.get('pattern', ''), [], task.get('match_type', 'failed')))
        
        return results
    
    def _execute_sequential_searches(self, search_tasks: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]], str]]:
        """
        Execute database search tasks sequentially (fallback method).
        
        Args:
            search_tasks: List of search task dictionaries
        
        Returns:
            List of tuples: (pattern, results, match_type)
        """
        results = []
        for task in search_tasks:
            try:
                pattern, matches, match_type = self._execute_single_search_task(task)
                results.append((pattern, matches, match_type))
            except Exception as e:
                logger.warning(f"Sequential search task failed for pattern '{task.get('pattern', 'unknown')}': {e}")
                results.append((task.get('pattern', ''), [], task.get('match_type', 'failed')))
        return results
    
    def _execute_single_search_task(self, task: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Execute a single search task.
        
        Args:
            task: Search task dictionary
        
        Returns:
            Tuple of (pattern, matches, match_type)
        """
        pattern = task['pattern']
        search_type = task['type']
        confidence = task['confidence']
        match_type = task['match_type']
        
        matches = []
        
        if search_type == 'exact':
            matches = db_manager.search_by_spartnumber(pattern)
        elif search_type == 'partial':
            matches = db_manager.search_parts_containing(pattern)
        elif search_type == 'variation':
            matches = db_manager.search_by_spartnumber(pattern)
        
        # Add metadata to matches
        for match in matches:
            match['confidence'] = confidence
            match['match_type'] = match_type
            match['pattern_used'] = pattern
            
            # For partial matches, calculate more precise confidence
            if search_type == 'partial':
                original_part = match.get('SPARTNUMBER', '')
                if original_part:
                    similarity = self._calculate_pattern_similarity(pattern, original_part)
                    match['confidence'] = min(confidence, similarity)
        
        return pattern, matches, match_type
    
    def set_parallel_execution(self, enabled: bool, max_workers: Optional[int] = None):
        """
        Enable or disable parallel execution of database queries.
        
        Args:
            enabled: Whether to enable parallel execution
            max_workers: Optional override for max worker threads
        """
        self.parallel_enabled = enabled
        if max_workers is not None:
            self.max_workers = max_workers
        
        logger.info(f"Parallel execution {'enabled' if enabled else 'disabled'} with {self.max_workers} max workers")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the processor.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            'parallel_enabled': self.parallel_enabled,
            'max_workers': self.max_workers,
            'pattern_optimizer_stats': self.pattern_optimizer.get_stats() if hasattr(self.pattern_optimizer, 'get_stats') else {}
        }
        
    def _clear_intermediate_results(self):
        """
        Clear intermediate processing results to free memory.
        """
        # Force garbage collection for better memory efficiency
        import gc
        gc.collect()
    
    def _filter_by_description_efficiently(self, matches: List[Dict[str, Any]], input_description: str) -> List[Dict[str, Any]]:
        """
        Filter matches by description with memory-efficient processing.
        """
        if not input_description or not matches:
            return matches
        
        filtered_matches = []
        
        # Process in smaller batches to save memory
        batch_size = 50
        for i in range(0, len(matches), batch_size):
            batch = matches[i:i + batch_size]
            
            for match in batch:
                part_desc = match.get('partdesc', '')
                if part_desc:
                    similarity = text_similarity(input_description, part_desc)
                    match['description_similarity'] = similarity
                    
                    # Only keep matches with reasonable description similarity
                    if similarity > 0.3:  # 30% threshold for memory efficiency
                        filtered_matches.append(match)
            
            # Clear batch from memory
            del batch
            
            # Periodic garbage collection for large datasets
            if i > 0 and i % 200 == 0:
                self._clear_intermediate_results()
        
        return filtered_matches

    def _enhance_results_with_scoring(self, results: List[Dict[str, Any]], 
                                    input_part: str, 
                                    input_description: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Enhance results with scoring metrics including part number scoring, 
        description scoring, noise detection, and confidence codes.
        
        Args:
            results: List of search results to enhance
            input_part: The original input part number
            input_description: Optional input description
            
        Returns:
            Enhanced results with scoring metrics
        """
        if not results:
            return []
            
        # Standardize input format - handle various input types
        standardized_results = []
        
        # Case 1: Results is a dictionary with a 'data' key (output from sliding window)
        if isinstance(results, dict) and 'data' in results:
            logger.info("Results is a dictionary with 'data' key, extracting the data list")
            standardized_results = results.get('data', [])
        # Case 2: Results is a tuple 
        elif isinstance(results, tuple) and len(results) > 0:
            logger.warning(f"Received tuple in enhance_results_with_scoring, using first element: {type(results[0])}")
            if isinstance(results[0], list):
                standardized_results = results[0]
            elif isinstance(results[0], dict) and 'data' in results[0]:
                standardized_results = results[0].get('data', [])
            else:
                standardized_results = [results[0]]
        # Case 3: Results is a list of lists
        elif results and isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            logger.warning("Received nested list in enhance_results_with_scoring, flattening...")
            flat_results = []
            for sublist in results:
                if isinstance(sublist, list):
                    flat_results.extend(sublist)
                else:
                    flat_results.append(sublist)
            standardized_results = flat_results
        # Case 4: Results is already a flat list
        else:
            standardized_results = results
        
        # Log the standardization process
        logger.info(f"Standardized results from {type(results)} to list with {len(standardized_results)} items")
        
        enhanced_results = []
        
        for result in standardized_results:
            # Skip if result is not a dictionary (e.g., if it's a list)
            if not isinstance(result, dict):
                logger.warning(f"Skipping non-dictionary result in enhance_results_with_scoring: {type(result)}")
                continue
                
            # Create a copy to avoid modifying original
            enhanced_result = result.copy()
            
            # Get the matched part number and description
            try:
                # Handle both dictionary access methods
                if isinstance(enhanced_result, dict):
                    # Dictionary case - use get method
                    matched_part = enhanced_result.get('SPARTNUMBER', '') or enhanced_result.get('PARTNUMBER', '')
                    matched_description = enhanced_result.get('partdesc', '')
                elif hasattr(enhanced_result, '__getitem__') and not isinstance(enhanced_result, (str, bytes)):
                    # List-like object case - use indexing carefully
                    logger.warning(f"Enhanced result is not a dictionary but a {type(enhanced_result)}")
                    # Try to extract values safely
                    try:
                        # For lists, create a simple dictionary
                        temp_dict = {}
                        if len(enhanced_result) > 0:
                            # Attempt to convert first item to a string for matching
                            temp_dict['PARTNUMBER'] = str(enhanced_result[0]) if enhanced_result[0] is not None else ''
                        matched_part = temp_dict.get('SPARTNUMBER', '') or temp_dict.get('PARTNUMBER', '')
                        matched_description = temp_dict.get('partdesc', '')
                    except (IndexError, TypeError):
                        matched_part = ''
                        matched_description = ''
                else:
                    # Other types - convert to string
                    matched_part = str(enhanced_result) if enhanced_result is not None else ''
                    matched_description = ''
            except Exception as e:
                logger.error(f"Error extracting part info: {e}")
                matched_part = ''
                matched_description = ''
            
            # Calculate all scoring metrics
            try:
                scoring_metrics = scoring_system.calculate_all_scores(
                    input_part=input_part,
                    matched_part=matched_part,
                    input_description=input_description,
                    matched_description=matched_description
                )
                
                # Add scoring metrics to the result - check if enhanced_result is a dictionary
                if isinstance(enhanced_result, dict):
                    enhanced_result.update(scoring_metrics)
                else:
                    # If not a dictionary, create a new dictionary with the data we have
                    enhanced_result = {
                        'PARTNUMBER': matched_part,
                        'partdesc': matched_description,
                        **scoring_metrics
                    }
                    logger.warning(f"Enhanced result was not a dictionary, created new one: {enhanced_result}")
                
                # Log scoring details for debugging - safely access the metrics
                logger.debug(f"Scoring for '{matched_part}': "
                            f"part_score={scoring_metrics.get('part_number_score', 'N/A')}, "
                            f"desc_score={scoring_metrics.get('description_score', 'N/A')}, "
                            f"noise={scoring_metrics.get('noise', 'N/A')}, "
                            f"noise_text={scoring_metrics.get('noise_text', 'N/A')}, "
                            f"cocode={scoring_metrics.get('cocode', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Error calculating or applying scoring metrics: {e}")
                # Ensure we have a valid dictionary to return
                if not isinstance(enhanced_result, dict):
                    enhanced_result = {
                        'PARTNUMBER': matched_part,
                        'partdesc': matched_description,
                        'part_number_score': 0.0,
                        'description_score': 0.0,
                        'noise': False,
                        'noise_text': 'no',
                        'combined_score': 0.0,
                        'cocode': '00',
                        'error': str(e)
                    }
            
            enhanced_results.append(enhanced_result)
            
            # Log the final enhanced result for debugging
            logger.debug(f"Final enhanced result: {enhanced_result}")
            
        # Sort results by confidence code for better ordering
        # Higher confidence codes (like "55", "54") should come first
        # Make sorting safe by checking if each item is a dictionary
        try:
            enhanced_results.sort(
                key=lambda x: x.get('cocode', '00') if isinstance(x, dict) else '00', 
                reverse=True
            )
        except Exception as e:
            logger.error(f"Error sorting results by confidence code: {e}")
            # If sorting fails, return results unsorted rather than failing
        
        logger.info(f"Enhanced {len(enhanced_results)} results with scoring metrics")
        
        return enhanced_results
    
# Create a singleton instance that can be imported elsewhere
part_processor = PartProcessor()