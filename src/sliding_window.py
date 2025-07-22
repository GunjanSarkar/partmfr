import re
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SlidingWindowProcessor:
    """
    A class focused solely on implementing sliding window search for part numbers.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def search(self, input_partnumber: str) -> Dict[str, Any]:
        """
        Perform sliding window search for a part number and find remanufacturer variants.
        
        Args:
            input_partnumber: The part number to search for (can contain spaces, special chars)
            
        Returns:
            Dictionary containing:
            - cleaned_part: The cleaned input part number
            - data: List of matches and remanufacturer variants
            - confidence: Match confidence score
        """
        # 1. Clean the input - remove ALL special characters and spaces
        logger.info(f"Original input: '{input_partnumber}'")
        cleaned_full = re.sub(r'[^A-Z0-9]', '', input_partnumber.upper())
        logger.info(f"Cleaned full number: '{cleaned_full}'")
        
        # 2. First try exact match with cleaned full number
        logger.info(f"Trying exact match with full cleaned number: {cleaned_full}")
        exact_matches = self.db_manager.search_by_spartnumber(cleaned_full)
        
        if exact_matches:
            logger.info("Found exact match with full number!")
            all_matches = []
            seen_spartnumbers = set()
            
            # Add exact matches first
            for match in exact_matches:
                spartnumber = match.get("SPARTNUMBER", "")
                match["search_candidate"] = cleaned_full
                match["confidence"] = 1.0
                match["is_remanufacturer"] = False
                match["noise_variant"] = False
                all_matches.append(match)
                seen_spartnumbers.add(spartnumber)
                
            # Look for remanufacturer variants
            variant_matches = self.db_manager.search_by_partnumber_like(f"%{cleaned_full}%")
            for variant in variant_matches:
                variant_number = variant.get("PARTNUMBER", "")
                spartnumber = variant.get("SPARTNUMBER", "")
                
                if spartnumber not in seen_spartnumbers and variant_number != cleaned_full:
                    len_diff = abs(len(variant_number) - len(cleaned_full))
                    if 1 <= len_diff <= 4 and cleaned_full in variant_number:
                        if variant.get("CLASS") in ["M", "O", "V"]:
                            variant["search_candidate"] = cleaned_full
                            variant["confidence"] = 0.85
                            variant["is_remanufacturer"] = True
                            variant["noise_variant"] = True
                            all_matches.append(variant)
                            seen_spartnumbers.add(spartnumber)
                            
            return {
                "cleaned_part": cleaned_full,
                "data": all_matches,
                "confidence": 1.0
            }
            
        # 3. If no exact match, try sliding window
        logger.info("No exact match found, trying sliding window")
        all_matches = []
        seen_spartnumbers = set()
        window_size = len(cleaned_full)
        
        while window_size >= 4:  # Don't try windows smaller than 4 characters
            for i in range(len(cleaned_full) - window_size + 1):
                candidate = cleaned_full[i:i + window_size]
                logger.info(f"Trying sliding window candidate: {candidate}")
                
                # Try exact match for this candidate
                matches = self.db_manager.search_by_spartnumber(candidate)
                if matches:
                    logger.info(f"Found exact match with sliding window: {candidate}")
                    
                    # Add the exact matches
                    for match in matches:
                        spartnumber = match.get("SPARTNUMBER", "")
                        if spartnumber not in seen_spartnumbers:
                            match["search_candidate"] = candidate
                            match["confidence"] = 0.9  # High confidence for sliding window match
                            match["is_remanufacturer"] = False
                            match["noise_variant"] = False
                            all_matches.append(match)
                            seen_spartnumbers.add(spartnumber)
                    
                    # Look for remanufacturer variants of this match
                    variant_matches = self.db_manager.search_by_partnumber_like(f"%{candidate}%")
                    for variant in variant_matches:
                        variant_number = variant.get("PARTNUMBER", "")
                        spartnumber = variant.get("SPARTNUMBER", "")
                        
                        if spartnumber not in seen_spartnumbers and variant_number != candidate:
                            len_diff = abs(len(variant_number) - len(candidate))
                            if 1 <= len_diff <= 4 and candidate in variant_number:
                                if variant.get("CLASS") in ["M", "O", "V"]:
                                    variant["search_candidate"] = candidate
                                    variant["confidence"] = 0.75  # Lower confidence for variants
                                    variant["is_remanufacturer"] = True
                                    variant["noise_variant"] = True
                                    all_matches.append(variant)
                                    seen_spartnumbers.add(spartnumber)
                    
                    # We found matches with this candidate, stop sliding window
                    return {
                        "cleaned_part": cleaned_full,
                        "data": all_matches,
                        "confidence": 0.9
                    }
                    
            window_size -= 1
            
        # No matches found at all
        logger.info("No matches found")
        return None
        if match:
            prefix, numeric_part = match.groups()
            prefix = prefix or ""  # Convert None to empty string
            logger.info(f"Split into prefix: {prefix}, numeric: {numeric_part}")
            
            # Try these combinations in order:
            candidates = [
                cleaned_full,  # Full number with prefix
                numeric_part,  # Just the numeric part
                prefix + numeric_part.rstrip('0')  # Full number without trailing zeros
            ]
            candidates = list(dict.fromkeys(candidates))  # Remove duplicates while preserving order
            
            # 2. Try exact matches in priority order
            for candidate in candidates:
                logger.info(f"Trying exact match with: {candidate}")
                exact_matches = self.db_manager.search_by_spartnumber(candidate)
                if exact_matches:
                    logger.info(f"Found exact match with: {candidate}")
                    return {
                        "cleaned_part": cleaned_full,
                        "data": [{**match, 
                                "search_candidate": candidate,
                                "confidence": 1.0 if candidate == cleaned_full else 0.9,
                                "is_remanufacturer": False,
                                "noise_variant": False} for match in exact_matches],
                        "confidence": 1.0 if candidate == cleaned_full else 0.9
                    }
            
        # 3. If no exact match, generate sliding window candidates
        logger.info("No exact match found, trying sliding window candidates")
        candidates = []
        window_size = len(cleaned_full)
        
        # Generate all possible substrings maintaining order
        while window_size > 0:
            for i in range(len(cleaned_full) - window_size + 1):
                candidate = cleaned_full[i:i + window_size]
                if candidate not in candidates:
                    candidates.append(candidate)
            window_size -= 1
            
        logger.info(f"Generated sliding window candidates: {candidates}")
        
        logger.info(f"Generated candidates: {candidates}")
        
        logger.info(f"Generated candidates: {candidates}")
        
        # 3. Search for exact matches in database
        all_matches = []
        seen_spartnumbers = set()
        
        for candidate in candidates:
            # Search for exact matches only
            matches = self.db_manager.search_by_spartnumber(candidate)
            
            if matches:
                    logger.info(f"Found exact matches for candidate: {candidate}")
                    
                    # First add the exact matches
                    for match in matches:
                        spartnumber = match.get("SPARTNUMBER", "")
                        if spartnumber not in seen_spartnumbers:
                            match["search_candidate"] = candidate
                            match["confidence"] = 1.0  # Exact match confidence
                            match["is_remanufacturer"] = False
                            match["noise_variant"] = False
                            all_matches.append(match)
                            seen_spartnumbers.add(spartnumber)
                    
                    # Then search for remanufacturer variants with LIKE
                    logger.info(f"Searching for remanufacturer variants of: {candidate}")
                    variant_matches = self.db_manager.search_by_partnumber_like(f"%{candidate}%")
                    
                    for variant in variant_matches:
                        variant_number = variant.get("PARTNUMBER", "")
                        spartnumber = variant.get("SPARTNUMBER", "")
                        
                        # Skip if we've seen this SPARTNUMBER or if it's an exact match
                        if spartnumber in seen_spartnumbers or variant_number == candidate:
                            continue
                            
                        # Check if variant has only 2-4 extra characters
                        len_diff = abs(len(variant_number) - len(candidate))
                        if 1 <= len_diff <= 4:
                            # Check if the variant contains our candidate
                            if candidate in variant_number:
                                variant["search_candidate"] = candidate
                                variant["confidence"] = 0.85  # Lower confidence for variants
                                variant["is_remanufacturer"] = True
                                variant["noise_variant"] = True
                                
                                # Only include variants with valid classification
                                if variant.get("CLASS") in ["M", "O", "V"]:
                                    all_matches.append(variant)
                                    seen_spartnumbers.add(spartnumber)
                    
                    # Early termination - if we found the main matches, stop sliding window
                    if all_matches:
                        logger.info(f"Early termination - found matches and variants")
                        break       
                    if not all_matches:
                            logger.info("No matches found for any candidate")
            return None
            
        return {
            "cleaned_part": cleaned_full,
            "data": all_matches,
            "confidence": 0.85
        }
