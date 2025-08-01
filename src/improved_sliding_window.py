import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ImprovedSlidingWindowProcessor:
    """
    Improved sliding window search for part numbers with optimized stopping criteria.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.min_window_size = 4  # Don't use windows smaller than 4 characters
    
    def clean_part_number(self, part_number: str) -> str:
        """Clean the part number by removing special characters"""
        return re.sub(r'[^A-Z0-9]', '', part_number.upper())
    
    def search(self, part_number: str) -> Optional[Dict[str, Any]]:
        """
        Search for a part number using improved sliding window approach
        Returns None if no matches found
        """
        # 1. Clean the part number
        cleaned_full = self.clean_part_number(part_number)
        logger.info(f"Original input: '{part_number}'")
        logger.info(f"Cleaned full number: '{cleaned_full}'")
        
        # 2. Try exact match first
        logger.info(f"Trying exact match with full cleaned number: {cleaned_full}")
        exact_matches = self.db_manager.search_by_spartnumber(cleaned_full)
        
        if exact_matches:
            logger.info("Found exact match with full number!")
            all_matches = []
            seen_spartnumbers = set()
            
            # Add exact matches
            for match in exact_matches:
                spartnumber = match.get("SPARTNUMBER", "")
                if spartnumber not in seen_spartnumbers:
                    match["search_candidate"] = cleaned_full
                    match["confidence"] = 1.0  # Highest confidence for exact match
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
                            variant["confidence"] = 0.75  # Lower confidence for variants
                            variant["is_remanufacturer"] = True
                            variant["noise_variant"] = True
                            all_matches.append(variant)
                            seen_spartnumbers.add(spartnumber)
            
            return {
                "cleaned_part": cleaned_full,
                "data": all_matches,
                "confidence": 1.0
            }
        
        # 3. If no exact match, try prefix removal strategy first (more refined)
        logger.info("No exact match found, trying prefix removal strategy")
        all_matches = []
        seen_spartnumbers = set()
        
        # Try removing characters from the left (prefix removal)
        # This is more refined than sliding window and often finds better matches
        for i in range(1, min(4, len(cleaned_full))):  # Try removing up to 3 characters from left
            candidate = cleaned_full[i:]
            logger.info(f"Trying prefix removal candidate: {candidate}")
            
            # Try exact match for this candidate
            matches = self.db_manager.search_by_spartnumber(candidate)
            if matches:
                logger.info(f"Found exact match with prefix removal: {candidate}")
                
                # Add the exact matches
                for match in matches:
                    spartnumber = match.get("SPARTNUMBER", "")
                    if spartnumber not in seen_spartnumbers:
                        match["search_candidate"] = candidate
                        match["confidence"] = 0.95  # High confidence for prefix removal (better than sliding)
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
                                variant["confidence"] = 0.85  # High confidence for variants of prefix removal
                                variant["is_remanufacturer"] = True
                                variant["noise_variant"] = True
                                all_matches.append(variant)
                                seen_spartnumbers.add(spartnumber)
                
                # We found matches with this candidate, return results
                return {
                    "cleaned_part": cleaned_full,
                    "data": all_matches,
                    "confidence": 0.95
                }
        
        # 4. If prefix removal fails, then use sliding window with optimized stopping criteria
        logger.info("Prefix removal didn't find matches, trying sliding window")
        
        # Determine how much of the string to search based on length
        total_length = len(cleaned_full)
        
        # Determine minimum position to start sliding window
        # For ≥10 chars: stop at last 4 chars
        # For 5-9 chars: stop at last 3 chars 
        # For <5 chars: use first 3 chars
        if total_length >= 10:
            min_start_pos = total_length - 4
        elif total_length >= 5:
            min_start_pos = total_length - 3
        else:
            min_start_pos = 0
            
        # Start with full length window
        window_size = total_length
        
        # Don't go below minimum window size
        while window_size >= self.min_window_size:
            for i in range(min(total_length - window_size + 1, min_start_pos + 1)):
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
            
            # Reduce window size and try again
            window_size -= 1
        
        # No matches found at all
        logger.info("No matches found")
        return None
