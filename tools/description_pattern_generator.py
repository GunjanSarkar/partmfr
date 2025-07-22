from openai import OpenAI
import re
from typing import List, Dict, Any, Optional
from config.settings import settings

class DescriptionPatternGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        
    def generate_regex_pattern(self, description: str) -> Optional[str]:
        """
        Generate regex pattern for part description using OpenAI.
        Tries primary model first, then falls back to alternatives if needed.
        
        Args:
            description: The part description to generate regex for
            
        Returns:
            Optional[str]: Generated regex pattern or None if all attempts fail
        """
        # Models to try in order with their max tokens
        models = [
            (settings.openai_model_primary, settings.max_tokens_primary),
            (settings.openai_model_fallback1, settings.max_tokens_fallback1),
            (settings.openai_model_fallback2, settings.max_tokens_fallback2)
        ]
        
        prompt = f"""Generate a regex pattern to match part descriptions similar to: "{description}"
        
        Requirements:
        1. Pattern should match key words regardless of their order
        2. Handle variations in spacing and punctuation
        3. Account for potential misspellings or variations
        4. Focus on significant words (ignore common words like 'the', 'and', etc)
        5. Pattern should allow for partial matches
        
        Output the regex pattern only, no explanation needed."""

        last_error = None
        for model, max_tokens in models:
            try:
                response = self.client.responses.create(
                    model=model,
                    input=[{
                        "role": "developer",
                        "content": [{
                            "type": "input_text",
                            "text": prompt
                        }]
                    }],
                    text={
                        "format": {
                            "type": "text"
                        }
                    },
                    reasoning={
                        "effort": "medium"
                    },
                    tools=[],
                    store=True,
                    max_tokens=max_tokens
                )
                
                if response and response.text:
                    # Clean and validate the regex pattern
                    pattern = response.text.strip()
                    try:
                        re.compile(pattern)
                        return pattern
                    except re.error:
                        continue
                        
            except Exception as e:
                last_error = str(e)
                continue
                
        if last_error:
            print(f"Failed to generate regex pattern: {last_error}")
        return None
        
    def calculate_description_match_score(self, input_desc: str, db_desc: str, pattern: Optional[str] = None) -> float:
        """
        Calculate similarity score between input description and database description.
        
        Args:
            input_desc: User provided description
            db_desc: Description from database
            pattern: Optional regex pattern (will be generated if not provided)
            
        Returns:
            float: Match score between 0 and 1
        """
        if not input_desc or not db_desc:
            return 0.0
            
        if pattern is None:
            pattern = self.generate_regex_pattern(input_desc)
            if pattern is None:
                return 0.0
                
        try:
            # Get word counts
            input_words = set(re.findall(r'\w+', input_desc.lower()))
            significant_words = {w for w in input_words if len(w) > 2}  # Filter out short words
            
            # Count matching words using the generated pattern
            matches = re.finditer(pattern, db_desc, re.IGNORECASE)
            matched_words = set()
            for match in matches:
                words = re.findall(r'\w+', match.group().lower())
                matched_words.update(w for w in words if w in significant_words)
            
            # Calculate match score
            if not significant_words:
                return 0.0
                
            return len(matched_words) / len(significant_words)
            
        except (re.error, Exception) as e:
            print(f"Error calculating description match score: {str(e)}")
            return 0.0
