import pandas as pd
import io
from typing import List, Dict, Any, Optional, Tuple

class FileProcessor:
    """
    Handles processing of Excel and CSV files to extract part numbers and descriptions
    """
    
    @staticmethod
    def process_file(file_content: bytes, file_extension: str) -> List[Dict[str, str]]:
        """
        Process the uploaded file and extract part numbers and descriptions
        
        Args:
            file_content: The binary content of the uploaded file
            file_extension: The file extension (csv, xlsx, xls)
            
        Returns:
            A list of dictionaries containing part numbers and descriptions
        """
        if file_extension == "csv":
            return FileProcessor._process_csv(file_content)
        elif file_extension in ["xlsx", "xls"]:
            return FileProcessor._process_excel(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    @staticmethod
    def _process_csv(file_content: bytes) -> List[Dict[str, str]]:
        """Process CSV file to extract part numbers and descriptions"""
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            return FileProcessor._extract_part_data(df)
        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")
    
    @staticmethod
    def _process_excel(file_content: bytes) -> List[Dict[str, str]]:
        """Process Excel file to extract part numbers and descriptions"""
        try:
            df = pd.read_excel(io.BytesIO(file_content))
            return FileProcessor._extract_part_data(df)
        except Exception as e:
            raise ValueError(f"Error processing Excel file: {str(e)}")
    
    @staticmethod
    def _extract_part_data(df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Extract part numbers and descriptions from a pandas DataFrame
        
        This method attempts to find columns that might contain part numbers and descriptions
        """
        part_data = []
        
        # Try to identify the column that contains part numbers
        partnumber_columns = [
            'PARTNUMBER',  # Make sure this is the first priority
            'PART_NUMBER', 'PART NUMBER', 'PART_NO', 'PART NO',
            'PN', 'P/N', 'PART#', 'PART #', 'PARTNR', 'PART NR'
        ]
        
        # Try to identify the column that contains part descriptions
        description_columns = [
            'PARTDESC',  # Make sure this is the first priority
            'DESCRIPTION', 'DESC', 'PART_DESC', 'PART DESC',
            'DETAILS', 'INFO', 'SPECIFICATION', 'SPEC', 'NOTES'
        ]
        
        # Case-insensitive search for column names
        original_columns = df.columns.tolist()
        
        # Always prioritize exact 'partnumber' and 'partdesc' columns
        partnumber_col = None
        description_col = None
        
        # First pass: Look for exact 'partnumber' and 'partdesc' matches (case-insensitive)
        for idx, col in enumerate(original_columns):
            col_upper = col.upper()
            if col_upper == 'PARTNUMBER':
                partnumber_col = df.columns[idx]
                print(f"Found exact 'partnumber' column: {col}")
            elif col_upper == 'PARTDESC':
                description_col = df.columns[idx]
                print(f"Found exact 'partdesc' column: {col}")
                
        # If partnumber not found, try other column names
        if not partnumber_col:
            # Check standard part number column names
            for possible_col in partnumber_columns:
                for idx, col in enumerate(original_columns):
                    if col.upper() == possible_col:
                        partnumber_col = df.columns[idx]
                        print(f"Found part number column: {col}")
                        break
                if partnumber_col:
                    break
                    
            # If still not found, try partial matches
            if not partnumber_col:
                for idx, col in enumerate(original_columns):
                    col_upper = col.upper()
                    for possible_col in partnumber_columns:
                        if possible_col in col_upper:
                            partnumber_col = df.columns[idx]
                            print(f"Found partial part number column match: {col}")
                            break
                    if partnumber_col:
                        break
            
            # Last resort: use first column
            if not partnumber_col and not df.empty:
                partnumber_col = df.columns[0]
                print(f"Warning: No part number column found, using first column: {df.columns[0]}")
        
        # If description not found, try other column names
        if not description_col:
            # Check standard description column names
            for possible_col in description_columns:
                for idx, col in enumerate(original_columns):
                    if col.upper() == possible_col:
                        description_col = df.columns[idx]
                        print(f"Found description column: {col}")
                        break
                if description_col:
                    break
            
            # If still not found, try partial matches
            if not description_col:
                for idx, col in enumerate(original_columns):
                    col_upper = col.upper()
                    for possible_col in description_columns:
                        if possible_col in col_upper:
                            description_col = df.columns[idx]
                            print(f"Found partial description column match: {col}")
                            break
                    if description_col:
                        break
        
        # Extract part numbers and descriptions with robust error handling
        if partnumber_col:
            try:
                # Handle NaN values and convert to string safely
                part_numbers = df[partnumber_col].fillna('').astype(str).tolist()
                
                # If we have both part numbers and descriptions
                if description_col:
                    try:
                        descriptions = df[description_col].fillna('').astype(str).tolist()
                    except Exception as e:
                        print(f"Warning: Error processing description column: {e}")
                        descriptions = [''] * len(part_numbers)
                        
                    # Ensure lists are the same length
                    min_length = min(len(part_numbers), len(descriptions))
                    for i in range(min_length):
                        pn = part_numbers[i].strip() if isinstance(part_numbers[i], str) else str(part_numbers[i]).strip()
                        desc = descriptions[i].strip() if isinstance(descriptions[i], str) else str(descriptions[i]).strip()
                        
                        if pn:  # Only add non-empty part numbers
                            part_data.append({
                                'part_number': pn,
                                'description': desc
                            })
                else:
                    # Only part numbers, no descriptions
                    for part_number in part_numbers:
                        pn = part_number.strip() if isinstance(part_number, str) else str(part_number).strip()
                        
                        if pn:  # Only add non-empty part numbers
                            part_data.append({
                                'part_number': pn,
                                'description': ''
                            })
            except Exception as e:
                print(f"Error extracting part data: {e}")
                
        return part_data
