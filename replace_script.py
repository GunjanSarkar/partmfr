import os
import re

def replace_calculate_confidence_calls(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Pattern to match calculate_confidence calls with part numbers
    pattern = r'confidence\s*=\s*calculate_confidence\(([^,]+),\s*([^)]+)\)'
    replacement = r'confidence = calculate_part_confidence(\1, \2)'
    
    # Replace all matching patterns
    modified_content = re.sub(pattern, replacement, content)
    
    # Handle special case for similarity variable
    pattern2 = r'similarity\s*=\s*calculate_confidence\(([^,]+),\s*([^)]+)\)'
    replacement2 = r'similarity = calculate_part_confidence(\1, \2)'
    
    modified_content = re.sub(pattern2, replacement2, modified_content)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)
    
    print("Replacements complete!")

# Path to processor.py
processor_file = r"c:\Users\gunjans\Desktop\Motor_Final_solution_donotpushtorepo\motorprallelprocessing\Motor_final_Solution\src\processor.py"

# Execute the replacement
replace_calculate_confidence_calls(processor_file)
