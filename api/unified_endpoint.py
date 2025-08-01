"""
Unified endpoint module for the Motor Parts API.
This provides a single endpoint that integrates all processing functionality.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union
import json
import time
import asyncio
import os

# Import models from the separate models file to avoid circular imports
from api.models import UnifiedRequest, UnifiedResponse
from src.database import db_manager
from src.processor import PartProcessor
from src.file_processor import FileProcessor

router = APIRouter()

# Create processor instances
part_processor = PartProcessor()
file_processor = FileProcessor()
    
@router.post("/api/unified", response_model=UnifiedResponse, tags=["Unified Processing"])
async def unified_processing(request: UnifiedRequest):
    """
    Unified endpoint for processing part numbers. Can handle both single parts and batch operations.
    
    - For single part: Set operation_type="single_part" and provide part_data as a dictionary with part_number and optional part_description
    - For batch processing: Set operation_type="batch" and provide part_data as a list of dictionaries
    
    Examples:
    ```
    # Single part
    {
        "operation_type": "single_part",
        "part_data": {
            "part_number": "DDE-A471014002",
            "part_description": "Optional description"
        }
    }
    
    # Batch
    {
        "operation_type": "batch",
        "part_data": [
            {"part_number": "DDE-A471014002"},
            {"part_number": "HENS-22137-1", "part_description": "Some description"}
        ]
    }
    ```
    """
    start_time = time.time()
    
    try:
        if request.operation_type == "single_part":
            if not request.part_data or not isinstance(request.part_data, dict):
                raise HTTPException(status_code=400, detail="Invalid part_data for single_part operation. Must provide a dictionary.")
            
            # Extract data
            part_number = request.part_data.get("part_number")
            part_description = request.part_data.get("part_description")
            
            if not part_number:
                raise HTTPException(status_code=400, detail="Missing part_number in part_data")
            
            # Process the part using optimized processor with early termination (async)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                part_processor.process_part_with_early_termination,
                part_number,
                part_description
            )
            
            # Add execution time to the result
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            
            return {
                "status": "success",
                "message": "Part processed successfully",
                "result": result,
                "execution_time": execution_time,
                "operation_type": "single_part"
            }
            
        elif request.operation_type == "batch":
            if not request.part_data or not isinstance(request.part_data, list):
                raise HTTPException(status_code=400, detail="Invalid part_data for batch operation. Must provide a list.")
            
            # Create a batch processor
            file_processor = FileProcessor()
            
            # Convert the list to the format expected by the batch processor
            batch_data = []
            for item in request.part_data:
                if not isinstance(item, dict) or "part_number" not in item:
                    raise HTTPException(status_code=400, detail="Each batch item must be a dictionary with at least a part_number key")
                
                batch_item = {
                    "part_number": item.get("part_number", ""),
                    "description": item.get("part_description", "")
                }
                batch_data.append(batch_item)
            
            # Process the batch
            batch_results = file_processor.process_batch(batch_data, 
                                                         max_workers=os.cpu_count(), 
                                                         show_progress=False)
            
            execution_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": f"Processed {len(batch_data)} parts in batch",
                "result": {
                    "processed_count": len(batch_data),
                    "batch_results": batch_results
                },
                "execution_time": execution_time,
                "operation_type": "batch"
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid operation_type: {request.operation_type}. Must be 'single_part' or 'batch'")
            
    except Exception as e:
        # Log the error
        import logging
        logging.error(f"Error in unified endpoint: {str(e)}")
        
        # Return an error response
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "result": None,
            "execution_time": time.time() - start_time,
            "operation_type": request.operation_type
        }
