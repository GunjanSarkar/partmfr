"""
Databricks compatible version of the Motor Part API.

This file adapts the FastAPI application from main.py to run in a Databricks environment.
Key differences:
- File path handling for DBFS
- Authentication integration with Databricks
- Logging configured for Databricks
- Environment variable handling for Databricks
"""

import os
import time
import json
import logging
import base64
import tempfile
import random
import string
import re
import sys
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from datetime import datetime
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor, as_completed

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Path as FastAPIPath, Body, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html
from pydantic import BaseModel, Field
import uvicorn

# Configure paths for Databricks environment (using DBFS)
# Adjust these paths according to your Databricks workspace structure
def get_databricks_paths():
    """Configure paths for Databricks environment"""
    # In Databricks, the working directory might be different
    # We can use DBFS for persistent storage
    current_dir = Path(os.getcwd())
    
    # Determine if we're running in Databricks
    in_databricks = 'DATABRICKS_RUNTIME_VERSION' in os.environ
    
    if in_databricks:
        # Use DBFS paths when running in Databricks
        base_path = Path("/dbfs/FileStore/motor_api")
        static_dir = base_path / "static"
        config_dir = base_path / "config"
        src_dir = base_path / "src"
        tools_dir = base_path / "tools"
    else:
        # Local paths for testing outside Databricks
        base_path = current_dir
        static_dir = current_dir / "static"
        config_dir = current_dir / "config"
        src_dir = current_dir / "src"
        tools_dir = current_dir / "tools"
    
    # Ensure directories exist
    os.makedirs(static_dir, exist_ok=True)
    
    return {
        "base_path": base_path,
        "static_dir": static_dir,
        "config_dir": config_dir,
        "src_dir": src_dir,
        "tools_dir": tools_dir,
        "in_databricks": in_databricks
    }

# Setup logging for Databricks
def setup_databricks_logging():
    """Configure logging for Databricks environment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("motor_api_databricks")
    return logger

# Initialize paths and logging
paths = get_databricks_paths()
logger = setup_databricks_logging()

# Add parent directory to path for imports
sys.path.insert(0, str(paths["base_path"]))

# Import application modules
try:
    from src.processor import PartProcessor
    from src.file_processor import FileProcessor
    from src.database import DatabaseManager
    from tools.basic_tools import calculate_similarity
    from tools.scoring_system import scoring_system
    from config.settings import get_settings, Settings, update_settings, TESTING, save_settings
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Make sure all required modules are available in the Databricks environment")
    raise

# Define models for API requests and responses
class PartNumberRequest(BaseModel):
    """
    Request model for processing a single part number.
    """
    part_number: str
    part_description: Optional[str] = None

class BatchProcessingRequest(BaseModel):
    """
    Request model for batch processing multiple parts.
    """
    parts: List[Dict[str, str]]  # List of {"part_number": str, "description": str}

class FileProcessingRequest(BaseModel):
    """
    Request model for processing parts from a file.
    """
    file_data: str  # Base64 encoded file content
    filename: str
    output_format: Optional[str] = "json"  # json or csv

class UnifiedProcessRequest(BaseModel):
    """
    Unified request model for all processing operations.
    """
    operation: str  # "single", "batch", or "file"
    # Single part processing
    part_number: Optional[str] = None
    part_description: Optional[str] = None
    # Batch processing
    parts: Optional[List[Dict[str, str]]] = None
    # File processing
    file_data: Optional[str] = None
    filename: Optional[str] = None
    output_format: Optional[str] = "json"

class UnifiedProcessResponse(BaseModel):
    """
    Unified response model for all processing operations (single, batch, file).
    
    For single part results, includes detailed scoring information:
    - part_number_score: 0-5 score based on matching characters / total input characters
    - description_score: 0-5 score based on matching words / total input description words
    - noise: boolean indicating if the matched part contains extra characters
    - cocode: two-digit confidence code combining part and description scores
    
    See /api/scoring/info endpoint for detailed scoring documentation.
    """
    operation: str
    status: str
    execution_time: float
    # Single part response fields
    cleaned_part: Optional[str] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    filtered_results: Optional[List[Dict[str, Any]]] = None
    remanufacturer_variants: Optional[List[Dict[str, Any]]] = None
    agent_messages: Optional[List[str]] = None
    description_match_found: Optional[bool] = None
    error_reason: Optional[str] = None
    failure_details: Optional[Dict[str, Any]] = None
    early_stopping: Optional[bool] = None
    match_type: Optional[str] = None
    candidate: Optional[str] = None
    # Batch processing response fields
    total_processed: Optional[int] = None
    successful: Optional[int] = None
    failed: Optional[int] = None
    results: Optional[List[Dict[str, Any]]] = None
    # File processing response fields
    download_url: Optional[str] = None
    csv_data: Optional[str] = None

class PartNumberResponse(BaseModel):
    """
    Response model for part number processing.
    
    The model includes detailed scoring information for each result:
    - part_number_score: 0-5 score based on matching characters / total input characters
    - description_score: 0-5 score based on matching words / total input description words
    - noise: boolean indicating if the matched part contains extra characters
    - cocode: two-digit confidence code combining part and description scores
    
    See /api/scoring/info endpoint for detailed scoring documentation.
    """
    status: str
    cleaned_part: str
    search_results: List[Dict[str, Any]]
    filtered_results: List[Dict[str, Any]]
    remanufacturer_variants: List[Dict[str, Any]]
    agent_messages: List[str]
    execution_time: float = 0.0
    description_match_found: Optional[bool] = None
    error_reason: Optional[str] = None
    failure_details: Optional[Dict[str, Any]] = None
    early_stopping: Optional[bool] = None
    match_type: Optional[str] = None
    candidate: Optional[str] = None

class BatchProcessingResponse(BaseModel):
    """
    Response model for batch processing multiple parts.
    
    Each result in the results list includes the same scoring information as 
    the single part processing response.
    """
    status: str
    total_processed: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    execution_time: float = 0.0
    download_url: Optional[str] = None

class ParallelConfigRequest(BaseModel):
    """
    Request model for configuring parallel processing.
    """
    enabled: bool
    max_workers: Optional[int] = None

class ParallelConfigResponse(BaseModel):
    """
    Response model for parallel processing configuration.
    """
    status: str
    parallel_enabled: bool
    max_workers: int

# Initialize FastAPI app
app = FastAPI(
    title="Motor Part API (Databricks Version)",
    description="API for processing motor part numbers and descriptions, optimized for Databricks deployment",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if available
if paths["static_dir"].exists():
    app.mount("/static", StaticFiles(directory=str(paths["static_dir"])), name="static")

# Initialize global variables
settings = get_settings()
db_manager = DatabaseManager()
part_processor = PartProcessor(db_manager)
file_processor = FileProcessor(db_manager, part_processor)

# Use threadpool for parallel processing
executor = ThreadPoolExecutor(max_workers=settings.max_workers if settings.parallel_enabled else 1)

# API Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint that redirects to the static HTML interface.
    """
    # Check if we're in Databricks and adjust the static HTML path accordingly
    if paths["in_databricks"]:
        # In Databricks, we might need to use a different path or approach
        static_html_path = paths["static_dir"] / "index.html"
        if static_html_path.exists():
            with open(static_html_path, "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            return RedirectResponse(url="/api/docs")
    else:
        return RedirectResponse(url="/static/index.html")

# API Documentation custom endpoint
@app.get("/api", response_class=HTMLResponse)
async def api_docs():
    """
    API documentation endpoint that redirects to the ReDoc interface.
    """
    return RedirectResponse(url="/api/redoc")

# Health check endpoint
@app.get("/api/ping")
async def ping():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy", "message": "Part Number Processing API is running"}

@app.get("/api/health")
async def api_health_check():
    """
    Detailed health check endpoint with timestamp.
    """
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/api/scoring/info", tags=["Scoring"])
async def scoring_system_info():
    """
    Get information about the scoring system used for part and description matching.
    
    This endpoint provides detailed documentation about how part numbers and descriptions
    are scored and how confidence codes are generated.
    
    Returns:
        Information about the scoring system including examples
    """
    from tools.scoring_system import PartNumberScoringSystem
    
    return {
        "part_number_score": {
            "description": "Measures how well the input part number matches the database part number",
            "calculation": "matching_characters / total_input_characters * 5",
            "scale": "0-5 points",
            "example": {
                "input": "BBM1693",
                "matched": "1693",
                "calculation": "4 matching chars / 7 total input chars = 57.14% = 2.857 → 3/5 (rounded)"
            }
        },
        "description_score": {
            "description": "Measures how many words from the input description match the database description",
            "calculation": "matching_words / total_input_words * 5",
            "scale": "0-5 points",
            "example": {
                "input": "HOSE-COOLANT SLEEVE",
                "matched": "HOSE",
                "calculation": "1 matching word / 3 total input words = 33.33% = 1.667 → 1/5 (rounded)"
            }
        },
        "confidence_code": {
            "description": "A two-digit code combining the part number score and description score",
            "format": "first digit = part number score, second digit = description score",
            "example": {
                "part_score": 5,
                "description_score": 1,
                "code": "51"
            }
        },
        "noise_detection": {
            "description": "Indicates if the matched part number contains extra characters beyond the input",
            "values": "0 = no noise, 1 = noise detected",
            "example": {
                "input": "1693",
                "matched": "1693B",
                "result": "Noise = 1 (extra 'B')"
            }
        },
        "rounding_rule": {
            "description": "Custom rounding rule used in the scoring system",
            "rule": "Round up if decimal part >= 0.8, otherwise round down",
            "examples": [
                {"raw_score": 2.7, "rounded": 2},
                {"raw_score": 2.8, "rounded": 3},
                {"raw_score": 3.1, "rounded": 3}
            ]
        }
    }

@app.get("/api/stats")
async def system_stats():
    """
    Get system statistics about API configuration.
    """
    return {
        "settings": settings.dict(),
        "system": {
            "databricks": paths["in_databricks"],
            "paths": {k: str(v) for k, v in paths.items() if k != "in_databricks"}
        }
    }

# Configuration endpoints
@app.get("/api/config", tags=["Configuration"])
async def get_config():
    """
    Get the current API configuration settings.
    """
    return settings.dict()

@app.post("/api/config", tags=["Configuration"])
async def update_config(config: Dict[str, Any]):
    """
    Update API configuration settings.
    
    Args:
        config: Dictionary of configuration settings to update
    """
    global settings
    try:
        settings = update_settings(config)
        save_settings(settings)
        
        # Update processor settings
        part_processor.update_settings(settings)
        
        # Update threadpool if parallel settings changed
        if "parallel_enabled" in config or "max_workers" in config:
            global executor
            executor.shutdown(wait=True)
            executor = ThreadPoolExecutor(max_workers=settings.max_workers if settings.parallel_enabled else 1)
        
        return {"status": "success", "message": "Configuration updated", "settings": settings.dict()}
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=400, detail=f"Error updating configuration: {str(e)}")

@app.post("/api/config/parallel", tags=["Configuration"], response_model=ParallelConfigResponse)
async def configure_parallel_processing(config: ParallelConfigRequest):
    """
    Configure parallel processing settings.
    
    Args:
        config: Parallel processing configuration
    """
    global settings, executor
    
    try:
        # Update settings
        settings_dict = settings.dict()
        settings_dict["parallel_enabled"] = config.enabled
        if config.max_workers is not None:
            settings_dict["max_workers"] = max(1, min(32, config.max_workers))  # Limit between 1-32
        
        # Apply settings
        settings = update_settings(settings_dict)
        save_settings(settings)
        
        # Recreate executor with new settings
        executor.shutdown(wait=True)
        executor = ThreadPoolExecutor(max_workers=settings.max_workers if settings.parallel_enabled else 1)
        
        return ParallelConfigResponse(
            status="success",
            parallel_enabled=settings.parallel_enabled,
            max_workers=settings.max_workers
        )
    except Exception as e:
        logger.error(f"Error configuring parallel processing: {e}")
        raise HTTPException(status_code=400, detail=f"Error configuring parallel processing: {str(e)}")

# Process part endpoints
@app.post("/api/process-part", tags=["Part Processing"], response_model=PartNumberResponse)
async def process_part(request: PartNumberRequest):
    """
    Process a single part number with optional description.
    
    Args:
        request: Part number request with optional description
    
    Returns:
        Processed part information with search results and confidence scores
    """
    start_time = timer()
    
    try:
        # Process the part
        result = part_processor.process_part(
            request.part_number, 
            description=request.part_description
        )
        
        # Record execution time
        execution_time = timer() - start_time
        
        # Add execution time to result
        result["execution_time"] = execution_time
        
        return result
    except Exception as e:
        logger.error(f"Error processing part: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error processing part: {str(e)}")

@app.post("/api/batch-process", tags=["Batch Processing"], response_model=BatchProcessingResponse)
async def batch_process(request: BatchProcessingRequest):
    """
    Process multiple parts in a batch.
    
    Args:
        request: Batch processing request with list of parts
    
    Returns:
        Batch processing results with success/failure counts
    """
    start_time = timer()
    
    try:
        parts = request.parts
        total = len(parts)
        
        if total == 0:
            return {
                "status": "error",
                "total_processed": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "execution_time": 0.0
            }
        
        # Process parts based on parallel configuration
        if settings.parallel_enabled:
            # Parallel processing
            futures = []
            for part_data in parts:
                part_number = part_data.get("part_number", "")
                description = part_data.get("description", None)
                
                future = executor.submit(
                    part_processor.process_part,
                    part_number,
                    description=description
                )
                futures.append((part_number, description, future))
            
            # Collect results
            results = []
            successful = 0
            failed = 0
            
            for part_number, description, future in futures:
                try:
                    result = future.result()
                    result["input"] = {
                        "part_number": part_number,
                        "description": description
                    }
                    successful += 1
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing part {part_number}: {e}")
                    failed += 1
                    results.append({
                        "status": "error",
                        "input": {
                            "part_number": part_number,
                            "description": description
                        },
                        "error_reason": str(e)
                    })
        else:
            # Sequential processing
            results = []
            successful = 0
            failed = 0
            
            for part_data in parts:
                part_number = part_data.get("part_number", "")
                description = part_data.get("description", None)
                
                try:
                    result = part_processor.process_part(
                        part_number,
                        description=description
                    )
                    result["input"] = {
                        "part_number": part_number,
                        "description": description
                    }
                    successful += 1
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing part {part_number}: {e}")
                    failed += 1
                    results.append({
                        "status": "error",
                        "input": {
                            "part_number": part_number,
                            "description": description
                        },
                        "error_reason": str(e)
                    })
        
        # Record execution time
        execution_time = timer() - start_time
        
        return {
            "status": "success",
            "total_processed": total,
            "successful": successful,
            "failed": failed,
            "results": results,
            "execution_time": execution_time
        }
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error in batch processing: {str(e)}")

@app.post("/api/file-process", tags=["File Processing"])
async def file_process(request: FileProcessingRequest, background_tasks: BackgroundTasks):
    """
    Process parts from a file (CSV, Excel, etc.).
    
    Args:
        request: File processing request with base64 encoded file content
        background_tasks: FastAPI background tasks object for async processing
    
    Returns:
        File processing results with download URL for results
    """
    start_time = timer()
    
    try:
        # Decode file data
        try:
            file_data = base64.b64decode(request.file_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid file data: {str(e)}")
        
        # Save file to temp location
        fd, temp_file_path = tempfile.mkstemp(suffix=f"_{request.filename}")
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(file_data)
        
        # Process file
        output_format = request.output_format.lower() if request.output_format else "json"
        if output_format not in ["json", "csv"]:
            output_format = "json"
        
        # Custom handling for Databricks environment
        if paths["in_databricks"]:
            # In Databricks, we might need to use a different approach for temporary files
            # and result storage
            
            # Process the file
            result = file_processor.process_file(
                temp_file_path, 
                output_format=output_format,
                parallel=settings.parallel_enabled,
                max_workers=settings.max_workers if settings.parallel_enabled else 1
            )
            
            # For Databricks, we can return the data directly or save to a specific DBFS location
            if output_format == "csv" and "csv_data" in result:
                # Generate a unique filename for DBFS
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                output_filename = f"results_{timestamp}_{random_str}.csv"
                output_path = paths["base_path"] / "results" / output_filename
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save CSV data
                with open(output_path, "w") as f:
                    f.write(result["csv_data"])
                
                # Create download URL (may need to be adjusted for Databricks)
                download_url = f"/api/download/{output_filename}"
                result["download_url"] = download_url
            
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Error removing temporary file: {e}")
            
            # Record execution time
            execution_time = timer() - start_time
            result["execution_time"] = execution_time
            
            return result
        else:
            # Standard approach for non-Databricks environment
            result = file_processor.process_file(
                temp_file_path, 
                output_format=output_format,
                parallel=settings.parallel_enabled,
                max_workers=settings.max_workers if settings.parallel_enabled else 1
            )
            
            # Handle CSV output
            if output_format == "csv" and "csv_data" in result:
                # Generate a unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                output_filename = f"results_{timestamp}_{random_str}.csv"
                output_dir = paths["static_dir"] / "downloads"
                os.makedirs(output_dir, exist_ok=True)
                output_path = output_dir / output_filename
                
                # Save CSV data
                with open(output_path, "w") as f:
                    f.write(result["csv_data"])
                
                # Create download URL
                download_url = f"/static/downloads/{output_filename}"
                result["download_url"] = download_url
            
            # Clean up temp file in background
            background_tasks.add_task(os.unlink, temp_file_path)
            
            # Record execution time
            execution_time = timer() - start_time
            result["execution_time"] = execution_time
            
            return result
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/api/process", tags=["Unified Processing"], response_model=UnifiedProcessResponse)
async def unified_process(request: UnifiedProcessRequest, background_tasks: BackgroundTasks):
    """
    Unified endpoint for processing parts (single, batch, or file).
    
    Args:
        request: Unified process request with operation type and relevant data
        background_tasks: FastAPI background tasks object for async processing
    
    Returns:
        Unified process response with operation-specific results
    """
    start_time = timer()
    
    try:
        operation = request.operation.lower()
        
        # Single part processing
        if operation == "single":
            if not request.part_number:
                raise HTTPException(status_code=400, detail="Part number is required for single processing")
            
            result = part_processor.process_part(
                request.part_number, 
                description=request.part_description
            )
            
            execution_time = timer() - start_time
            
            return {
                "operation": "single",
                "status": "success",
                "execution_time": execution_time,
                **result
            }
        
        # Batch processing
        elif operation == "batch":
            if not request.parts:
                raise HTTPException(status_code=400, detail="Parts list is required for batch processing")
            
            parts = request.parts
            total = len(parts)
            
            if total == 0:
                return {
                    "operation": "batch",
                    "status": "error",
                    "execution_time": 0.0,
                    "total_processed": 0,
                    "successful": 0,
                    "failed": 0,
                    "results": []
                }
            
            # Process parts based on parallel configuration
            if settings.parallel_enabled:
                # Parallel processing
                futures = []
                for part_data in parts:
                    part_number = part_data.get("part_number", "")
                    description = part_data.get("description", None)
                    
                    future = executor.submit(
                        part_processor.process_part,
                        part_number,
                        description=description
                    )
                    futures.append((part_number, description, future))
                
                # Collect results
                results = []
                successful = 0
                failed = 0
                
                for part_number, description, future in futures:
                    try:
                        result = future.result()
                        result["input"] = {
                            "part_number": part_number,
                            "description": description
                        }
                        successful += 1
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing part {part_number}: {e}")
                        failed += 1
                        results.append({
                            "status": "error",
                            "input": {
                                "part_number": part_number,
                                "description": description
                            },
                            "error_reason": str(e)
                        })
            else:
                # Sequential processing
                results = []
                successful = 0
                failed = 0
                
                for part_data in parts:
                    part_number = part_data.get("part_number", "")
                    description = part_data.get("description", None)
                    
                    try:
                        result = part_processor.process_part(
                            part_number,
                            description=description
                        )
                        result["input"] = {
                            "part_number": part_number,
                            "description": description
                        }
                        successful += 1
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing part {part_number}: {e}")
                        failed += 1
                        results.append({
                            "status": "error",
                            "input": {
                                "part_number": part_number,
                                "description": description
                            },
                            "error_reason": str(e)
                        })
            
            execution_time = timer() - start_time
            
            return {
                "operation": "batch",
                "status": "success",
                "execution_time": execution_time,
                "total_processed": total,
                "successful": successful,
                "failed": failed,
                "results": results
            }
        
        # File processing
        elif operation == "file":
            if not request.file_data or not request.filename:
                raise HTTPException(status_code=400, detail="File data and filename are required for file processing")
            
            # Decode file data
            try:
                file_data = base64.b64decode(request.file_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid file data: {str(e)}")
            
            # Save file to temp location
            fd, temp_file_path = tempfile.mkstemp(suffix=f"_{request.filename}")
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(file_data)
            
            # Process file
            output_format = request.output_format.lower() if request.output_format else "json"
            if output_format not in ["json", "csv"]:
                output_format = "json"
            
            # Handle Databricks environment if needed
            file_result = file_processor.process_file(
                temp_file_path, 
                output_format=output_format,
                parallel=settings.parallel_enabled,
                max_workers=settings.max_workers if settings.parallel_enabled else 1
            )
            
            # Handle CSV output
            if output_format == "csv" and "csv_data" in file_result:
                # Generate a unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                output_filename = f"results_{timestamp}_{random_str}.csv"
                
                if paths["in_databricks"]:
                    # Databricks path handling
                    output_dir = paths["base_path"] / "results"
                    download_url = f"/api/download/{output_filename}"
                else:
                    # Standard path handling
                    output_dir = paths["static_dir"] / "downloads"
                    download_url = f"/static/downloads/{output_filename}"
                
                os.makedirs(output_dir, exist_ok=True)
                output_path = output_dir / output_filename
                
                # Save CSV data
                with open(output_path, "w") as f:
                    f.write(file_result["csv_data"])
                
                file_result["download_url"] = download_url
            
            # Clean up temp file in background
            background_tasks.add_task(os.unlink, temp_file_path)
            
            execution_time = timer() - start_time
            
            return {
                "operation": "file",
                "status": "success",
                "execution_time": execution_time,
                **file_result
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid operation: {operation}")
    except Exception as e:
        logger.error(f"Error in unified processing: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error in unified processing: {str(e)}")

# Download file endpoint (used for file processing results)
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    Download a processed file result.
    
    Args:
        filename: The filename to download
    
    Returns:
        File response with the requested file
    """
    try:
        if paths["in_databricks"]:
            # Databricks path handling
            file_path = paths["base_path"] / "results" / filename
        else:
            # Standard path handling
            file_path = paths["static_dir"] / "downloads" / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="text/csv" if filename.endswith(".csv") else "application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

# Main Databricks entrypoint
def main():
    """
    Main entry point for running the FastAPI application in Databricks.
    """
    # Configure host and port for Databricks
    host = "0.0.0.0"  # Listen on all interfaces
    port = int(os.getenv("PORT", "8000"))  # Default to port 8000
    
    logger.info(f"Starting Motor Part API (Databricks) on {host}:{port}")
    logger.info(f"Running in Databricks: {paths['in_databricks']}")
    
    # Start the FastAPI application using uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
