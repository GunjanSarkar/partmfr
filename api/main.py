from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union
import json
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import io
import csv
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import re
import logging

from src.processor import PartProcessor
from src.file_processor import FileProcessor
from api.models import PartNumberRequest, PartNumberResponse, UnifiedProcessResponse

load_dotenv()

# Create an instance of the PartProcessor
part_processor = PartProcessor()

tags_metadata = [
    {
        "name": "Processing",
        "description": "Operations for processing part numbers including single, batch, and file processing"
    },
    {
        "name": "Configuration",
        "description": "API configuration and settings management"
    },
    {
        "name": "Health & Stats",
        "description": "System health checks and statistics"
    },
    {
        "name": "Scoring",
        "description": "Information about the scoring system used for part and description matching"
    }
]

app = FastAPI(
    title="Unified Motor Part Processing API",
    description="""
    A comprehensive API for processing motor part numbers with bidirectional sliding window search.
    
    Features:
    - Advanced bidirectional sliding window search for improved part number matching
    - Clean part numbers by removing suffixes/prefixes
    - Match part numbers against database records
    - Filter parts by classification (M, O, V)
    - Accurate description scoring based on input word matches
    - Process single parts or batch operations
    - Support for file processing
    
    Scoring System:
    - Part Number Score (0-5): Based on matching characters / total input characters
    - Description Score (0-5): Based on matching words / total input words
    - Confidence Code: Two-digit code combining both scores (e.g. "51" = 5/5 part number, 1/5 description)
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "persistAuthorization": True
    },
    openapi_tags=tags_metadata
)

# Include unified endpoint router
try:
    from api.unified_endpoint import router as unified_router
    app.include_router(unified_router)
    print("✅ Unified endpoint router loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load unified endpoint: {e}")

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    # Create downloads directory if it doesn't exist
    downloads_dir = static_dir / "downloads"
    if not downloads_dir.exists():
        downloads_dir.mkdir(parents=True)
    
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class BatchProcessingRequest(BaseModel):
    parts: List[Dict[str, str]]  # List of {"part_number": str, "description": str}

class FileProcessingRequest(BaseModel):
    file_data: str  # Base64 encoded file content
    filename: str
    output_format: Optional[str] = "json"  # json or csv

class UnifiedProcessRequest(BaseModel):
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

class BatchProcessingResponse(BaseModel):
    """
    Response model for batch processing.
    
    Each result in the results list contains detailed scoring information:
    - part_number_score: 0-5 score based on matching characters / total input characters
    - description_score: 0-5 score based on matching words / total input description words
    - noise: boolean indicating if the matched part contains extra characters
    - cocode: two-digit confidence code combining part and description scores
    
    See /api/scoring/info endpoint for detailed scoring documentation.
    """
    status: str
    total_processed: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    execution_time: float = 0.0
    download_url: Optional[str] = None

class ParallelConfigRequest(BaseModel):
    enabled: bool
    max_workers: Optional[int] = None

class ParallelConfigResponse(BaseModel):
    status: str
    parallel_enabled: bool
    max_workers: int
    performance_stats: Dict[str, Any]

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = static_dir / "index.html"
    if html_path.exists():
        try:
            # Specify UTF-8 encoding explicitly to handle special characters
            return HTMLResponse(content=html_path.read_text(encoding='utf-8'), status_code=200)
        except UnicodeDecodeError:
            # Fall back to Latin-1 which can handle all byte values
            return HTMLResponse(content=html_path.read_text(encoding='latin-1'), status_code=200)
    return HTMLResponse(content="<h1>Motor Part Lookup System</h1><p>Index page not found.</p>", status_code=404)

@app.post("/api/process", response_model=UnifiedProcessResponse, tags=["Processing"])
async def unified_process(request: UnifiedProcessRequest):
    """
    Unified endpoint for all processing operations: single part, batch, or file processing.
    
    Parameters:
    - operation: Type of operation ("single", "batch", or "file")
    - part_number: For single part processing
    - parts: For batch processing (list of part numbers and descriptions)
    - file_data: For file processing (base64 encoded file content)
    
    Returns:
    - Processed results based on the operation type
    - Cleaned and matched part numbers
    - Classification information (M, O, V)
    - Part descriptions and additional metadata
    - Confidence scores for each match:
      - part_number_score (0-5): Based on matching characters / total input characters
      - description_score (0-5): Based on matching words / total input description words
      - noise: Boolean indicating if matched part contains extra characters
      - cocode: Two-digit confidence code combining part and description scores
    
    For detailed scoring information, see the /api/scoring/info endpoint.
    
    Args:
        request: UnifiedProcessRequest containing operation type and relevant data
        
    Returns:
        UnifiedProcessResponse with operation-specific results
    """
    start_time = time.time()
    
    try:
        if request.operation == "single":
            # Single part processing
            if not request.part_number:
                raise HTTPException(status_code=400, detail="part_number is required for single operation")
            
            result = await process_single_part_unified(request.part_number, request.part_description)
            execution_time = time.time() - start_time
            
            return UnifiedProcessResponse(
                operation="single",
                status=result["status"],
                execution_time=execution_time,
                cleaned_part=result["cleaned_part"],
                search_results=result["search_results"],
                filtered_results=result["filtered_results"],
                remanufacturer_variants=result["remanufacturer_variants"],
                agent_messages=result["agent_messages"],
                description_match_found=result.get("description_match_found"),
                error_reason=result.get("error_reason"),
                failure_details=result.get("failure_details"),
                early_stopping=result.get("early_stopping"),
                match_type=result.get("match_type"),
                candidate=result.get("candidate")
            )
            
        elif request.operation == "batch":
            # Batch processing
            if not request.parts:
                raise HTTPException(status_code=400, detail="parts list is required for batch operation")
            
            batch_results = []
            successful = 0
            failed = 0
            
            # Process parts in chunks
            chunk_size = 10
            total_parts = len(request.parts)
            
            for i in range(0, total_parts, chunk_size):
                chunk = request.parts[i:i + chunk_size]
                
                # Process chunk asynchronously
                chunk_tasks = []
                for part_data in chunk:
                    task = asyncio.create_task(
                        process_single_part_async(part_data, part_processor)
                    )
                    chunk_tasks.append(task)
                
                # Wait for chunk completion
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                
                # Process chunk results
                for j, result in enumerate(chunk_results):
                    if isinstance(result, Exception):
                        failed += 1
                        batch_results.append({
                            "part_number": chunk[j].get("part_number", ""),
                            "description": chunk[j].get("description", ""),
                            "error": str(result)
                        })
                    else:
                        batch_results.append(result)
                        if result.get("result", {}).get("status") == "success":
                            successful += 1
                        else:
                            failed += 1
            
            execution_time = time.time() - start_time
            
            return UnifiedProcessResponse(
                operation="batch",
                status="success",
                execution_time=execution_time,
                total_processed=total_parts,
                successful=successful,
                failed=failed,
                results=batch_results
            )
            
        elif request.operation == "file":
            # File processing
            if not request.file_data or not request.filename:
                raise HTTPException(status_code=400, detail="file_data and filename are required for file operation")
            
            import base64
            
            # Decode base64 file data
            try:
                file_content = base64.b64decode(request.file_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 file data: {str(e)}")
            
            # Process file
            file_extension = request.filename.split('.')[-1].lower()
            if file_extension not in ["csv", "xlsx", "xls"]:
                raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or Excel file.")
            
            # Extract part data from file
            try:
                part_data_list = FileProcessor.process_file(file_content, file_extension)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            # Process parts in batch
            batch_results = []
            successful = 0
            failed = 0
            
            # Process parts in chunks
            chunk_size = 10
            total_parts = len(part_data_list)
            
            for i in range(0, total_parts, chunk_size):
                chunk = part_data_list[i:i + chunk_size]
                
                # Process chunk asynchronously
                chunk_tasks = []
                for part_data in chunk:
                    task = asyncio.create_task(
                        process_single_part_async(part_data, part_processor)
                    )
                    chunk_tasks.append(task)
                
                # Wait for chunk completion
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                
                # Process chunk results
                for j, result in enumerate(chunk_results):
                    if isinstance(result, Exception):
                        failed += 1
                        batch_results.append({
                            "part_number": chunk[j].get("part_number", ""),
                            "description": chunk[j].get("description", ""),
                            "error": str(result)
                        })
                    else:
                        batch_results.append(result)
                        if result.get("result", {}).get("status") == "success":
                            successful += 1
                        else:
                            failed += 1
            
            execution_time = time.time() - start_time
            
            # Handle different output formats
            if request.output_format == "csv":
                # Create CSV data
                csv_data = create_csv_data(batch_results)
                
                return UnifiedProcessResponse(
                    operation="file",
                    status="success",
                    execution_time=execution_time,
                    total_processed=total_parts,
                    successful=successful,
                    failed=failed,
                    csv_data=csv_data
                )
            else:
                # Return JSON response
                return UnifiedProcessResponse(
                    operation="file",
                    status="success",
                    execution_time=execution_time,
                    total_processed=total_parts,
                    successful=successful,
                    failed=failed,
                    results=batch_results
                )
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid operation: {request.operation}. Must be 'single', 'batch', or 'file'")
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in unified process: {str(e)}\n{error_details}")
        
        return UnifiedProcessResponse(
            operation=request.operation,
            status="error",
            execution_time=time.time() - start_time,
            error_reason=str(e),
            failure_details={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": error_details
            }
        )

async def process_single_part_unified(part_number: str, part_description: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a single part number with the same logic as the original endpoint.
    
    Args:
        part_number: The part number to process
        part_description: Optional part description
        
    Returns:
        Dictionary containing processing results
    """
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        # Apply preprocessing for spaces directly
        original_part = part_number
        print(f"Processing part number: '{original_part}'")
        
        # Try preprocessing the part number to handle spaces
        if ' ' in original_part:
            cleaned_part = re.sub(r'\s+', '', original_part)
            print(f"Preprocessed part number: '{original_part}' -> '{cleaned_part}'")
            
            # Quick check if the cleaned part exists in database
            from src.database import db_manager
            loop = asyncio.get_event_loop()
            direct_match = await loop.run_in_executor(None, db_manager.search_by_spartnumber, cleaned_part)
            if direct_match:
                print(f"Found direct match after removing spaces: {len(direct_match)} results for '{cleaned_part}'")
                part_number = cleaned_part
                print(f"Using cleaned part number for processing: '{part_number}'")
            else:
                print(f"No direct match for cleaned part: '{cleaned_part}', proceeding with original")
                part_number = original_part
        else:
            part_number = original_part
        
        # Process the part using optimized processor with early termination
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            part_processor.process_part_with_early_termination,
            part_number,
            part_description
        )
        
        # Check if we have results - safely handle missing keys
        has_filtered_results = "filtered_results" in result and result["filtered_results"] and len(result["filtered_results"]) > 0
        has_remanufacturer_variants = "remanufacturer_variants" in result and result["remanufacturer_variants"] and len(result["remanufacturer_variants"]) > 0
        has_results = has_filtered_results or has_remanufacturer_variants
        
        if has_results:
            # Pass through early stopping information from sliding window search
            if isinstance(result, dict):
                early_stopping = result.get("early_stopping", False)
                match_type = result.get("match_type", None)
                candidate = result.get("candidate", None)
                
                if early_stopping:
                    print(f"Early stopping was applied with match type '{match_type}' for candidate '{candidate}'")
            else:
                early_stopping = False
                match_type = None
                candidate = None
                print(f"Warning: Expected dictionary but got {type(result)}")
            
            return {
                "status": "success",
                "cleaned_part": result.get("cleaned_part", ""),
                "search_results": result.get("search_results", []),
                "filtered_results": result.get("filtered_results", []),
                "remanufacturer_variants": result.get("remanufacturer_variants", []),
                "agent_messages": result.get("agent_messages", []),
                "description_match_found": result.get("description_match_found"),
                "early_stopping": early_stopping,
                "match_type": match_type,
                "candidate": candidate
            }
        else:
            # Determine failure reason based on processing result
            if not isinstance(result, dict):
                error_reason = "Result has unexpected format"
                cleaned_part = ""
                search_results = []
                filtered_results = []
                remanufacturer_variants = []
                agent_messages = [f"Error: Unexpected result type {type(result)}"]
            else:
                error_reason = result.get("error_reason", "No matches found")
                cleaned_part = result.get("cleaned_part", "")
                search_results = result.get("search_results", [])
                # Make sure filtered_results and remanufacturer_variants exist
                filtered_results = result.get("filtered_results", [])
                remanufacturer_variants = result.get("remanufacturer_variants", [])
                agent_messages = result.get("agent_messages", [])
            
            return {
                "status": "no_results",
                "cleaned_part": cleaned_part,
                "search_results": search_results,
                "filtered_results": filtered_results,
                "remanufacturer_variants": remanufacturer_variants,
                "agent_messages": agent_messages,
                "description_match_found": None if not isinstance(result, dict) else result.get("description_match_found"),
                "error_reason": error_reason,
                "failure_details": {
                    "part_number": part_number,
                    "part_description": part_description,
                    "reason": error_reason
                }
            }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing part: {str(e)}\n{error_details}")
        
        return {
            "status": "error",
            "cleaned_part": "",
            "search_results": [],
            "filtered_results": [],
            "remanufacturer_variants": [],
            "agent_messages": [f"Error: {str(e)}"],
            "description_match_found": None,
            "error_reason": str(e),
            "failure_details": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": error_details,
                "part_number": part_number,
                "part_description": part_description
            }
        }

def create_csv_data(batch_results: List[Dict[str, Any]]) -> str:
    """
    Create CSV data string from batch processing results.
    
    Args:
        batch_results: List of dictionaries containing part processing results
        
    Returns:
        CSV data as string
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header row
    header = [
        "original_part_number", 
        "description", 
        "status", 
        "cleaned_part",
        "description_match_found",
        "part_number", 
        "class", 
        "manufacturer",
        "part_description",
        "part_id",
        "confidence",
        "is_remanufacturer",
        "original_part",
        "similarity"
    ]
    writer.writerow(header)
    
    # Write data rows
    for item in batch_results:
        original_part = item.get("part_number", "")
        description = item.get("description", "")
        
        if "error" in item:
            # Error case
            writer.writerow([
                original_part, 
                description, 
                "error", 
                "", "", "", "", "", "", "", "false", "", ""
            ])
        else:
            # Success case
            result = item.get("result", {})
            if not isinstance(result, dict):
                writer.writerow([
                    original_part, description, "error", "", "false", "", "", "", "", "", "false", "", ""
                ])
            else:
                status = result.get("status", "")
                cleaned_part = result.get("cleaned_part", "")
                description_match_found = str(result.get("description_match_found", False)).lower()
                
                manufacturer_parts = result.get("filtered_results", [])
                remanufacturer_parts = result.get("remanufacturer_variants", [])
                
                if not manufacturer_parts and not remanufacturer_parts:
                    writer.writerow([
                        original_part, description, status, cleaned_part, description_match_found,
                        "", "", "", "", "", "false", "", ""
                    ])
                else:
                    # Add rows for manufacturer parts
                    for part in manufacturer_parts:
                        writer.writerow([
                            original_part, description, status, cleaned_part, description_match_found,
                            part.get("PARTNUMBER", ""), part.get("CLASS", ""), part.get("PARTMFR", ""),
                            part.get("partdesc", ""), part.get("PARTINDEX", ""),
                            f"{part.get('confidence', 0) * 100:.0f}%" if "confidence" in part else "",
                            "false", "", ""
                        ])
                    
                    # Add rows for remanufacturer parts
                    for part in remanufacturer_parts:
                        writer.writerow([
                            original_part, description, status, cleaned_part, description_match_found,
                            part.get("PARTNUMBER", ""), part.get("CLASS", ""), part.get("PARTMFR", ""),
                            part.get("partdesc", ""), part.get("PARTINDEX", ""), "",
                            "true", part.get("original_part", ""),
                            f"{part.get('similarity', 0) * 100:.0f}%" if "similarity" in part else ""
                        ])
    
    return output.getvalue()

@app.post("/api/process-part", response_model=PartNumberResponse, tags=["Processing"])
async def process_part_number(request: PartNumberRequest):
    """
    Process a single part number and optional description.
    
    This endpoint searches for matching parts and calculates confidence scores:
    - part_number_score (0-5): Based on matching characters / total input characters
    - description_score (0-5): Based on matching words / total input description words
    - noise: Boolean indicating if matched part contains extra characters
    - cocode: Two-digit confidence code combining part and description scores
    
    For detailed scoring information, see the /api/scoring/info endpoint.
    """
    start_time = time.time()
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500, 
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )
        
        # Apply preprocessing for spaces directly in the API
        original_part = request.part_number
        
        # Log the original input for debugging
        print(f"API received part number: '{original_part}'")
        
        # Try preprocessing the part number to handle spaces directly
        if ' ' in original_part:
            cleaned_part = re.sub(r'\s+', '', original_part)
            print(f"API preprocessed part number: '{original_part}' -> '{cleaned_part}'")
            
            # Quick check if the cleaned part exists in database (async)
            from src.database import db_manager
            loop = asyncio.get_event_loop()
            direct_match = await loop.run_in_executor(None, db_manager.search_by_spartnumber, cleaned_part)
            if direct_match:
                print(f"API found direct match after removing spaces: {len(direct_match)} results for '{cleaned_part}'")
                
                # If we find a match, use the cleaned part number
                part_number = cleaned_part
                print(f"Using cleaned part number for processing: '{part_number}'")
            else:
                print(f"No direct match for cleaned part: '{cleaned_part}', proceeding with original")
                part_number = original_part
        else:
            part_number = original_part
        
        # Process the part using optimized processor with early termination (async)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            part_processor.process_part_with_early_termination,
            part_number,
            request.part_description
        )
        
        # Add execution time to the result
        execution_time = time.time() - start_time
        result["execution_time"] = execution_time
        
        # Check if we have results
        has_results = (result["filtered_results"] and len(result["filtered_results"]) > 0) or (result["remanufacturer_variants"] and len(result["remanufacturer_variants"]) > 0)
        
        if has_results:
            # Pass through early stopping information from sliding window search
            # Ensure result is a dictionary before using get()
            if isinstance(result, dict):
                early_stopping = result.get("early_stopping", False)
                match_type = result.get("match_type", None)
                candidate = result.get("candidate", None)
                
                # Log if early stopping was applied
                if early_stopping:
                    print(f"API: Early stopping was applied with match type '{match_type}' for candidate '{candidate}'")
            else:
                # Handle case where result might be a list
                early_stopping = False
                match_type = None
                candidate = None
                print(f"Warning: Expected dictionary but got {type(result)}")
            
            return PartNumberResponse(
                status="success",
                part_number=request.part_number,
                cleaned_part=result["cleaned_part"],
                filtered_results=result["filtered_results"],
                remanufacturer_variants=result["remanufacturer_variants"],
                execution_time=execution_time,
                error_reason=result.get("error_reason")
            )
        else:
            # Determine failure reason based on processing result
            if not isinstance(result, dict):
                error_reason = "Result has unexpected format"
                cleaned_part = ""
                search_results = []
                filtered_results = []
                remanufacturer_variants = []
                agent_messages = [f"Error: Unexpected result type {type(result)}"]
            else:
                error_reason = result.get("error_reason", "No matches found")
                cleaned_part = result.get("cleaned_part", "")
                search_results = result.get("search_results", [])
                filtered_results = result.get("filtered_results", [])
                remanufacturer_variants = result.get("remanufacturer_variants", [])
                agent_messages = result.get("agent_messages", [])
            
            return PartNumberResponse(
                status="no_results",
                part_number=request.part_number,
                cleaned_part=cleaned_part,
                filtered_results=filtered_results,
                remanufacturer_variants=remanufacturer_variants,
                execution_time=execution_time,
                error_reason=error_reason
            )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing part: {str(e)}\n{error_details}")
        
        return PartNumberResponse(
            status="error",
            part_number=request.part_number,
            cleaned_part="",
            filtered_results=[],
            remanufacturer_variants=[],
            execution_time=time.time() - start_time,
            error_reason=str(e)
        )

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    Download a processed file from the static/downloads directory
    """
    file_path = static_dir / "downloads" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv"
    )

@app.post("/api/process-file")
async def process_file(
    file: UploadFile = File(...),
    output_format: str = Query("json", description="Output format: 'json' or 'csv'")
):
    """
    Process an uploaded Excel or CSV file containing part numbers
    
    Args:
        file: The uploaded file (CSV, XLSX, XLS)
        output_format: The desired output format ('json' or 'csv')
        
    Returns:
        JSON response or CSV file download based on output_format
    """
    start_time = time.time()
    try:
        # Check file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ["csv", "xlsx", "xls"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload a CSV or Excel file."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process file to extract part numbers and descriptions
        try:
            part_data_list = FileProcessor.process_file(file_content, file_extension)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Process parts in batch with optimized async processing
        batch_results = []
        successful = 0
        failed = 0
        
        # Process parts in chunks for better memory efficiency
        chunk_size = 10  # Process 10 parts at a time
        total_parts = len(part_data_list)
        
        print(f"Processing {total_parts} parts in chunks of {chunk_size}")
        
        for i in range(0, total_parts, chunk_size):
            chunk = part_data_list[i:i + chunk_size]
            chunk_start_time = time.time()
            
            # Process chunk asynchronously
            chunk_tasks = []
            for part_data in chunk:
                task = asyncio.create_task(
                    process_single_part_async(part_data, part_processor)
                )
                chunk_tasks.append(task)
            
            # Wait for chunk completion
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Process chunk results
            for j, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    print(f"Error processing part {chunk[j].get('part_number', 'unknown')}: {str(result)}")
                    failed += 1
                    batch_results.append({
                        "part_number": chunk[j].get("part_number", ""),
                        "description": chunk[j].get("description", ""),
                        "error": str(result)
                    })
                else:
                    batch_results.append(result)
                    if result.get("result", {}).get("status") == "success":
                        successful += 1
                    else:
                        failed += 1
            
            chunk_time = time.time() - chunk_start_time
            print(f"Processed chunk {i//chunk_size + 1}/{(total_parts + chunk_size - 1)//chunk_size} in {chunk_time:.2f}s")
        
        execution_time = time.time() - start_time
        
            # Create CSV data for download link
        csv_data = create_csv_data(batch_results)
        
        # Create unique filename for the download
        timestamp = int(time.time())
        base_filename = Path(file.filename).stem
        download_filename = f"{base_filename}_processed_{timestamp}.csv"
        
        # Save CSV data to static directory for download
        download_path = static_dir / "downloads"
        if not download_path.exists():
            download_path.mkdir(parents=True)
        
        csv_file_path = download_path / download_filename
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_data)
        
        # Create download URL
        download_url = f"/static/downloads/{download_filename}"
        
        # Return JSON response with download URL
        return BatchProcessingResponse(
            status="success",
            total_processed=len(part_data_list),
            successful=successful,
            failed=failed,
            results=batch_results,
            execution_time=execution_time,
            download_url=download_url  # Include download URL in response
        )    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/parallel", response_model=ParallelConfigResponse)
async def configure_parallel_execution(config: ParallelConfigRequest):
    """
    Configure parallel execution settings for database queries.
    
    This endpoint allows you to enable/disable parallel execution and adjust
    the number of worker threads for better performance tuning.
    """
    try:
        # Update parallel execution settings
        part_processor.set_parallel_execution(
            enabled=config.enabled,
            max_workers=config.max_workers
        )
        
        # Get current performance stats
        stats = part_processor.get_performance_stats()
        
        return ParallelConfigResponse(
            status="success",
            parallel_enabled=stats['parallel_enabled'],
            max_workers=stats['max_workers'],
            performance_stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure parallel execution: {str(e)}")

@app.get("/api/config/parallel", response_model=ParallelConfigResponse)
async def get_parallel_configuration():
    """
    Get current parallel execution configuration and performance statistics.
    """
    try:
        stats = part_processor.get_performance_stats()
        
        return ParallelConfigResponse(
            status="success",
            parallel_enabled=stats['parallel_enabled'],
            max_workers=stats['max_workers'],
            performance_stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get parallel configuration: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Part Number Processing API is running"}

@app.get("/api/health")
async def api_health_check():
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
    from tools.pattern_optimizer import PatternOptimizer
    from src.database import db_manager
    pattern_optimizer = PatternOptimizer()
    
    return {
        "processing": {
            "pattern_stats": pattern_optimizer.get_stats(),
            "patterns_used": len(pattern_optimizer.get_pattern_priority())
        },
        "database": {
            "connection_pool_max": db_manager.pool.max_connections,
            "cache_enabled": db_manager.cache.enabled,
            "cache_size": len(db_manager.cache.cache)
        },
        "timestamp": time.time()
    }

@app.get("/database-stats")
async def database_stats():
    from src.database import db_manager
    from config.settings import settings
    
    try:
        # Query through Databricks for database statistics
        count_query = f"SELECT COUNT(*) as count FROM {settings.databricks_table_name}"
        count_result = db_manager._execute_query(count_query)
        total_records = count_result[0]['count'] if count_result else 0
        
        classes_query = f"SELECT COUNT(DISTINCT class) as count FROM {settings.databricks_table_name}"
        classes_result = db_manager._execute_query(classes_query)
        unique_classes = classes_result[0]['count'] if classes_result else 0
        
        distribution_query = f"""
            SELECT class, COUNT(*) as count 
            FROM {settings.databricks_table_name} 
            GROUP BY class 
            ORDER BY count DESC 
            LIMIT 5
        """
        class_distribution = db_manager._execute_query(distribution_query)
        
        return {
            "total_records": total_records,
            "unique_classes": unique_classes,
            "class_distribution": [{"class": row.get('class', ''), "count": row.get('count', 0)} 
                                  for row in class_distribution],
            "processing_system": "Python procedural pipeline (no LangGraph)"
        }
    except Exception as e:
        return {"error": str(e)}

def create_csv_response(batch_results, original_filename):
    """
    Create a CSV file from batch processing results
    
    Args:
        batch_results: List of dictionaries containing part processing results
        original_filename: Original filename to use as a base for the CSV filename
        
    Returns:
        StreamingResponse object with CSV data
    """
    # Create CSV file in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header row with detailed information
    header = [
        "original_part_number", 
        "description", 
        "status", 
        "cleaned_part",
        "description_match_found",
        "part_number", 
        "class", 
        "manufacturer",
        "part_description",
        "part_id",
        "confidence",
        "is_remanufacturer",
        "original_part",
        "similarity"
    ]
    writer.writerow(header)
    
    # Write data rows
    for item in batch_results:
        original_part = item.get("part_number", "")
        description = item.get("description", "")
        
        if "error" in item:
            # Error case
            writer.writerow([
                original_part, 
                description, 
                "error", 
                "",  # cleaned_part
                "false",  # description_match_found
                "",  # part_number
                "",  # class
                "",  # manufacturer
                "",  # part_description
                "",  # part_id
                "",  # confidence
                "false",  # is_remanufacturer
                "",  # original_part
                "",  # similarity
                item.get("error", "")  # add error at the end
            ])
        else:
            # Success case
            result = item.get("result", {})
            # Ensure result is a dictionary and not a list
            if not isinstance(result, dict):
                print(f"Warning: Expected dictionary result but got {type(result)}")
                status = "error"
                cleaned_part = ""
                description_match_found = "false"
                manufacturer_parts = []
                remanufacturer_parts = []
            else:
                status = result.get("status", "")
                cleaned_part = result.get("cleaned_part", "")
                description_match_found = str(result.get("description_match_found", False)).lower()
                
                # Process manufacturer parts
                manufacturer_parts = result.get("filtered_results", [])
                
                # Process remanufacturer parts
                remanufacturer_parts = result.get("remanufacturer_variants", [])
            
            # If no parts found, add a single row with basic info
            if not manufacturer_parts and not remanufacturer_parts:
                writer.writerow([
                    original_part,
                    description,
                    status,
                    cleaned_part,
                    description_match_found,
                    "",  # part_number
                    "",  # class
                    "",  # manufacturer
                    "",  # part_description
                    "",  # part_id
                    "",  # confidence
                    "false",  # is_remanufacturer
                    "",  # original_part
                    ""   # similarity
                ])
            else:
                # Add rows for manufacturer parts
                for part in manufacturer_parts:
                    writer.writerow([
                        original_part,
                        description,
                        status,
                        cleaned_part,
                        description_match_found,
                        part.get("PARTNUMBER", ""),
                        part.get("CLASS", ""),
                        part.get("PARTMFR", ""),
                        part.get("partdesc", ""),
                        part.get("PARTINDEX", ""),
                        f"{part.get('confidence', 0) * 100:.0f}%" if "confidence" in part else "",
                        "false",  # is_remanufacturer
                        "",  # original_part
                        ""   # similarity
                    ])
                
                # Add rows for remanufacturer parts
                for part in remanufacturer_parts:                        writer.writerow([
                            original_part,
                            description,
                            status,
                            cleaned_part,
                            description_match_found,
                            part.get("PARTNUMBER", ""),
                            part.get("CLASS", ""),
                            part.get("PARTMFR", ""),
                            part.get("partdesc", ""),
                            part.get("PARTINDEX", ""),
                            "",  # confidence
                            "true",  # is_remanufacturer
                            part.get("original_part", ""),
                            f"{part.get('similarity', 0) * 100:.0f}%" if "similarity" in part else ""
                        ])
    
    # Reset stream position to the beginning
    output.seek(0)
    
    # Generate filename for download
    base_name = original_filename.rsplit('.', 1)[0]
    download_filename = f"{base_name}_results.csv"
    
    # Return streaming response
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={download_filename}"}
    )

async def process_single_part_async(part_data: Dict[str, Any], processor) -> Dict[str, Any]:
    """
    Process a single part asynchronously with optimized early termination.
    """
    try:
        # Extract part number and description
        part_number = part_data.get("part_number", "")
        description = part_data.get("description", "")
        
        # Process the part using optimized method with early termination
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            processor.process_part_with_early_termination,
            part_number,
            description
        )
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            print(f"Warning: Expected dictionary result but got {type(result)} for part {part_number}")
            result = {
                "status": "error",
                "cleaned_part": "",
                "search_results": [],
                "filtered_results": [],
                "remanufacturer_variants": [],
                "agent_messages": [f"Error: Unexpected result type {type(result)}"],
                "description_match_found": None,
                "error_reason": f"Unexpected result type: {type(result)}"
            }
        
        # Return structured result
        return {
            "part_number": part_number,
            "description": description,
            "result": result
        }
        
    except Exception as e:
        # Return error result
        return {
            "part_number": part_data.get("part_number", ""),
            "description": part_data.get("description", ""),
            "error": str(e)
        }

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],  # Allow your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
