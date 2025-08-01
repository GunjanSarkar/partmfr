"""
Shared data models for the Motor Parts API.
"""

from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union


class PartNumberRequest(BaseModel):
    """Request model for single part number processing"""
    part_number: str
    part_description: Optional[str] = None


class PartNumberResponse(BaseModel):
    """Response model for single part number processing"""
    status: str
    part_number: str
    cleaned_part: str
    filtered_results: List[Dict[str, Any]]
    remanufacturer_variants: List[Dict[str, Any]]
    execution_time: float
    error_reason: Optional[str] = None


class UnifiedProcessResponse(BaseModel):
    """
    Unified response model for all processing operations.
    
    This can handle:
    - Single part processing
    - Batch file processing 
    - CSV generation
    """
    status: str
    message: str
    operation_type: str  # "single_part", "batch_processing", "csv_generation"
    
    # Single part processing fields
    part_number: Optional[str] = None
    cleaned_part: Optional[str] = None
    filtered_results: Optional[List[Dict[str, Any]]] = None
    remanufacturer_variants: Optional[List[Dict[str, Any]]] = None
    error_reason: Optional[str] = None
    
    # Batch processing fields
    total_processed: Optional[int] = None
    successful: Optional[int] = None
    failed: Optional[int] = None
    results: Optional[List[Dict[str, Any]]] = None
    
    # Common fields
    execution_time: float
    timestamp: Optional[str] = None


class UnifiedRequest(BaseModel):
    """Request model for the unified endpoint that can handle both single parts and batch processing"""
    operation_type: str  # "single_part" or "batch"
    part_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None


class UnifiedResponse(BaseModel):
    """Response model for the unified endpoint"""
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None
    execution_time: float
    operation_type: str
