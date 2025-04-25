"""
File handling utilities for knowledge base.
"""
import os
import glob
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)

def list_documents(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    List document files in a directory.
    
    Args:
        directory: Directory path
        extensions: List of file extensions to include
        
    Returns:
        List of file paths
    """
    # Default extensions if not provided
    if extensions is None:
        extensions = [
            ".txt", ".md", ".pdf", ".docx", ".doc", 
            ".csv", ".xlsx", ".xls", ".html", ".htm"
        ]
    
    # Ensure directory exists
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    
    # Find files
    files = []
    for ext in extensions:
        # Use glob to find files with this extension
        pattern = os.path.join(directory, f"**/*{ext}")
        matching_files = glob.glob(pattern, recursive=True)
        files.extend(matching_files)
    
    # Sort files
    files.sort()
    
    logger.info(f"Found {len(files)} document files in {directory}")
    return files

def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata for a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file metadata
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {}
    
    # Get file info
    file_stat = os.stat(file_path)
    file_size = file_stat.st_size
    creation_time = file_stat.st_ctime
    modification_time = file_stat.st_mtime
    
    # Get file name and extension
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    
    metadata = {
        "file_path": os.path.abspath(file_path),
        "file_name": file_name,
        "file_extension": file_ext,
        "file_size_bytes": file_size,
        "file_size_kb": round(file_size / 1024, 2),
        "creation_time": creation_time,
        "modification_time": modification_time
    }
    
    return metadata

def check_supported_file(file_path: str, supported_extensions: List[str]) -> bool:
    """
    Check if file is supported.
    
    Args:
        file_path: Path to file
        supported_extensions: List of supported extensions
        
    Returns:
        True if file is supported
    """
    # Get file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    # Check if extension is supported
    return ext in supported_extensions

def create_directory_index(directory: str) -> Dict[str, Any]:
    """
    Create index of a document directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Dictionary with directory index information
    """
    # Check if directory exists
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return {}
    
    # List all files
    all_files = list_documents(directory)
    
    # Group by extension
    extension_groups = {}
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in extension_groups:
            extension_groups[ext] = []
        extension_groups[ext].append(file_path)
    
    # Create index
    index = {
        "directory": os.path.abspath(directory),
        "file_count": len(all_files),
        "extension_counts": {ext: len(files) for ext, files in extension_groups.items()},
        "extensions": list(extension_groups.keys()),
        "files": all_files
    }
    
    return index