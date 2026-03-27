"""Logging utilities for the ICL project.

This module provides consistent logging configuration across the project.
"""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str, 
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup a logger with consistent formatting.
    
    Args:
        name: Name of the logger (typically __name__ of the calling module)
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Optional custom format string. If None, uses default.
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from icl.utils.logger import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("Starting process...")
        >>> logger.debug("Debug information")  # Only shows if level=DEBUG
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name.
    
    This is useful when you want to retrieve a logger that was already
    configured with setup_logger().
    
    Args:
        name: Name of the logger to retrieve
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
