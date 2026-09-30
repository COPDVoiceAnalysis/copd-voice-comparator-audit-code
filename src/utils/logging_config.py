"""
Centralized logging configuration for the voice biomarker pipeline.
Provides application-level logging setup following Python best practices.

LOGGING BEST PRACTICES:
======================

1. STANDARD PATTERN (Recommended for most modules):
   --------------------------------------------------
   In your module:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   
   # Use logger throughout your module:
   logger.info("Processing data...")
   logger.warning("Something unexpected...")
   logger.error("Error occurred!")
   ```
   
   In your main/entry point (call ONCE at startup):
   ```python
   from src.utils.logging_config import setup_application_logging
   
   setup_application_logging(log_file_path="path/to/logfile.log")
   ```

2. SIMPLE SCRIPTS (Standalone utilities without file logging):
   -----------------------------------------------------------
   ```python
   from src.utils.logging_config import setup_basic_logging
   import logging
   
   setup_basic_logging()
   logger = logging.getLogger(__name__)
   ```

WHY THIS APPROACH?
==================
- Module-level loggers (logging.getLogger(__name__)) automatically inherit configuration
- Centralized setup ensures consistent formatting across all modules
- Logger names show which module emitted each log message
- Easy to filter logs by module or adjust levels per module if needed
"""

import logging
import sys


def setup_application_logging(
    level: int = logging.INFO,
    log_file_path: str | None = None,
) -> None:
    """
    Set up application-wide logging configuration.
    
    This should be called ONCE at application startup (e.g., in main()).
    All modules will inherit this configuration through the logging hierarchy.

    Args:
        level: Logging level for the entire application
        log_file_path: Path for file handler (if provided, file logging will be enabled)
        
    Example:
        ```python
        from src.utils.logging_config import setup_application_logging
        
        setup_application_logging(log_file_path="logs/training.log")
        
        # Now all module loggers will use this configuration
        logger.info("Application started")  # Will log to both console and file
        ```
    """
    # Create handlers list
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file_path:
        handlers.append(logging.FileHandler(log_file_path))
    
    # Configure root logger for entire application
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True  # Override any existing configuration
    )


def setup_basic_logging(level: int = logging.INFO) -> None:
    """
    Set up basic logging configuration for simple scripts.
    
    Use this for standalone utilities or scripts that don't need file logging.
    For full applications with file logging, use setup_application_logging instead.
    
    Args:
        level: Logging level (default: INFO)
        
    Example:
        ```python
        from src.utils.logging_config import setup_basic_logging
        import logging
        
        setup_basic_logging()
        logger = logging.getLogger(__name__)
        logger.info("Script started")
        ```
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True
    )
