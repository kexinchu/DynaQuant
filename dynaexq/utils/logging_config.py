"""
Logging configuration for DynaExq.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    format_string: str | None = None,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )

