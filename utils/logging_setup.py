"""
utils.logging_setup: Colored logging configuration
"""
import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console logging."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = colored_levelname

        # Format the message
        formatted = super().format(record)

        # Reset levelname for other handlers
        record.levelname = levelname

        return formatted


def configure_logger(verbose: int = 0, log_file: Optional[str] = None) -> None:
    """
    Configure logging with appropriate verbosity and optional file output.

    Args:
        verbose: Verbosity level (0=INFO, 1=DEBUG, 2+=more detailed DEBUG)
        log_file: Optional path to log file
    """
    # Determine log level based on verbosity
    if verbose == 0:
        console_level = logging.INFO
    elif verbose == 1:
        console_level = logging.DEBUG
    else:  # verbose >= 2
        console_level = logging.DEBUG

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)

    # Format strings
    if verbose >= 2:
        console_format = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s"
    else:
        console_format = "[%(asctime)s] %(levelname)s %(message)s"

    # Create formatters
    colored_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S")
    console_handler.setFormatter(colored_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        file_format = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_file}")

    # Reduce verbosity of third-party libraries
    if verbose < 2:
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("torchvision").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.debug(f"Logging configured with verbosity level {verbose}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


# Context manager for temporary log level changes
class LogLevel:
    """Context manager to temporarily change log level."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
