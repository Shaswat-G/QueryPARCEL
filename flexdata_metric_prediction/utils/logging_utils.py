"""Simple file and console logging utility for training runs."""

import logging
import sys
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """Minimal logger that writes to both console and file.

    Args:
        log_dir: Directory to save log files
        log_name: Name for the log file (default: timestamp-based)
        level: Logging level (default: INFO)
    """

    def __init__(
        self,
        log_dir: str | Path,
        log_name: str | None = None,
        level: int = logging.INFO,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"training_{timestamp}.log"

        self.log_path = self.log_dir / log_name

        # Configure root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_path, mode="w")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Logging initialized: {self.log_path}")

    def get_log_path(self) -> Path:
        """Return path to log file."""
        return self.log_path


def setup_training_logger(log_dir: str | Path, log_name: str | None = None) -> TrainingLogger:
    """Convenience function to setup logger.

    Args:
        log_dir: Directory to save log files
        log_name: Optional log filename

    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(log_dir=log_dir, log_name=log_name)
