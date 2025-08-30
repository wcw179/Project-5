import sys

from loguru import logger

from src.config import BASE_DIR, config

# Define log file path
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file_path = LOG_DIR / "app.log"

# Remove default handler
logger.remove()

# Define console log format
console_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
)

# Add a console logger
logger.add(
    sys.stdout,
    level=config.logging.level.upper(),
    format=console_format,
    colorize=True,
)

# Add a file logger
if config.logging.format == "json":
    logger.add(
        log_file_path,
        level=config.logging.level.upper(),
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        serialize=True,  # This enables JSON logging
        enqueue=True,  # Make logging non-blocking
        backtrace=True,
        diagnose=True,
    )
else:
    logger.add(
        log_file_path,
        level=config.logging.level.upper(),
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

# Export the configured logger
__all__ = ["logger"]
