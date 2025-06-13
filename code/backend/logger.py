import logging
import os

# configure logging
def configure_logging():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create a logger specifically for your ingestion pipeline
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)  # Set desired level

    # Prevent logs from propagating to the root logger
    logger.propagate = False

    # Create a file handler for this logger
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(logging.INFO)

    # Add console output too
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define custom log format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Attach the handler to your logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize the logger
logger = configure_logging()