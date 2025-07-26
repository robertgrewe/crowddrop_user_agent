# logging_config.py

import logging
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables (ensure this file can also load them if run independently for testing)
load_dotenv(find_dotenv())

def configure_logging():
    """
    Configures the application's logging based on environment variables.
    """
    # Get log level from environment variable, default to INFO if not set
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    
    numeric_level = getattr(logging, LOG_LEVEL, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {LOG_LEVEL}")

    # Configure the root logger
    # This ensures all loggers (including those created with getLogger(__name__)) inherit this config
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() # Output logs to console
        ]
    )
    # Set uvicorn's access log level to match, so you don't get duplicate access logs if set to INFO
    # or higher, but still get them if set to DEBUG.
    logging.getLogger("uvicorn.access").setLevel(numeric_level)
    logging.getLogger("uvicorn.error").setLevel(numeric_level)

    # Inform that logging has been configured
    logging.info(f"Logging configured with level: {LOG_LEVEL}")

# Call the configuration function immediately when this module is imported
configure_logging()
