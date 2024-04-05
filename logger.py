# Create a logger instance with a specified name
import logging


logger = logging.getLogger("ModelRunner")

# Set the logger's level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.INFO)

# Create a handler for outputting log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)
