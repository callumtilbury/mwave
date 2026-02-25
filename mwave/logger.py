import logging

# Initialize the logger
logger = logging.getLogger(__name__.split(".")[0])
logger.setLevel(logging.INFO)

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter(
    '%(asctime)s - %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


# Function to set logging level
def set_logging_level(level: int) -> None:
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
