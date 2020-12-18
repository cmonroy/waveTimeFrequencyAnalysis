import logging
logger = logging.getLogger(__name__)
if len(logger.handlers) == 0 :  # Avoid re-adding handlers (When script is re-run with spyder for instance)
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(c_handler)
logger.setLevel(logging.INFO)

