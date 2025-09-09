import logging
import os


def configure_logging() -> None:
    level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    logging.getLogger('uvicorn').setLevel(getattr(logging, level, logging.INFO))
    logging.getLogger('uvicorn.error').setLevel(getattr(logging, level, logging.INFO))
    logging.getLogger('uvicorn.access').setLevel(getattr(logging, level, logging.INFO))


