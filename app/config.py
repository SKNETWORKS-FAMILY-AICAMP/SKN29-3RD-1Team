import logging
import os

from dotenv import load_dotenv


load_dotenv()


LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO"
).upper()

ENABLE_LOCAL_MODEL = os.getenv(
    "ENABLE_LOCAL_MODEL", 
    "False"
).lower() == "true"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=(
        "%(asctime)s "
        "[%(levelname)s] "
        "%(name)s - "
        "%(message)s"
    ),
)

logger = logging.getLogger("app")