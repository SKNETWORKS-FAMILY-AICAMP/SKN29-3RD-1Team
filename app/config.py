import logging
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

ENABLE_LOCAL_MODEL = os.getenv(
    "ENABLE_LOCAL_MODEL", 
    "False"
).lower() == "true"

QWEN_MODEL_NAME = os.getenv(
    "QWEN_MODEL_NAME",
    "Qwen/Qwen3-0.6B"
)

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(BASE_DIR / "chroma_db")
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger("app")

def get_llm_setting_string():
    if ENABLE_LOCAL_MODEL:
        return QWEN_MODEL_NAME + "_" + OPENAI_MODEL
    return OPENAI_MODEL