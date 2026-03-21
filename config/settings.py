from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SAMPLE_DATA_PATH = os.getenv("SAMPLE_DATA_PATH", "./sample_data")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./output/generated")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
MAX_SAMPLE_ROWS = int(os.getenv("MAX_SAMPLE_ROWS", "10000"))
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"