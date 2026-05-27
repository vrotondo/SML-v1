import os

MODEL = "claude-sonnet-4-6"
DB_PATH = os.environ.get("LEGAL_DB_PATH", "legal_firm.db")
MAX_TOKENS = 4096
MAX_AGENT_ITERATIONS = 15
