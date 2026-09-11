from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'piri.db'}")
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PAIMANA_API_URL = os.getenv("PAIMANA_API_URL", "")
PAIMANA_API_TOKEN = os.getenv("PAIMANA_API_TOKEN", "")
