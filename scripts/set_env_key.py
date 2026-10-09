"""Write an API key into .env without it appearing on screen or in shell history.

    python scripts/set_env_key.py                       # SHARADAR_API_KEY (default)
    python scripts/set_env_key.py SHARADAR_SAMPLE_API_KEY

Paste the key at the hidden prompt and press Enter. Only its length is printed.
"""
import getpass
import os
import re
import sys
from pathlib import Path

name = sys.argv[1] if len(sys.argv) > 1 else "SHARADAR_API_KEY"
env = Path(__file__).resolve().parents[1] / ".env"
key = getpass.getpass(f"Paste {name} (hidden), then press Enter: ")
key = re.sub(r"\x1b\[[0-9;]*~", "", key).strip().strip("'\"").strip()   # drop paste markers/quotes
if not key or any(c.isspace() for c in key):
    sys.exit("Nothing saved: the key was empty or contained spaces. Run again and paste only the key.")
text = env.read_text() if env.exists() else ""
line = f"{name}={key}"
if re.search(rf"(?m)^{re.escape(name)}=.*$", text):
    text = re.sub(rf"(?m)^{re.escape(name)}=.*$", lambda m: line, text)
else:
    text = text.rstrip("\n") + ("\n" if text else "") + line + "\n"
env.write_text(text)
os.chmod(env, 0o600)
print(f"Saved {name} to .env ({len(key)} characters).")
