"""Configuration loading. All thresholds live in config/*.yaml."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the project root: $SCREENER_HOME, else the nearest parent with config/settings.yaml."""
    env = os.getenv("SCREENER_HOME")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "config" / "settings.yaml").exists():
            return p
    pkg_root = Path(__file__).resolve().parent.parent
    if (pkg_root / "config" / "settings.yaml").exists():
        return pkg_root
    raise FileNotFoundError("Could not find config/settings.yaml. Run from the project folder "
                            "or set SCREENER_HOME.")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dotenv_file(root: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing environment variables win."""
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


@dataclass
class Config:
    root: Path
    settings: Dict[str, Any]
    flags: Dict[str, Any]
    profiles: Dict[str, Any]
    screens: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ---- paths -------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.root / self.settings["paths"]["data_dir"]

    @property
    def outputs_dir(self) -> Path:
        return self.root / self.settings["paths"]["outputs_dir"]

    @property
    def compliance_dir(self) -> Path:
        return self.root / self.settings["paths"]["compliance_dir"]

    # ---- providers ---------------------------------------------------------
    def provider_settings(self, provider: str) -> Dict[str, Any]:
        provs = self.settings.get("providers", {})
        if provider not in provs:
            raise KeyError(f"Unknown provider '{provider}'. Known: {', '.join(provs)}")
        return provs[provider]

    def retention(self, provider: str) -> str:
        return self.provider_settings(provider).get("license", {}).get("retention", "delete_on_cancel")

    def is_licensed(self, provider: str) -> bool:
        """True when the provider's raw data may not be published or kept after cancellation."""
        return self.retention(provider) != "public_domain"

    @property
    def active_provider(self) -> str:
        return self.settings.get("active_provider", "synthetic")

    # ---- misc --------------------------------------------------------------
    def fingerprint(self) -> str:
        """Hash of every config value — stamped on each run so results are reproducible."""
        blob = json.dumps({"settings": self.settings, "flags": self.flags,
                           "profiles": self.profiles, "screens": self.screens},
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def copy(self) -> "Config":
        return copy.deepcopy(self)


def load_config(root: Optional[Path] = None) -> Config:
    root = Path(root).resolve() if root else find_project_root()
    load_dotenv_file(root)
    cfg_dir = root / "config"
    settings = _load_yaml(cfg_dir / "settings.yaml")
    flags = _load_yaml(cfg_dir / "flags.yaml")
    profiles = _load_yaml(cfg_dir / "industry_profiles.yaml").get("profiles", {})
    screens = {}
    for p in sorted((cfg_dir / "screens").glob("*.yaml")):
        spec = _load_yaml(p)
        screens[spec.get("name", p.stem)] = spec
    return Config(root=root, settings=settings, flags=flags, profiles=profiles, screens=screens)
