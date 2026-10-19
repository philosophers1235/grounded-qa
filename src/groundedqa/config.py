import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    index_dir: Path = Path(".gqa_index")
    top_k: int = 5
    min_support: float = 0.5
    contact: str = "the team that maintains these documents"
    embedder: str | None = None
    effort: str = "low"
    max_tokens: int = 2000

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        defaults = cls()
        return cls(
            model=env.get("GQA_MODEL") or defaults.model,
            index_dir=Path(env.get("GQA_INDEX_DIR") or defaults.index_dir),
            top_k=int(env.get("GQA_TOP_K") or defaults.top_k),
            min_support=float(env.get("GQA_MIN_SUPPORT") or defaults.min_support),
            contact=env.get("GQA_CONTACT") or defaults.contact,
            embedder=env.get("GQA_EMBEDDER") or None,
        )
