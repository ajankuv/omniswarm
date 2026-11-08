import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    omniroute_base_url: str
    db_path: str
    connect_timeout: float
    read_timeout: float
    max_connections: int
    cache_enabled: bool
    cache_ttl_seconds: float
    cache_min_confidence: str
    cache_max_entries: int
    savings_usd_per_mtok: float
    # provider prefixes models may come from; empty = no restriction
    allowed_providers: frozenset[str] = field(default_factory=frozenset)


def _providers(raw: str) -> frozenset[str]:
    """'mistral, gemini/ ,groq' -> {'mistral', 'gemini', 'groq'}."""
    return frozenset(p.strip().rstrip("/") for p in raw.split(",") if p.strip().rstrip("/"))


def get_settings() -> Settings:
    return Settings(
        omniroute_base_url=os.environ.get(
            "OMNISWARM_OMNIROUTE_URL", "http://localhost:20128/v1"
        ),
        db_path=os.environ.get("OMNISWARM_DB_PATH", "omniswarm.db"),
        connect_timeout=float(os.environ.get("OMNISWARM_CONNECT_TIMEOUT", "3")),
        read_timeout=float(os.environ.get("OMNISWARM_READ_TIMEOUT", "60")),
        max_connections=int(os.environ.get("OMNISWARM_MAX_CONN", "20")),
        # verified-answer cache: only pass + high-confidence answers are cached,
        # and only re-served for repeats within the TTL. 0 TTL = never expire.
        cache_enabled=os.environ.get("OMNISWARM_CACHE", "1") not in ("0", "false", "False"),
        cache_ttl_seconds=float(os.environ.get("OMNISWARM_CACHE_TTL", str(7 * 24 * 3600))),
        cache_min_confidence=os.environ.get("OMNISWARM_CACHE_MIN_CONFIDENCE", "high"),
        # hard cap so the cache cannot grow without bound; 0 disables the cap
        cache_max_entries=int(os.environ.get("OMNISWARM_CACHE_MAX_ENTRIES", "5000")),
        # blended $/1M tokens of the premium model you'd otherwise pay for;
        # ~GPT-4o / Claude Sonnet input+output. Drives the dashboard "$ saved" stat.
        savings_usd_per_mtok=float(os.environ.get("OMNISWARM_SAVINGS_USD_PER_MTOK", "5")),
        # only offer/accept models from these gateway providers (comma-separated
        # prefixes, e.g. "mistral,gemini,groq"). Empty = every provider the gateway lists.
        # Keeps the catalog, recommender and auto-failover off providers that can't
        # run in your deployment; a new gateway provider then needs an explicit opt-in.
        allowed_providers=_providers(os.environ.get("OMNISWARM_ALLOWED_PROVIDERS", "")),
    )
