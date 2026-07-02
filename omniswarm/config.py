import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    omniroute_base_url: str
    db_path: str
    connect_timeout: float
    read_timeout: float
    max_connections: int


def get_settings() -> Settings:
    return Settings(
        omniroute_base_url=os.environ.get(
            "OMNISWARM_OMNIROUTE_URL", "http://localhost:20128/v1"
        ),
        db_path=os.environ.get("OMNISWARM_DB_PATH", "omniswarm.db"),
        connect_timeout=float(os.environ.get("OMNISWARM_CONNECT_TIMEOUT", "3")),
        read_timeout=float(os.environ.get("OMNISWARM_READ_TIMEOUT", "60")),
        max_connections=int(os.environ.get("OMNISWARM_MAX_CONN", "20")),
    )
