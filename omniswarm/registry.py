from dataclasses import dataclass, field

from omniswarm.modelconfig import load_model_config

_R = "Numbered evaluation steps:\n1. Is the answer directly responsive to the task?\n2. Is it factually consistent and free of obvious errors?\n3. Is it complete for what was asked?\n4. Ignore length; do not reward verbosity."


@dataclass(frozen=True)
class TaskType:
    name: str
    model: str
    rubric: str
    validators: tuple[str, ...] = ("non_empty",)
    budget: int = 8
    high_stakes: bool = False
    params: dict = field(default_factory=dict)


# Per-task NON-model metadata (model comes from config). Task types not listed
# here use TaskType defaults.
_META: dict[str, dict] = {
    "reasoning": {"high_stakes": True},
}


def build_registry(cfg: dict) -> dict[str, "TaskType"]:
    reg: dict[str, TaskType] = {}
    for name, model in cfg["models"].items():
        meta = _META.get(name, {})
        reg[name] = TaskType(name=name, model=model, rubric=_R, **meta)
    if "general" not in reg:  # safety net: there must always be a fallback type
        reg["general"] = TaskType("general", "mistral/mistral-large-latest", _R)
    return reg


def _ok(mid: object) -> bool:
    """Return True only for non-empty, non-auto/, non-tllm/ model IDs."""
    return isinstance(mid, str) and bool(mid) and not mid.startswith(("auto/", "tllm/"))


def effective_config(overrides: dict) -> dict:
    cfg = load_model_config()
    ov_models = overrides.get("models") or {}
    for k, v in ov_models.items():
        if _ok(v):
            cfg["models"][k] = v
    if _ok(overrides.get("judge")):
        cfg["judge"] = overrides["judge"]
    if _ok(overrides.get("synth")):
        cfg["synth"] = overrides["synth"]
    ov_members = {m["role"]: m["model"] for m in (overrides.get("members") or []) if _ok(m.get("model"))}
    if ov_members:
        cfg["members"] = [
            {**mem, "model": ov_members.get(mem["role"], mem["model"])} for mem in cfg["members"]
        ]
    return cfg


def _install(cfg: dict) -> None:
    global REGISTRY, JUDGE_MODEL, COUNCIL_SYNTH_MODEL, COUNCIL_GENERATORS
    global COUNCIL_MEMBERS, _DEFAULT_ROSTER, _ROSTER_OVERRIDES, _BY_ROLE
    REGISTRY = build_registry(cfg)
    JUDGE_MODEL = cfg["judge"]
    COUNCIL_SYNTH_MODEL = cfg["synth"]
    COUNCIL_GENERATORS = cfg["council_generators"]
    COUNCIL_MEMBERS = cfg["members"]
    _DEFAULT_ROSTER = cfg["default_roster"]
    _ROSTER_OVERRIDES = cfg["roster_overrides"]
    _BY_ROLE = {m["role"]: m for m in COUNCIL_MEMBERS}


def apply_runtime(overrides: dict) -> None:
    _install(effective_config(overrides))


REGISTRY: dict[str, TaskType] = {}
JUDGE_MODEL: str = ""
COUNCIL_SYNTH_MODEL: str = ""
COUNCIL_GENERATORS: list[str] = []
COUNCIL_MEMBERS: list[dict] = []
_DEFAULT_ROSTER: list[str] = []
_ROSTER_OVERRIDES: dict[str, list[str]] = {}
_BY_ROLE: dict[str, dict] = {}

_install(load_model_config())  # defaults at import; app.py re-applies runtime overrides at startup


def council_roster(task_type: str) -> list[dict]:
    roles = _ROSTER_OVERRIDES.get(task_type, _DEFAULT_ROSTER)
    members = [_BY_ROLE[r] for r in roles if r in _BY_ROLE]
    return members or COUNCIL_MEMBERS


def resolve_roster(task_type: str, active_roster: list[str] | None = None) -> list[dict]:
    if active_roster:
        members = [_BY_ROLE[r] for r in active_roster if r in _BY_ROLE]
        if members:
            return members
    return council_roster(task_type)


def get_task_type(name: str | None) -> TaskType:
    if name is None:
        return REGISTRY["general"]
    return REGISTRY.get(name, REGISTRY["general"])
