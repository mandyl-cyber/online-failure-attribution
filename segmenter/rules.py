"""rules.yaml 的加载与预编译。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

DEFAULT_RULES = Path(__file__).with_name("rules.yaml")


def _compile_list(xs) -> list[re.Pattern]:
    return [re.compile(x) for x in (xs or [])]


class Rules:
    def __init__(self, path: str | Path | None = None, overrides: dict | None = None):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("需要 PyYAML：pip install pyyaml") from e
        self.path = Path(path) if path else DEFAULT_RULES
        self.raw: dict[str, Any] = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        for dotted, v in (overrides or {}).items():      # 例如 {"segment.mode": "phase"}
            d = self.raw
            *ks, last = dotted.split(".")
            for k in ks:
                d = d.setdefault(k, {})
            d[last] = v

        sh = self.raw["shell"]
        self.sh_wrappers = re.compile(sh["wrappers"])
        self.sh_irreversible = _compile_list(sh["irreversible"])
        self.sh_irreversible_exempt = re.compile(sh["irreversible_exempt"])
        self.sh_verify = {k: _compile_list(v) for k, v in sh["verify"].items()}
        self.sh_install = _compile_list(sh["install"])
        self.sh_state = _compile_list(sh["state_change"])
        self.sh_undo = _compile_list(sh["undo"])
        self.sh_redirect_exempt = re.compile(sh["redirect_exempt"])
        self.sh_env = _compile_list(sh["env_check"])
        self.sh_probe = _compile_list(sh.get("probe"))
        self.sh_noop = re.compile(sh.get("noop", "^$"))
        self.sh_search = _compile_list(sh["search"])
        self.sh_read = _compile_list(sh["read"])
        self.sh_ipython_write = re.compile(sh["ipython_write"])

        ev = self.raw["events"]
        self.edit_failed = _compile_list(ev["edit_failed"])
        self.verify_fail = _compile_list(ev["verify"]["fail"])
        self.verify_pass = _compile_list(ev["verify"]["pass"])
        self.plan_switch = _compile_list(ev["plan_switch"])
        self.submit_commands = _compile_list(ev["handoff"]["submit_commands"])
        self.message_requires_wait = bool(ev["handoff"].get("message_requires_wait", False))
        self.reference = ev["reference"]
        self.stop_symbols = set(self.reference.get("stop_symbols", []))
        self.stop_basenames = set(self.reference.get("stop_basenames", []))
        self.drift_enabled = bool(ev.get("drift", {}).get("enabled", False))

        self.tree = self.raw["tree"]
        self.segment = self.raw["segment"]
        self.trigger = self.raw["trigger"]

        cl = self.raw["claims"]
        self.claims = cl
        self.cl_plan = _compile_list(cl["plan_step"])
        self.cl_completion = _compile_list(cl["completion"])
        self.cl_completion_neg = re.compile(cl["completion_negation"])
        self.cl_env_obs = _compile_list(cl["env_assertion_obs"])
        self.cl_env_thought = _compile_list(cl["env_assertion_thought"])


_default: Rules | None = None


def default_rules() -> Rules:
    global _default
    if _default is None:
        _default = Rules()
    return _default
