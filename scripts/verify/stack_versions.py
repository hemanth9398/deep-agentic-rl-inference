"""Verify the pinned stack: SkyRL inference backend, SkyRL <-> Verifiers, vf-eval CLI.

Usage: python scripts/verify/stack_versions.py

Prints what the installed SkyRL accepts as an inference backend (at SkyRL main b896332 it is
"vllm" only, which conflicts with an SGLang-based design; see swarmrl/integrations/skyrl.py),
and whether Verifiers imports and its CLI runs.
"""

import importlib
import shutil
import subprocess
import sys


def version(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed")
    except Exception as e:  # noqa: BLE001
        return f"missing ({type(e).__name__})"


def main() -> int:
    for mod in ("skyrl", "verifiers", "sglang", "transformers", "peft", "trl", "torch"):
        print(f"{mod:14s} {version(mod)}")
    ok = True
    try:
        from skyrl.train.config.config import InferenceEngineConfig  # noqa: F401
        import dataclasses
        f = {x.name: x for x in dataclasses.fields(InferenceEngineConfig)}
        print("skyrl inference backend default:", f["backend"].default)
        if f["backend"].default != "sglang":
            print("WARNING: SkyRL will not drive SGLang engines; use option (a) or (b) in integrations/skyrl.py")
    except Exception as e:  # noqa: BLE001
        print("skyrl config check skipped:", type(e).__name__, e)
    if shutil.which("vf-eval"):
        r = subprocess.run(["vf-eval", "--help"], capture_output=True, text=True)
        print("vf-eval --help:", "ok" if r.returncode == 0 else f"exit {r.returncode}")
        ok &= r.returncode == 0
    else:
        print("vf-eval not on PATH")
    print("Competition rules on teacher models / external environments: check manually (not automatable).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
