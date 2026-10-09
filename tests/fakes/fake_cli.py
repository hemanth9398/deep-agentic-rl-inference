#!/usr/bin/env python3
"""Fake `docker` / `kubectl` for tests: containers and pods are directories under $FAKE_CLI_STATE.

Container path P lives at <state>/<name>/fs/P. Images are directories (a built tag maps to its
build context's repo/). Every invocation is logged to <state>/calls.jsonl for assertions.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

STATE = Path(os.environ["FAKE_CLI_STATE"])
STATE.mkdir(parents=True, exist_ok=True)
IMAGES = STATE / "images.json"


def log_call(tool, argv):
    with open(STATE / "calls.jsonl", "a") as f:
        f.write(json.dumps({"tool": tool, "argv": argv}) + "\n")


def images():
    return json.loads(IMAGES.read_text()) if IMAGES.exists() else {}


def fs(name, path="/"):
    return STATE / name / "fs" / path.lstrip("/")


def start(name, image, workdir):
    src = images().get(image, image)
    if not Path(src).is_dir():
        print(f"Unable to find image '{image}'", file=sys.stderr)
        sys.exit(125)
    shutil.copytree(src, fs(name, workdir), ignore=shutil.ignore_patterns(".git", "__pycache__"))
    (STATE / name / "workdir").write_text(workdir)


def run_in(name, workdir, script, stdin):
    if not (STATE / name).exists():
        print(f"Error: No such container: {name}", file=sys.stderr)
        sys.exit(125)
    p = subprocess.run(["bash", "-c", script], cwd=fs(name, workdir), input=stdin, capture_output=True)
    sys.stdout.buffer.write(p.stdout)
    sys.stderr.buffer.write(p.stderr)
    sys.exit(p.returncode)


def copy(src, target):
    name, dest = target.split(":", 1)
    name = name.split("/")[-1]  # kubectl: ns/pod
    shutil.copytree(src.rstrip(".").rstrip("/"), fs(name, dest), dirs_exist_ok=True)


def docker(a):
    cmd = a[0]
    if cmd == "run":
        name = a[a.index("--name") + 1]
        wd = a[a.index("-w") + 1]
        image = a[a.index("sleep") - 1]
        start(name, image, wd)
        print(name)
    elif cmd == "exec":
        i = 1
        stdin = None
        if a[i] == "-i":
            stdin, i = sys.stdin.buffer.read(), i + 1
        wd = a[i + 1]
        name = a[i + 2]
        run_in(name, wd, a[-1], stdin)
    elif cmd == "cp":
        copy(a[1], a[2])
    elif cmd == "rm":
        shutil.rmtree(STATE / a[-1], ignore_errors=True)
    elif cmd == "image":
        sys.exit(0 if a[2] in images() else 1)
    elif cmd == "build":
        tag, ctx = a[a.index("-t") + 1], a[-1]
        imgs = images()
        imgs[tag] = str(Path(ctx) / "repo")
        IMAGES.write_text(json.dumps(imgs))
        print("sha256:fake")
    else:
        sys.exit(f"fake docker: unsupported {cmd}")


def kubectl(a):
    cmd = a[0]
    if cmd == "apply":
        obj = json.loads(sys.stdin.read())
        (STATE / f"{obj['kind']}-{obj['metadata']['name']}.json").write_text(json.dumps(obj))
        if obj["kind"] == "Pod":
            c = obj["spec"]["containers"][0]
            start(obj["metadata"]["name"], c["image"], c["workingDir"])
    elif cmd == "wait":
        sys.exit(0)
    elif cmd == "exec":
        stdin = sys.stdin.buffer.read() if "-i" in a[: a.index("--")] else None
        name = a[a.index("--") - 1]
        inner = a[-1]
        head, rest = inner.split(" && ", 1)  # "cd <workdir> && ..."
        run_in(name, head[3:].strip("'"), rest, stdin)
    elif cmd == "cp":
        copy(a[1], a[2])
    elif cmd == "delete":
        shutil.rmtree(STATE / a[a.index("-n") + 2], ignore_errors=True)
    else:
        sys.exit(f"fake kubectl: unsupported {cmd}")


if __name__ == "__main__":
    tool = os.environ["FAKE_TOOL"]
    log_call(tool, sys.argv[1:])
    (docker if tool == "docker" else kubectl)(sys.argv[1:])
