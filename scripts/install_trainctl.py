#!/usr/bin/env python3
"""Install the stdlib-only trainctl outside any temporary checkout."""
from pathlib import Path
import argparse
import hashlib
import shlex
import shutil
import sys


def install(prefix, *, share_existing=False):
    prefix = Path(prefix).expanduser().resolve()
    source = Path(__file__).resolve().parents[1]/"src/wheeled_algo/training_control.py"
    destination = prefix/"share/wheel-leg-trainctl/trainctl.py"
    executable = prefix/"bin/trainctl"
    direct = prefix/"bin/wheel-leg-trainctl"
    managed_direct = prefix/"bin/wheelbipe-trainctl"
    marker = "# wheel-leg-trainctl launcher"
    preserved = None
    original = executable.read_text() if executable.exists() else ""
    if original and marker not in original:
        # Kaiser already has a separate packed-policy control console. Keep
        # every existing command and source intact, adding one explicit route.
        if not share_existing or "-m transformer_rl.common.training_cli" not in original:
            raise ValueError(f"Refusing to replace an unrelated trainctl executable: {executable}")
        original_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
        preserved = destination.parent/f"original-trainctl-{original_hash}.sh"
    elif original:
        preserved = next((Path(line.removeprefix("# preserved_trainctl=")) for line in original.splitlines()
                          if line.startswith("# preserved_trainctl=")), None)
        if preserved is not None and not preserved.is_file():
            raise ValueError(f"Preserved original trainctl is missing: {preserved}")
    if direct.exists() and marker not in direct.read_text():
        raise ValueError(f"Refusing to replace an unrelated executable: {direct}")
    if managed_direct.exists() and marker not in managed_direct.read_text():
        raise ValueError(f"Refusing to replace an unrelated executable: {managed_direct}")
    package = source.parent
    managed_files = [package/"managed_training.py",package/"durable_checkpoint.py",
        *sorted((package/"trainctl_shared").glob("*.py")),package/"trainctl_shared/provenance.json"]
    package_hash = hashlib.sha256("".join(str(p.relative_to(package))+hashlib.sha256(p.read_bytes()).hexdigest()
        for p in managed_files).encode()).hexdigest()
    managed_root = destination.parent/("managed-"+package_hash[:16])
    for path in managed_files:
        target = managed_root/"src/wheeled_algo"/path.relative_to(package)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=path.read_bytes():
            raise ValueError(f"Versioned managed controller differs: {target}")
        if not target.exists():
            shutil.copyfile(path,target)
    (managed_root/"src/wheeled_algo/__init__.py").touch(exist_ok=True)
    managed_invocation = f"exec {shlex.quote(sys.executable)} {shlex.quote(str(managed_root/'src/wheeled_algo/managed_training.py'))} \"$@\""
    destination.parent.mkdir(parents=True, exist_ok=True)
    executable.parent.mkdir(parents=True, exist_ok=True)
    if original and marker not in original:
        if preserved.exists() and preserved.read_text() != original:
            raise ValueError("Preserved trainctl hash collision")
        preserved.write_text(original)
        preserved.chmod(0o755)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)
    invocation = f"exec {shlex.quote(sys.executable)} {shlex.quote(str(destination))} \"$@\""
    launcher = f"#!/bin/sh\n{marker}\n{invocation}\n"
    for target in (direct, executable, managed_direct):
        content = launcher
        if target == managed_direct:
            content = f"#!/bin/sh\n{marker}\n{managed_invocation}\n"
        elif target == executable:
            preserved_line = f"# preserved_trainctl={preserved}\n" if preserved else ""
            fallback = f"exec {shlex.quote(str(preserved))} \"$@\"" if preserved else invocation
            content = (f"#!/bin/sh\n{marker}\n{preserved_line}"
                       f"case \"${{1:-}}\" in\n  wheel-leg) shift; {invocation} ;;\n"
                       f"  wheelbipe) shift; {managed_invocation} ;;\n"
                       f"  *) {fallback} ;;\nesac\n")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(content)
        temporary.chmod(0o755)
        temporary.replace(target)
    print(executable)
    print(f"sha256={hashlib.sha256(destination.read_bytes()).hexdigest()}")
    print(f"Disk-backed shared console: trainctl wheelbipe; managed controller sha256={package_hash}")
    if preserved is not None:
        print(f"Existing console preserved: {preserved}; use trainctl wheel-leg for wheel-leg training")
    return executable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path.home()/".local")
    parser.add_argument("--share-existing", action="store_true",
                        help="Preserve the known transformer-rl console and add trainctl wheel-leg")
    try:
        args = parser.parse_args()
        install(args.prefix, share_existing=args.share_existing)
    except (OSError, ValueError) as error:
        sys.exit(str(error))
