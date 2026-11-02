#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$task_root/scripts/train_prone_motion.sh" --task WheelLeg-JumpMotion-v1 "$@"
