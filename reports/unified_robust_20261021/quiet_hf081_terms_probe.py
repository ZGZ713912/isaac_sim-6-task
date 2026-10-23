"""Offline decomposition of existing quiet wheel smoothing and neural correction."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUN = ROOT / "runs/unified_robust_20261022/fresh_nominal067_211131/robust"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from analyze_robust_getup import hf_rms
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.robust_getup import quiet_gate, tilt_gate


def main():
    torch.set_num_threads(1)
    mean, _ = load_mean(ROOT / "runs/unified_robust_20261022/quiet_severe_reference065/model_migrated.pt")
    report = json.loads((RUN / "report.json").read_text())
    path = RUN / "trajectory.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["trajectory_sha256"]
    data = np.load(path)
    cases = {}
    for case in ("stand_low", "stand", "stand_high"):
        lane = next(row["lane"] for row in report["lanes"] if row["name"] == case)
        selected = data["alive"][:, lane] & (data["time"][:, lane] >= 3.)
        obs = torch.from_numpy(data["observations"][selected, lane].copy())
        captured = {}
        original = mean.quiet_correction
        def correction(*args, **kwargs):
            value = original(*args, **kwargs)
            captured["correction"] = value
            return value
        def parent_hook(_module, _args, output):
            captured["motion"] = output
        handle = mean.parent.register_forward_hook(parent_hook)
        try:
            with torch.no_grad(), patch.object(mean, "quiet_correction", side_effect=correction):
                action = mean(obs)
        finally:
            handle.remove()
        assert bool((quiet_gate(obs) == 1.).all())
        assert bool((tilt_gate(obs, True, True) == 0.).all())
        motion = torch.maximum(torch.minimum(captured["motion"], mean.motion_bound), -mean.motion_bound)
        smooth = obs[:, 22:28] + mean.quiet_alpha * (motion - obs[:, 22:28])
        corr = mean.quiet_bound * captured["correction"]
        assert float((smooth + corr - action).abs().max()) < 1e-6
        s = smooth[:, 4:].numpy() * 10.
        c = corr[:, 4:].numpy() * 10.
        a = action[:, 4:].numpy() * 10.
        sr, cr, ar = [hf_rms(value, .005) for value in (s, c, a)]
        cross = (ar*ar - sr*sr - cr*cr) / (2.*sr*cr)
        gammas = {str(gamma): hf_rms(s + gamma * c + (1. - gamma) * c.mean(0, keepdims=True), .005)
                  for gamma in (0., .25, .5, .75, 1., 1.25, 1.5)}
        cases[case] = dict(whole_wheel20_rms_rad_s=ar, smoothing_term20_rms_rad_s=sr,
            neural_correction20_rms_rad_s=cr, hf_term_correlation=cross,
            ideal_AC_neural_gain_same_input_wheel20_rms_rad_s=gammas,
            reconstruction_raw_max_error=float((smooth + corr - action).abs().max()))
    result = dict(schema="quiet081-offline-term-decomposition-v1", cases=cases,
        trajectory_sha256=report["trajectory_sha256"], new_macro_updates=0,
        actor_changed=False, native_runs=0,
        scope="Same recorded inputs only; retaining segment mean uses future information and is not a deployment method or closed-loop benefit")
    write(REPORTS / "quiet_hf081_terms.json", result)
    with SummaryWriter(str(RUN.parent), filename_suffix=".quiet081_terms") as writer:
        for case, row in cases.items():
            for key, value in row.items():
                if isinstance(value, dict):
                    for gamma, metric in value.items():
                        writer.add_scalar("Diagnosis/QuietHF081/" + case + "/AC_gain_" + gamma + "_wheel20_rms_rad_s", metric, 505)
                else:
                    writer.add_scalar("Diagnosis/QuietHF081/" + case + "/" + key, value, 505)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
