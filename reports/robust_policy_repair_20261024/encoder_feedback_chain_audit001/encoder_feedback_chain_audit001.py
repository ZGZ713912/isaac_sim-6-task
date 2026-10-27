"""Read-only source audit of encoder correlation; not a hardware experiment."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
RMCS=Path('/home/noir/Documents/workspace/RMCS')
OUT=BASE/'reports/robust_policy_repair_20261024/encoder_feedback_chain_audit001'
OUT.mkdir(exist_ok=False)

def sha(data):
    return hashlib.sha256(data).hexdigest()

files={
    'observation.cpp':(RMCS, 'rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/observation.cpp'),
    'action.cpp':(RMCS, 'rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/action.cpp'),
    'unified_observation.hpp':(RMCS, 'rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/unified_observation.hpp'),
    'dm_motor.hpp':(RMCS,'rmcs_ws/src/rmcs_core/src/hardware/device/dm_motor.hpp'),
    'wheel-leg-infantry-unified-rl.yaml':(RMCS,'rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-unified-rl.yaml'),
    'robust_getup.py':(WORK,'src/wheeled_tasks/chassis/robust_getup.py'),
    'getup_feedback.py':(WORK,'src/wheeled_tasks/chassis/getup_feedback.py'),
    'shared_encoder_bias_probe.py':(WORK,'scripts/shared_encoder_bias_probe.py'),
}
needles={
    'observation.cpp':['leg_jacobian_ * motor_q + leg_offset_', 'position[i] = q_[i];', '*model_angle_outputs_[i] = q_[i];'],
    'action.cpp':['policy_profile_.leg_kp * (policy_targets_.head<4>() - q_.head<4>())',
                  'const double target = *angle_inputs_[i] + delta[i];'],
    'dm_motor.hpp':['angle_bias_ + sign * p_des', 'angle_ = sign * (raw_angle - angle_bias_);'],
    'wheel-leg-infantry-unified-rl.yaml':['joint_control_mode: "torque"'],
    'robust_getup.py':['self.bias[ids, 6:10] = self.uniform'],
    'getup_feedback.py':['self.bias[ids] = (torch.rand'],
}
sources={}
for label,(root,name) in files.items():
    file=root/name
    raw=file.read_bytes()
    snapshot=OUT/'source_snapshot'/root.name/name
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes(raw)
    lines=raw.decode().splitlines()
    evidence=[]
    for needle in needles.get(label,[]):
        hits=[(i+1,line) for i,line in enumerate(lines) if needle in line]
        assert len(hits)==1,(label,needle,hits)
        evidence.append(dict(line=hits[0][0],text=hits[0][1].strip()))
    sources[label]=dict(path=str(file),sha256=sha(raw),evidence=evidence)

bias=.004
result=dict(schema='wheel-leg-encoder-feedback-chain-source-audit-v1',
    observed_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
    RMCS_current_HEAD=subprocess.check_output(['git','-C',str(RMCS),'rev-parse','HEAD'],text=True).strip(),
    RMCS_active_branch=subprocess.check_output(['git','-C',str(RMCS),'branch','--show-current'],text=True).strip(),
    skill='/home/noir/.agents/skills/RMCS_Specification/SKILL.md',
    code_findings=[
        'Current unified RL YAML selects host torque. Model q is used by both the actor observation and the host leg PD position error.',
        'The optional internal motor position-PD path maps the model error back to the same encoder API; DmMotor applies the calibration bias to both command and reported angle.',
        'Simulation currently draws policy leg encoder zero and PD leg encoder zero independently, each within+/-0.004rad. This models a second static mismatch as well as a common zero uncertainty.',
    ],
    independent_zero_disagreement=dict(max_abs_rad=2*bias,rms_rad=math.sqrt(2/3)*bias,
        proportional_torque_max_nm_at_Kp160=160*2*bias,
        proportional_torque_rms_nm_at_Kp160=160*math.sqrt(2/3)*bias),
    prepared_test=dict(helper=sources['shared_encoder_bias_probe.py']['path'],
        helper_sha256=sources['shared_encoder_bias_probe.py']['sha256'],
        intervention='Only PD leg zero becomes the same draw as policy leg zero. Both native draws still consumed; policy bias and every other random draw stay unchanged.',
        common_encoder_zero_range_rad=[-bias,bias],native_test_completed=False,
        CPU_tests=2,CPU_tests_file=str(BASE/'reports/robust_policy_repair_20261024/shared_encoder_probe_CPU002.xml'),
        current_training_changed=False,current_serial_queue_changed=False,adopted=False),
    limitation='Source-level modeling mismatch hypothesis. No live RMCS process, CAN/internal encoder equality, closed-loop benefit or sole training root cause was verified. Quantization and sample-age differences remain separate physical uncertainties.',
    RMCS_files_modified=False,new_PPO=0,new_native_steps=0,hardware_motion=False,
    behavior_qualified=False,hardware_ready=False,producer_sha256=sha(Path(__file__).read_bytes()))
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
(OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(independent_zero_disagreement=result['independent_zero_disagreement'],
    RMCS_HEAD=result['RMCS_current_HEAD'],native_test_completed=False)))
