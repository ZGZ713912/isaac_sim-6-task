#!/usr/bin/env python3
"""Fresh managed PPO with GPU-releasing behavior reviews and final ONNX export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from wheeled_algo.managed_training import WheelbipeSession
from wheeled_algo.trainctl_shared.training_control import file_pin,read_json,write_json,verify_pin
from wheeled_tasks.wheelbipe_port.training_profile import specification,behavior_gate


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def data_json(path):
    """Training/ONNX receipts contain per-lane masses; control files stay small."""
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(descriptor,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>8*1024**2:
            raise ValueError('Training data receipt exceeds the bounded 8MiB reader')
        return json.load(stream)


def validate_admission(receipt,parity,num_envs):
    settings=receipt["arguments"]
    if (receipt["status"]!="training_complete" or receipt["completed_updates"]<30
            or not receipt["policy_tensors_finite"] or receipt["fresh_initial_optimizer_state_count"]!=0
            or receipt["final_optimizer_state_count"]!=17 or settings["training_profile"]!="foundation-v1"
            or settings["num_envs"]!=num_envs or settings["physics_hz"]!=200
            or settings["leg_armature"]!=.015795 or settings["leg_kp"]!=60. or settings["leg_kd"]!=2.
            or settings["randomization_profile"]!="requested" or settings["randomization_strength"]!=1.
            or settings["solver_position_iterations"]!=64 or settings["solver_velocity_iterations"]!=32
            or settings["external_forces_every_iteration"] or receipt["constraint_gap_hard_limit_enabled"]
            or receipt["native"]["gas_cancellation_feedforward"]):
        raise ValueError("Engineering admission does not match this new foundation campaign")
    for name,expected in receipt["implementation_files_sha256"].items():
        if file_pin(ROOT/name)["sha256"]!=expected:
            raise ValueError("Campaign learner source changed after native admission: "+name)
    for name,expected in receipt["deployment"]["files_sha256"].items():
        if name.startswith(("src/","configs/","contracts/","model/","third_party/")):
            if file_pin(ROOT/name)["sha256"]!=expected:
                raise ValueError("Campaign simulation/config data changed after native admission: "+name)
    if (parity["checkpoint_sha256"]!=receipt["checkpoint_sha256"]
            or parity["max_absolute_error"]>1.e-5 or parity["gas_cancellation_feedforward"]):
        raise ValueError("Pilot export/parity/control contract failed")


def resource_available():
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:')}
    gpu=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True)
    return memory['MemAvailable']>3*1024**3 and int(gpu.stdout.splitlines()[0])>3072


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--num-envs',type=int,choices=(512,1024),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--python',type=Path,required=True)
    args=p.parse_args()
    args.session=args.session.resolve()
    args.session.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.session.parent/(args.session.name+'.campaign.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.session.exists():
        p.error('Refusing to adopt or restart an existing campaign')
    admitted=data_json(args.admission)
    parity_path=Path(admitted['latest_checkpoint']).parent/'onnx/contract.json'
    validate_admission(admitted,data_json(parity_path),args.num_envs)
    if file_pin(admitted['latest_checkpoint'])['sha256']!=admitted['checkpoint_sha256']:
        raise ValueError('Admission checkpoint changed')
    parity=data_json(parity_path)
    if file_pin(parity_path.parent/'policy.onnx')['sha256']!=parity['onnx_sha256']:
        raise ValueError('Admission ONNX changed')
    common=['--mode','train','--task','flat','--num-envs',str(args.num_envs),'--iterations','20000',
        '--save-interval','500','--seed',str(args.seed),'--device','cuda:0','--headless',
        '--training-profile','foundation-v1','--physics-hz','200','--leg-armature','.015795',
        '--leg-kp','60','--leg-kd','2','--leg-target-projection-penalty','1',
        '--solver-position-iterations','64','--solver-velocity-iterations','32',
        '--no-external-forces-every-iteration','--randomization-profile','requested','--randomization-strength','1']
    session=WheelbipeSession.create(args.session,ROOT,args.python,common,
        name=args.session.name,environment={k:os.environ[k] for k in
            ('OMNI_KIT_ACCEPT_EULA','PYTHONNOUSERSITE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH') if k in os.environ})
    state=dict(schema='wheelbipe-repair-campaign-v1',status='starting',source=str(ROOT),
        session=str(session.root),fresh_actor_critic_empty_adam=True,total_updates=20000,
        population=args.num_envs,seed=args.seed,admission=file_pin(args.admission),
        behavior_qualified=False,hardware_io=False,started_at=timestamp(),reviews=[])
    path=session.root/'campaign.json'
    write_json(path,state)
    process=None
    def interrupt(signum,frame):
        raise KeyboardInterrupt('Campaign stop requested')
    signal.signal(signal.SIGTERM,interrupt)
    def run_inference(output,checkpoint,final=False):
        nonlocal process
        if not resource_available():
            raise RuntimeError('Insufficient host/GPU memory after learner exit; left stopped')
        command=[str(args.python),'-u',str(ROOT/'scripts/audit_wheelbipe_behavior.py'),
            '--source-root',str(ROOT),'--vehicle','port','--training-profile','foundation-v1',
            '--checkpoint',str(checkpoint),'--output',str(output),'--num-envs','32','--steps','400','--seed','111']
        if final:
            command+=['--cases','stand_low','stand_mid','stand_high','forward','reverse','spin_positive','spin_negative']
        with output.with_suffix('.log').open('x') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
            state['inference_pid']=process.pid
            write_json(path,state,replace=True)
            code=process.wait()
        receipt=data_json(output/'receipt.json')
        if code or not receipt or receipt.get('status')!='complete':
            raise RuntimeError(f'Inference failed with exit {code}; see {output}.log')
        process=None
        if receipt['checkpoint_sha256']!=file_pin(checkpoint)['sha256']:
            raise ValueError('Inference loaded a different checkpoint')
        return receipt
    try:
        session.start()
        while True:
            current=session.status()
            state.update(status='training' if current['active'] else current['phase'],
                completed_updates=current['update'],learner=current,updated_at=timestamp())
            write_json(path,state,replace=True)
            if current['active']:
                time.sleep(5.)
                continue
            if current['phase']=='completed':
                break
            update=current['update']
            if (current['phase']!='stopped' or current.get('stop_reason')!=f'behavior_review:{update}'
                    or current.get('requested')!='resume'):
                state['status']='manual_stop' if current['phase']=='stopped' else 'failed'
                write_json(path,state,replace=True)
                return
            attempt=session.attempt()
            sequence=current['request_sequence']
            checkpoint=Path(current['checkpoint'])
            session.verify_sources()
            for pin in read_json(attempt/'control/result.json')['checkpoint_pins']:
                verify_pin(pin)
            output=session.root/f'review_{update:05d}'
            state['status']='behavior_review'
            write_json(path,state,replace=True)
            receipt=run_inference(output,checkpoint)
            gate=behavior_gate(receipt)
            required=update in specification()['required_behavior_reviews']
            state['reviews'].append(dict(update=update,required=required,gate=gate,
                checkpoint=file_pin(checkpoint),receipt=file_pin(output/'receipt.json')))
            write_json(output/'gate.json',gate)
            if required and not gate['passed']:
                state['status']='stopped_behavior_review_failed'
                write_json(path,state,replace=True)
                return
            # The compare-and-start is inside the same ownership lock as manual
            # controls, so a stop/pause during inference cannot be undone.
            try:
                session.start(expected_attempt=attempt,expected_request_sequence=sequence)
            except ValueError as error:
                if 'Manual control changed' not in str(error):
                    raise
                state.update(status='manual_stop',reason=str(error))
                write_json(path,state,replace=True)
                return
        if current['update']!=20000:
            raise ValueError('The fixed original total budget was not completed')
        last=session.attempt()/'train'
        trained=data_json(last/'receipt.json')
        checkpoint=Path(trained['latest_checkpoint'])
        if trained['status']!='training_complete' or trained['completed_updates']!=20000:
            raise ValueError('Final learner receipt is incomplete')
        exported=session.root/'onnx'
        state['status']='exporting'
        write_json(path,state,replace=True)
        with (session.root/'export.log').open('x') as stream:
            subprocess.run([str(args.python),str(ROOT/'scripts/export_wheelbipe_port_onnx.py'),
                '--checkpoint',str(checkpoint),'--output',str(exported)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        shutil.copytree(last/'params',exported/'params')
        final=run_inference(session.root/'final_evaluation',checkpoint,final=True)
        requirements=read_json(ROOT/'contracts/wheel_leg_acceptance_requirements_v2.json')
        write_json(session.root/'capability_status.json',dict(checkpoint=file_pin(checkpoint),
            requirements=file_pin(ROOT/'contracts/wheel_leg_acceptance_requirements_v2.json'),
            capabilities=[dict(id=row['id'],requirement=row['requirement'],status='not_qualified',
                evidence='Only finite cold-command diagnostics; full ability scope not evaluated') for row in requirements['capabilities']]))
        state.update(status='complete',finished_at=timestamp(),checkpoint=file_pin(checkpoint),
            onnx=file_pin(exported/'policy.onnx'),contract=file_pin(exported/'contract.json'),
            final_diagnostic=file_pin(session.root/'final_evaluation/receipt.json'))
        write_json(path,state,replace=True)
    except BaseException as error:
        if session.status()['active']:
            session.command('stop')
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        state.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        write_json(path,state,replace=True)
        raise


if __name__=='__main__':
    main()
