"""Write reviewable trainer/decoder drafts; never mutate the active sources."""
from pathlib import Path
import ast,json,hashlib,time
R=Path('reports/robust_policy_repair_20261024')
source=Path('scripts/train_robust_getup.py');s=source.read_text();old_sha=hashlib.sha256(source.read_bytes()).hexdigest();edits=[]
def replace(old,new):
 global s
 assert s.count(old)==1,(old[:100],s.count(old))
 s=s.replace(old,new);edits.append(dict(before=old,after=new))
replace('"motion_leg_damping_version", "motion_leg_damping_alpha"):',
        '"motion_leg_damping_version", "motion_leg_damping_alpha",\n                    "quiet_stance_authority_version", "quiet_stance_retained_authority", "quiet_stance_recovery_threshold"):')
replace('        function_root = resume_info.get("schema") == "robust-getup-checkpoint-v3"',
        '        function_root = resume_info.get("schema") == "robust-getup-checkpoint-v3"\n        fallen_function_root = resume_info.get("schema") == "robust-getup-checkpoint-v5"\n        if fallen_function_root and (args.mode != "train" or not args.recovery_only or args.entry_recovery_only):\n            parser.error("V5 resumes require the existing fallen output-only training mode")')
replace('        if args.mode == "train" and (function_root or revised_function):',
        '        if args.mode == "train" and (function_root or revised_function) and not fallen_function_root:')
replace('        if function_root or revised_function:\n', '        if function_root or revised_function or fallen_function_root:\n')
replace('    if args.resume and function_root:\n', '    if args.resume and (function_root or fallen_function_root):\n')
replace('        from wheeled_algo.function_root_checkpoint import validate_training_settings\n        from export_robust_getup import load_mean as load_authenticated_mean\n        root_info = torch.load(resume_info["training_function_root"]["checkpoint"],',
'''        if fallen_function_root:
            from wheeled_algo.fallen_output_checkpoint import validate_training_settings
            root_record = resume_info["policy_function_root"]
        else:
            from wheeled_algo.function_root_checkpoint import validate_training_settings
            root_record = resume_info["training_function_root"]
        from export_robust_getup import load_mean as load_authenticated_mean
        root_info = torch.load(root_record["checkpoint"],''')
replace('        mean.action_bound[:4].fill_(task["leg_action_clip"])\n        env.actor_variant = args.actor_variant',
'''        mean.action_bound[:4].fill_(task["leg_action_clip"])
        if args.resume and fallen_function_root:
            # Rebuild the authenticated complete runtime, including its explicit
            # zero-strength stance extension. Optimizer layout stays legacy28.
            mean, _ = load_authenticated_mean(args.resume)
            mean = mean.to(args.device)
        env.actor_variant = args.actor_variant''')
replace('("robust-getup-checkpoint-v2", "robust-getup-checkpoint-v3") or state["infos"]["bank_sha256"]',
        '("robust-getup-checkpoint-v2", "robust-getup-checkpoint-v3", "robust-getup-checkpoint-v5") or state["infos"]["bank_sha256"]')
replace('            if args.entry_recovery_only:\n                from wheeled_algo.entry_optimizer import freeze_except_entry',
'''            if args.resume and fallen_function_root:
                runner.alg.actor.requires_grad_(False)
                mean.recovery[-1].requires_grad_(True)
            if args.entry_recovery_only:
                from wheeled_algo.entry_optimizer import freeze_except_entry''')
replace('                target=Path(path)\n',
'''                if args.resume and fallen_function_root:
                    from wheeled_algo.fallen_output_checkpoint import SCHEMA, make_progress, environment_revision
                    info.update(schema=SCHEMA,policy_function_root=deepcopy(state["infos"]["policy_function_root"]),
                        policy_function_stage="native-training",
                        policy_function_learning=make_progress(args.resume,state,runner.alg.actor.state_dict(),report["new_optimizer_updates"]))
                    revision=environment_revision(root_info["source_sha256"],hashes,task)
                    if revision is not None:
                        info["policy_function_environment_revision"]=revision
                target=Path(path)
''')
replace('                original_save(str(pending), infos=info)\n                pending.replace(target)',
'''                original_save(str(pending), infos=info)
                if args.resume and fallen_function_root:
                    from wheeled_algo.fallen_output_checkpoint import validate_state
                    from export_robust_getup import load_mean as load_checked_mean
                    root_path=state["infos"]["policy_function_root"]["checkpoint"]
                    root_state=torch.load(root_path,map_location="cpu",weights_only=False)
                    validate_state(torch.load(pending,map_location="cpu",weights_only=False),root_state,
                        torch.load(args.resume,map_location="cpu",weights_only=False),state["infos"]["policy_function_root"]["optimizer_actor_parameter_names"])
                    load_checked_mean(pending)
                pending.replace(target)''')
ast.parse(s);(R/'train_robust_getup_fallen_draft001.py').write_text(s)
export=Path('scripts/export_robust_getup.py');e=export.read_text();anchor='    if info.get("schema") == "robust-getup-checkpoint-v4":\n';assert e.count(anchor)==1
branch='''    if info.get("schema") == "robust-getup-checkpoint-v5":
        from wheeled_algo.fallen_output_checkpoint import MODULE, optimizer_actor_names, validate_state
        if info["source_sha256"].get(MODULE) != digest(ROOT / MODULE):
            raise ValueError("Fallen output contract source is missing or changed")
        for name, expected in info["source_sha256"].items():
            if name.startswith("src/wheeled_algo/") and digest(ROOT / name) != expected:
                raise ValueError("Fallen output policy dependency changed: " + name)
        origin = info.get("policy_function_root", {})
        root_path = Path(origin.get("checkpoint", ""))
        if not root_path.is_file() or digest(root_path) != origin.get("sha256"):
            raise ValueError("Fallen output policy function root is missing or changed")
        mean, _ = load_mean(root_path, _entry_ancestors=(*_entry_ancestors, resolved))
        names = optimizer_actor_names(mean)
        record = info.get("policy_function_learning", {})
        source = Path(record.get("source_checkpoint", ""))
        if not source.is_file() or digest(source) != record.get("source_sha256"):
            raise ValueError("Fallen output immediate source is missing or changed")
        if source.resolve() != root_path.resolve():
            load_mean(source, _entry_ancestors=(*_entry_ancestors, resolved))
        validate_state(state, torch.load(root_path, map_location="cpu", weights_only=False),
                       torch.load(source, map_location="cpu", weights_only=False), names)
        weights = {k.removeprefix("mlp.mean."): v for k, v in state["actor_state_dict"].items()
                   if k.startswith("mlp.mean.")}
        mean.load_state_dict(weights, strict=True)
        return mean.eval(), info
'''
e=e.replace(anchor,branch+anchor);ast.parse(e);(R/'export_robust_getup_fallen_draft001.py').write_text(e)
receipt=dict(schema='fallen-output-code-integration-draft-v1',status='not_applied',original_trainer_sha256=old_sha,
 original_export_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),trainer_draft_sha256=hashlib.sha256(s.encode()).hexdigest(),
 export_draft_sha256=hashlib.sha256(e.encode()).hexdigest(),trainer_edits=edits,new_PPO=0,qualification=None,created_unix=time.time())
(R/'fallen_output_code_draft001.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Drafts only:',len(edits),'trainer changes; active sources unchanged')
