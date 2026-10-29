import collections,json,pathlib,runpy,sys
root=pathlib.Path('/root/gpufree-data/workspace/prone_motion_long')
audit=pathlib.Path('/root/gpufree-data/workspace/wheelleg_environment_20261029/native001')
sys.path.insert(0,str(root/'scripts'))
def capture(frame,event,arg):
 if event=='line' and frame.f_code.co_filename.endswith('/chassis/env.py') and frame.f_lineno==417:
  sys.settrace(None)
  data=frame.f_locals;env=data['self'];stage=env.sim.stage
  from isaaclab.sim.utils import get_current_stage
  current=get_current_stage()
  prims=list(stage.Traverse());counts=collections.Counter(p.GetTypeName() for p in prims)
  per_env=collections.Counter(str(p.GetPath()).split('/')[3] for p in prims if p.GetTypeName()=='PhysicsSphericalJoint' and str(p.GetPath()).startswith('/World/envs/'))
  result={'actual_loops':data['loop_count'],'expected_loops':6*data['num_envs'],'types':dict(counts),'per_env':dict(per_env),'sim_stage':str(stage.GetRootLayer().identifier),'current_stage':str(current.GetRootLayer().identifier),'same_stage':stage==current,'robot_roots':[str(p.GetPath()) for p in prims if str(p.GetPath()).endswith('/Robot')][:10]}
  (audit/'stage_diagnosis.json').write_text(json.dumps(result,indent=2));stage.GetRootLayer().Export(str(audit/'authored_stage.usda'))
  print('LOOP_DIAGNOSIS',json.dumps(result),flush=True)
 return capture
sys.settrace(capture)
sys.argv=[str(root/'scripts/train_prone_motion.py'),'--native-probe','--output',str(root/'runs/server_compatibility_20261029/probe001_diagnostic')]
runpy.run_path(sys.argv[0],run_name='__main__')
