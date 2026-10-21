import pathlib, sys, types
root=pathlib.Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
snapshot=pathlib.Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/flat_gui1346_validation_20261021/source001')
sys.path[:0]=[str(root/'src'),str(root/'scripts')]
for name in ['play_v5_grounded', 'play_chassis_suite', 'play_flat_base', 'play_flat_speed_preview']:
 module=types.ModuleType(name); module.__file__=str(root/'scripts'/(name+'.py')); sys.modules[name]=module
 exec(compile((snapshot/(name+'.py')).read_text(),module.__file__,'exec'),module.__dict__)
code=(snapshot/'record_flat_gui_validation.py').read_text()
exec(compile(code,str(root/'scripts/record_flat_gui_validation.py'),'exec'),{'__name__':'__main__','__file__':str(root/'scripts/record_flat_gui_validation.py')})
