"""Read native wheel collider settings without creating a robot or stepping."""
import json
import os
from pathlib import Path
import time

os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
from isaaclab.app import AppLauncher

launcher = AppLauncher(dict(headless=True, device='cuda:0', enable_cameras=False,
    kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
try:
    import carb.settings
    import isaaclab
    import omni.physx
    from omni.physx.bindings import _physx
    settings = carb.settings.get_settings()
    keys = {key: getattr(_physx, key) for key in dir(_physx)
            if key.startswith('SETTING_') and any(word in key for word in ('CYLINDER', 'CONE', 'CUSTOM_GEOMETRY'))}
    result = dict(schema='native-collision-geometry-settings-readback-v1',
        values={key: dict(setting_path=value, value=settings.get(value)) for key, value in keys.items()},
        physx_module=str(omni.physx.__file__), isaaclab_module=str(isaaclab.__file__),
        robot_spawned=False, physics_steps=0, new_PPO=0, qualification=None, created_unix=time.time())
    target = Path(__file__).with_name('collision_settings077.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
finally:
    launcher.app.close()
