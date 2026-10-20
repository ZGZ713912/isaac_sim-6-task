from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tests'))
sys.path.insert(0,str(HERE/'staged_root/src'))
raise SystemExit(pytest.main(['-q',str(HERE/'integration_tests/test_motion_grip_adaptation.py'),'--junitxml='+str(HERE/'cpu_staged002.xml')]))
