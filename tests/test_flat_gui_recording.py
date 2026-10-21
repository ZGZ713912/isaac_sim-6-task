"""The video clock and keyboard scenarios must retain failures and exact speed."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from record_flat_gui_validation import GuiRecorder, copy_capture_image, phase_index, validation_cases
from play_flat_speed_preview import package_policy


def test_asymmetric_keys_and_exact_stop_phase_boundaries():
    cases = {c['name']: c for c in validation_cases()}
    forward = cases['start_stop_3_to_2p5']
    reverse = cases['backward_start_stop_2p5_to_3']
    assert [p['key'] for p in forward['phases']] == ['W', None, 'S', None]
    assert [p['key'] for p in reverse['phases']] == ['S', None, 'W', None]
    assert forward['phases'][2]['vx_limit'] == reverse['phases'][0]['vx_limit'] == 2.5
    assert phase_index(forward, 799, .005) == 0
    assert phase_index(forward, 800, .005) == 1
    assert phase_index(forward, 1600, .005) == 2
    assert phase_index(forward, 2400, .005) == 3
    assert len(cases) == 15


@pytest.mark.parametrize('ticks,terminal,frames', [(400, False, 50), (77, True, 10)])
def test_frame_clock_keeps_1x_and_uncensored_terminal(ticks, terminal, frames):
    case = validation_cases(2.)[0]
    recorder = GuiRecorder([case], terminal_hold_s=0.)
    recorder.case = case
    recorder.policy_dt, recorder.stride = .005, 8
    recorder.current = dict(peak_abs_pitch_deg=0., peak_abs_roll_deg=0., closure_gap_peak_mm=0.)
    recorder.trace = SimpleNamespace(write=lambda value: None, flush=lambda: None)
    recorder.frame_log = SimpleNamespace(flush=lambda: None)
    captured = []
    recorder.capture = lambda *a, **kw: captured.append(recorder.case_ticks)
    recorder.finish_encoder = lambda: None
    recorder.save = lambda: None
    sample = dict(velocity_m_s=[.4, 0, 0], omega_rad_s=[0, 0, 0], height_m=.305,
        real_time_factor=.05, pitch_deg=1., roll_deg=0., gap_mm=.1, status='PAUSED: non_wheel' if terminal else 'running')
    for tick in range(1, ticks+1):
        recorder.sample(sample, {}, terminal and tick == ticks)
    assert recorder.finished
    assert len(captured) == frames
    assert captured[-1] == ticks
    assert recorder.current['simulated_seconds'] == ticks * .005
    assert recorder.current['terminal'] is terminal
    assert frames / 25 - ticks * .005 == pytest.approx(0., abs=1/25)


def test_unbounded_or_nonfinite_durations_are_rejected():
    for seconds in (0., float('nan'), float('inf'), 31.):
        with pytest.raises(ValueError):
            validation_cases(seconds)


def test_descriptive_model_name_keeps_exact_authenticated_bytes(tmp_path):
    import hashlib
    model=tmp_path/(tmp_path.name+'.onnx')
    model.write_bytes(b'authenticated model')
    proof=dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    assert package_policy(tmp_path,proof)==model
    (tmp_path/'policy.onnx').write_bytes(b'another model')
    with pytest.raises(ValueError):
        package_policy(tmp_path,proof)


def test_renderer_buffer_is_copied_before_native_memory_is_reused():
    source=bytearray([10,20,30,255,40,50,60,255])
    image=copy_capture_image(source,len(source),2,1)
    source[:]=b'\0'*8
    assert image.getpixel((0,0))==(10,20,30)
    assert image.getpixel((1,0))==(40,50,60)
    with pytest.raises(ValueError):
        copy_capture_image(b'bad',3,2,1)


def test_stop_requires_yaw_and_translation_and_censored_is_explicit():
    from analyze_flat_gui_recording import response_metrics
    rows=[]
    for tick in range(1,201):
        rows.append(dict(case_tick=tick,pressed_keys=['W'] if tick<=100 else [],
            requested_command=[1.,0.,.305] if tick<=100 else [0.,0.,.305],
            velocity_m_s=[1. if tick<=100 else 0.,0.,0.],
            omega_rad_s=[0.,0.,1. if tick<=140 else 0.],position_m=[tick*.005,0.,0.]))
    phases=response_metrics(rows)
    assert phases[0]['response90_s']==pytest.approx(.005)
    assert phases[1]['settle_time_s']==pytest.approx(.205)
    assert phases[1]['stop_path_m']==pytest.approx(.205)
    assert not phases[1]['censored']
    assert response_metrics(rows[:160])[1]['censored']
