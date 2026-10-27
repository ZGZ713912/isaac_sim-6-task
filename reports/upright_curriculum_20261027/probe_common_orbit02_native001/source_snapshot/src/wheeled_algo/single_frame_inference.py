"""Independent ONNX arms may run concurrently; each call remains batch size one."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class SingleFrameInference:
    def __init__(self, sessions, parallel=False):
        if not sessions:
            raise ValueError('Single-frame inference needs at least one session')
        self.sessions = sessions
        self.pool = ThreadPoolExecutor(max_workers=len(sessions)) if parallel else None

    @staticmethod
    def run_arm(session, frames):
        return np.concatenate([session.run(None, {'obs': frame[None]})[0] for frame in frames], axis=0)

    def run(self, frames_by_arm):
        if set(frames_by_arm) != set(self.sessions):
            raise ValueError('Each independent inference arm must have input frames')
        if self.pool is None:
            return {index: self.run_arm(self.sessions[index], frames) for index, frames in frames_by_arm.items()}
        jobs = {index: self.pool.submit(self.run_arm, self.sessions[index], frames)
                for index, frames in frames_by_arm.items()}
        # Join every arm before the caller can advance physics. No shared ONNX
        # session, simulator mutation, sampling or RNG runs in these workers.
        return {index: job.result() for index, job in jobs.items()}

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
