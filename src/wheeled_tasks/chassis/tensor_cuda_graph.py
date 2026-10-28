"""Replay deterministic tensor math with its original kernels and fresh outputs.

Only pure, fixed-shape calculations belong here: never native simulation,
random draws, time/history advances, resets or optimizer work. The caller keeps
its ordinary input validity checks outside capture.
"""
from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


class TensorCudaGraph:
    def __init__(self, function, *examples):
        if not examples or any(x.device.type != "cuda" or x.requires_grad for x in examples):
            raise ValueError("CUDA tensor replay requires fixed CUDA inference inputs")
        self.inputs = tuple(x.detach().clone() for x in examples)
        self.layouts = tuple((x.shape, x.dtype, x.device) for x in examples)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(examples[0].device), torch.inference_mode():
            stream = torch.cuda.Stream(device=examples[0].device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function(*self.inputs)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(self.graph, stream=stream):
                outputs = function(*self.inputs)
            self.outputs, self.output_spec = tree_flatten(outputs)
        if any(not isinstance(x, torch.Tensor) for x in self.outputs):
            raise ValueError("CUDA tensor replay must return tensors only")

    def __call__(self, *inputs):
        if len(inputs) != len(self.inputs) or any(
                (x.shape, x.dtype, x.device) != layout or x.requires_grad
                for x, layout in zip(inputs, self.layouts)):
            raise ValueError("CUDA tensor replay input shape/device/dtype changed")
        with torch.inference_mode():
            for target, value in zip(self.inputs, inputs):
                target.copy_(value)
            self.graph.replay()
            # Replays write into persistent storage. Do not expose that storage
            # to callers, diagnostics or a subsequent physical interval.
            return tree_unflatten([x.clone() for x in self.outputs], self.output_spec)
