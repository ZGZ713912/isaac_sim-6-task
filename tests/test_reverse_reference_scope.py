import numpy as np
import onnxruntime as ort
import torch

from wheeled_algo.intent_speed_reference import IntentSpeedReferenceActor
from wheeled_algo.reverse_reference_scope import ReverseReferenceScope
from test_static_grip_adaptation import means,observations
from test_guarded_feedback_residual import write_source
from types import SimpleNamespace


def test_only_straight_high_reverse_uses_the_reference(means,tmp_path):
    parent=means[1]
    reference=IntentSpeedReferenceActor(parent,'a'*64,[.15,.3,.02,.02],[-.04,-.01],.5)
    scoped=ReverseReferenceScope(parent,reference)
    obs=observations();obs[:5,0]=torch.tensor([3.,-3.,0.,-.5,-3.]);obs[:5,2]=torch.tensor([0.,0.,4.,0.,2.])
    chosen=(obs[:,:1]<-.75)&(obs[:,2:3].abs()<=1.5)
    assert torch.equal(scoped(obs),torch.where(chosen,reference(obs),parent(obs)))
    assert torch.equal(scoped(obs)[~chosen[:,0]],parent(obs)[~chosen[:,0]])
    assert not any(p.requires_grad for p in scoped.parameters())
    source=tmp_path/'source.onnx';write_source(source,SimpleNamespace(mlp=parent.command.base))
    graph=scoped.export_graph(source)
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    runtime=ort.InferenceSession(graph.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    original=ort.InferenceSession(parent.export_graph(source).SerializeToString(),sess_options=options,providers=['CPUExecutionProvider'])
    for values in (obs,obs[:1]):
        np.testing.assert_allclose(runtime.run(None,{'obs':values.numpy()})[0],scoped(values).numpy(),rtol=5e-6,atol=5e-6)
    for frame in obs[~chosen[:,0]]:
        values={'obs':frame.numpy()[None]}
        np.testing.assert_array_equal(runtime.run(None,values)[0],original.run(None,values)[0])
