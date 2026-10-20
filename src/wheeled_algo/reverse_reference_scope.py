"""Apply an authenticated reference controller only to straight high reverse."""
from copy import deepcopy
import json

import torch
from torch import nn


class ReverseReferenceScope(nn.Module):
    def __init__(self,parent,reference):
        super().__init__()
        self.parent=deepcopy(parent).eval().requires_grad_(False)
        self.reference=deepcopy(reference).eval().requires_grad_(False)

    def forward(self,obs):
        selected=(obs[:,:1]<-.75)&(obs[:,2:3].abs()<=1.5)
        return torch.where(selected,self.reference(obs),self.parent(obs))

    def export_graph(self,source):
        import onnx
        import numpy as np
        from onnx import helper,numpy_helper
        parent=self.parent.export_graph(source)
        reference=self.reference.export_graph(source)
        if ([(x.domain,x.version)for x in parent.opset_import]!=[(x.domain,x.version)for x in reference.opset_import]
                or len(parent.graph.input)!=1 or len(reference.graph.input)!=1):
            raise ValueError('Reference and parent graph interfaces differ')
        graph=deepcopy(parent);nodes=[];initializers=[];outputs=[]
        input_name=parent.graph.input[0].name
        # Copy both measured graphs, preserving every arithmetic node. This
        # avoids retracing the already exported original controller.
        for prefix,model in (('reverse_scope.parent.',parent),('reverse_scope.reference.',reference)):
            old_input=model.graph.input[0].name
            def rename(name):return input_name if name==old_input else prefix+name if name else ''
            outputs.append(rename(model.graph.output[0].name))
            for index,node in enumerate(model.graph.node):
                copied=deepcopy(node);copied.name=prefix+f'node{index}'
                copied.input[:]=[rename(n)for n in node.input];copied.output[:]=[rename(n)for n in node.output]
                nodes.append(copied)
            for value in model.graph.initializer:
                copied=deepcopy(value);copied.name=rename(value.name);initializers.append(copied)
        prefix='reverse_scope.'
        constants=dict(zero=np.array([0],dtype=np.int64),one=np.array([1],dtype=np.int64),
            two=np.array([2],dtype=np.int64),three=np.array([3],dtype=np.int64),
            axis=np.array([1],dtype=np.int64),step=np.array([1],dtype=np.int64),
            vx_limit=np.array(-.75,dtype=np.float32),yaw_limit=np.array(1.5,dtype=np.float32))
        initializers.extend(numpy_helper.from_array(v,prefix+k)for k,v in constants.items())
        nodes.extend([
            helper.make_node('Slice',[input_name,prefix+'zero',prefix+'one',prefix+'axis',prefix+'step'],[prefix+'vx']),
            helper.make_node('Slice',[input_name,prefix+'two',prefix+'three',prefix+'axis',prefix+'step'],[prefix+'yaw']),
            helper.make_node('Less',[prefix+'vx',prefix+'vx_limit'],[prefix+'reverse']),
            helper.make_node('Abs',[prefix+'yaw'],[prefix+'yaw_abs']),
            helper.make_node('LessOrEqual',[prefix+'yaw_abs',prefix+'yaw_limit'],[prefix+'straight']),
            helper.make_node('And',[prefix+'reverse',prefix+'straight'],[prefix+'selected']),
            helper.make_node('Where',[prefix+'selected',outputs[1],outputs[0]],[parent.graph.output[0].name])])
        del graph.graph.node[:];graph.graph.node.extend(nodes)
        del graph.graph.initializer[:];graph.graph.initializer.extend(initializers)
        del graph.graph.value_info[:]
        metadata={p.key:p.value for p in parent.metadata_props}
        metadata.update(actor_adaptation='reverse-reference-scope-v1',reference_scope=json.dumps(dict(
            active='vx<-.75m/s AND abs(yaw)<=1.5rad/s',
            remaining_states='exact authenticated original1338 function for identical observations',
            new_trainable_parameters=0,new_optimizer_updates=0),separators=(',',':')))
        onnx.helper.set_model_props(graph,metadata);onnx.checker.check_model(graph,full_check=True)
        return graph
