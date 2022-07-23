from builder.exporter import Exporter, opset_version
from builder.onnxops import Input, Node
import builder.onnxops
import onnx

if False:
    from builder.onnxops import Abs, Add
    from builder.onnxops import BatchNormalization
    from builder.onnxops import Cast, Concat, Constant, Conv
    from builder.onnxops import LSTM
    from builder.onnxops import MatMul, MaxPool, Mod, Mul
    from builder.onnxops import OneHot
    from builder.onnxops import Pad
    from builder.onnxops import Relu, Reshape, Resize
    from builder.onnxops import Sigmoid, Slice, Sub
    from builder.onnxops import Tanh, Tile, Transpose

if False:
    __all__ = ['Exporter',
            'Input',
            'Abs', 'Add',
            'BatchNormalization',
            'Cast', 'Concat', 'Constant', 'Conv',
            'LSTM',
            'MatMul', 'MaxPool', 'Mod', 'Mul',
            'OneHot',
            'Pad',
            'Relu', 'Reshape', 'Resize',
            'Sigmoid', 'Slice', 'Sub',
            'Tile', 'Tanh', 'Transpose']

__all__ = ['Exporter', 'Input']

def make_op_factory(op_name):
    return lambda *args, **kwargs : Node(*args, op_name=op_name, **kwargs).value_node

def initialize_ops(version):
    opnames = set()
    for s in onnx.defs.get_all_schemas_with_history():
        if s.since_version <= version:
            opnames.add(s.name)
    for op in opnames:
        opfun = make_op_factory(op)
        setattr(builder.onnxops, op, opfun)
        globals()[op] = opfun
        __all__.append(op)

initialize_ops(opset_version)
