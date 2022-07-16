import onnx
import numpy as np
from builder.exporter import onnx_type


class Value:
    def __init__(self, value_name, *args, **kwargs):
        super().__init__()
        self.value_name = value_name

    def node_input_values(self):
        return []

    @property
    def value_node(self):
        return None

    @staticmethod
    def as_value(value):
        if isinstance(value, Value):
            return value
        if not type(value) is np.ndarray:
            value = np.array(value)
        if type(value) is np.ndarray:
            dtype = value.dtype
            return Constant(value=onnx.helper.make_tensor(name='', data_type=onnx_type(dtype), dims=value.shape, vals=value.flatten().astype(dtype)))
        else:
            raise ValueError()

    @property
    def value_optional(self):
        return self.value_descriptor.option == onnx.defs.OpSchema.FormalParameterOption.Optional

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __sub__(self, other):
        return Sub(self, other)


class NodeValue(Value):
    def __init__(self, value_index, **kwargs):
        self.value_index = value_index
        super().__init__(value_name=self.value_descriptor.name, **kwargs)

    @property
    def value_descriptor(self):  
        return self.value_node.op_schema.outputs[self.value_index]

    def __eq__(self, other):
        return self.value_node is other.value_node and self.value_index == other.value_index

    def __hash__(self):
        return hash((id(self.value_node), self.value_index))


class Node(NodeValue):

    def __getattr__(self, name):
        try:
            for index, value_descriptor in enumerate(self.op_schema.outputs):
                if value_descriptor.name == name:
                    if index == 0:
                        return self
                    else:
                        return SecondaryValue(self, index)
        except ValueError:
            raise AttributeError(
                f"{self.op_type} object has no attribute '{name}'")

    def __init__(self, *args, op_name=None, **kwargs):
        if op_name is None:
            op_name = type(self).__name__
        self.op_schema = onnx.defs.get_schema(op_name)
        super().__init__(0, **kwargs)
        for index, descriptor in enumerate(self.op_schema.inputs):
            if descriptor.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                setattr(self, descriptor.name, [
                        Value.as_value(v) for v in args[index:]])
                break
            if index < len(args):
                setattr(self, descriptor.name, Value.as_value(args[index]))
            elif descriptor.name in kwargs:
                setattr(self, descriptor.name, Value.as_value(kwargs[descriptor.name]))
            else:
                setattr(self, descriptor.name, None)

        for name, value in kwargs.items():
            if name in self.op_schema.attributes:
                if isinstance(value, type):
                    value = onnx_type(type)
                setattr(self, name, value)

    @property
    def op_type(self):
        return self.op_schema.name

    @property
    def value_node(self):
        return self

    def node_attribute_values(self):
        attributes = {}
        for name in self.op_schema.attributes:
            value = getattr(self, name)
            if value is not None:
                attributes[name] = value
        return attributes

    def node_input_values(self):
        result = []
        for descriptor in self.op_schema.inputs:
            if descriptor.option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
                result += getattr(self, descriptor.name)
            else:
                result.append(getattr(self, descriptor.name))
        while result and not result[-1]:
            result.pop()
        return result

    def node_output_values(self):
        result = []
        for value_descriptor in self.op_schema.outputs:
            value = getattr(self, value_descriptor.name)
            if value:
                result.append(value)
            else:
                break
        return result


class SecondaryValue(NodeValue):
    # For ops that have more than one output
    def __init__(self, value_node, value_index):
        self._value_node = value_node
        super().__init__(value_index)

    @property
    def value_node(self):
        return self._value_node

    def __getattr__(self, name):
        return getattr(self.value_node, name)


class Input(Value):
    def __init__(self, exporter, name, elt_type=None, shape=None):
        super().__init__(value_name=exporter.add_graph_input(name, self, elt_type=elt_type, shape=shape))
        
def make_op_factory(op_name):
    return lambda *args, **kwargs : Node(*args, op_name=op_name, **kwargs)


for op in ['Abs', 'Add', 'BatchNormalization', 'Cast', 'Concat',  'Constant', 'Conv', 'LSTM', 'MatMul', 'MaxPool', 'Mod',
           'Mul', 'OneHot', 'Pad', 'Relu', 'Reshape', 'Resize', 'Sigmoid', 'Slice', 'Sub', 'Tanh', 'Tile', 'Transpose']:
    globals()[op] = make_op_factory(op)
