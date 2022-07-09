import onnx
import numpy as np
from builder.exporter import onnx_type


def _append_if(list_, *values):
    for value in values:
        if value:
            list_.append(value)
    return list_


def _update_if(dict_, values):
    for k, v in values.items():
        if v:
            dict_[k] = v
    return dict_


class OpNotSupportedError(ValueError):
    pass


class ParameterDescriptor:
    def __init__(self, name, index, optional, variadic):
        self.name = name
        self.index = index
        self.optional = optional
        self.variadic = variadic


class ValueDescriptor:
    def __init__(self, name, index, optional):
        self.name = name
        self.index = index
        self.optional = optional


class Value:
    def __init__(self, value_descriptor, *args, **kwargs):
        super().__init__()
        self.value_descriptor = value_descriptor

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
    def value_index(self):
        return self.value_descriptor.index

    @property
    def value_name(self):
        return self.value_descriptor.name

    @property
    def value_optional(self):
        return self.value_descriptor.optional

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __eq__(self, other):
        return self.value_node is other.value_node and self.value_index == other.value_index

    def __hash__(self):
        return hash((id(self.value_node), self.value_index))


class Node(Value):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            op_schema = onnx.defs.get_schema(cls.__name__)
            cls.node_parameter_descriptors = [ParameterDescriptor(input.name, index, input.option == onnx.defs.OpSchema.FormalParameterOption.Optional,
                                              input.option == onnx.defs.OpSchema.FormalParameterOption.Variadic) for index, input in enumerate(op_schema.inputs)]
            cls.node_attributes = [
                attribute for attribute in op_schema.attributes]
            cls.node_value_descriptors = [ValueDescriptor(
                output.name, index, output.option == onnx.defs.OpSchema.FormalParameterOption.Optional) for index, output in enumerate(op_schema.outputs)]

        except onnx.onnx_cpp2py_export.defs.SchemaError:
            pass

    def __getattr__(self, name):
        cls = self.__class__
        try:
            for value_descriptor in cls.node_value_descriptors:
                if value_descriptor.name == name:
                    if value_descriptor.index == 0:
                        return self
                    else:
                        return SecondaryValue(self, value_descriptor)
        except ValueError:
            raise AttributeError(
                f"{cls.__name__} object has no attribute '{name}'")

    def __init__(self, *args, **kwargs):
        super().__init__(
            value_descriptor=self.__class__.node_value_descriptors[0], **kwargs)
        for descriptor in type(self).node_parameter_descriptors:
            if descriptor.variadic:
                setattr(self, descriptor.name, [Value.as_value(v) for v in args[descriptor.index:]])
                break
            if descriptor.index < len(args):
                setattr(self, descriptor.name, Value.as_value(args[descriptor.index]))
            elif descriptor.name in kwargs:
                setattr(self, descriptor.name, Value.as_value(kwargs[descriptor.name]))
            else:
                setattr(self, descriptor.name, None)

        for name, value in kwargs.items():
            if name in self.__class__.node_attributes:
                setattr(self, name, value)

    @property
    def op_type(self):
        return type(self).__name__

    @property
    def value_node(self):
        return self

    def used_node_output_values(self, exporter, used_values):
        return [value if (not value.value_optional or value in used_values) else None for value in self.node_output_values(exporter)]

    def build_node(self, exporter, used_values):
        node_name = exporter.exporter_node_name(self)
        input_names = [exporter.exporter_value_name(node_input_value)
                       for node_input_value in self.node_input_values(exporter)]
        output_names = [exporter.exporter_value_name(node_output_value)
                        for node_output_value in self.used_node_output_values(exporter, used_values)]
        attributes = {}
        for name in self.__class__.node_attributes:
            value = getattr(self, name)
            if isinstance(value, onnx.TensorProto):
                value.name = exporter.exporter_value_name(self.value_node)
            elif isinstance(value, type):
                value = onnx_type(type)
            if value is not None:
                attributes[name] = value

        return onnx.helper.make_node(
            self.op_type, input_names, output_names, node_name, **attributes)

    def node_input_values(self, exporter):
        result = []
        for descriptor in self.__class__.node_parameter_descriptors:
            if descriptor.variadic:
                result += getattr(self, descriptor.name)
            else:
                result.append(getattr(self, descriptor.name))
        while result and not result[-1]:
            result.pop()
        return result

    def node_output_values(self, exporter):
        result = []
        for value_descriptor in self.__class__.node_value_descriptors:
            value = getattr(self, value_descriptor.name)
            if value:
                result.append(value)
            else:
                break
        return result


class SecondaryValue(Value):
    # For ops that have more than one output
    def __init__(self, value_node, value_descriptor):
        super().__init__(value_descriptor)
        self._value_node = value_node

    @property
    def value_node(self):
        return self._value_node

    def __getattr__(self, name):
        return getattr(self.value_node, name)


class Placeholder(Node):
    node_parameter_descriptors = []
    node_attributes = ['elt_type', 'shape']
    node_value_descriptors = [ValueDescriptor('output', 0, False)]

    def __init__(self, elt_type=np.float32, shape=None, **kwargs):
        super().__init__(elt_type=elt_type, shape=shape, **kwargs)


class Abs(Node):
    pass


class Add(Node):
    pass


class BatchNormalization(Node):
    pass


class Cast(Node):
    pass

class Concat(Node):
    pass


class Constant(Node):
    pass

class Conv(Node):
    pass


class LSTM(Node):
    pass


class MatMul(Node):
    pass


class MaxPool(Node):
    pass


class Mod(Node):
    pass


class Mul(Node):
    pass


class OneHot(Node):
    pass


class Pad(Node):
    pass


class Relu(Node):
    pass


class Reshape(Node):
    pass


class Resize(Node):
    pass


class Sigmoid(Node):
    pass


class Slice(Node):
    pass


class Sub(Node):
    pass


class Tanh(Node):
    pass


class Tile(Node):
    pass


class Transpose(Node):
    pass
