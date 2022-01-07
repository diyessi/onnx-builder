from onnx import TensorProto, helper
import numpy as np


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


class Value:
    def __init__(self, value_name, value_index, value_optional=False, **kwargs):
        super().__init__(**kwargs)
        self._value_name = value_name
        self._value_index = value_index
        self._value_optional = value_optional

    @property
    def value_index(self):
        return self._value_index

    @property
    def value_name(self):
        return self._value_name

    @property
    def value_optional(self):
        return self._value_optional

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


class Node:
    node_attributes = []

    def __init__(self, op_type, node_name=None, **kwargs):
        super().__init__(**kwargs)
        self._node_name = node_name
        self._op_type = op_type

    @property
    def op_type(self):
        return self._op_type

    @property
    def node_name(self):
        return self._node_name

    def used_node_output_values(self, exporter, used_values):
        return [value if (not value.value_optional or value in used_values) else None for value in self.node_output_values(exporter)]

    def build_node(self, exporter, used_values):
        node_name = exporter.exporter_node_name(self)
        input_names = [exporter.exporter_value_name(node_input_value)
                       for node_input_value in self.node_input_values(exporter)]
        output_names = [exporter.exporter_value_name(node_output_value)
                        for node_output_value in self.used_node_output_values(exporter, used_values)]
        return helper.make_node(
            self.op_type, input_names, output_names, node_name, **self.node_attribute_dict(exporter, node_name))

    def node_input_values(self, exporter):
        result = []
        for name in self.__class__.node_inputs:
            value = getattr(self, name)
            if value:
                result.append(value)
            else:
                break
        return result

    def node_attribute_dict(self, exporter, node_name):
        result = {}
        for name in self.__class__.node_attributes:
            value = getattr(self, name)
            if value:
                result[name] = value
        return result

    def node_output_values(self, exporter):
        result = []
        for name in self.__class__.node_outputs:
            value = getattr(self, name)
            if value:
                result.append(value)
            else:
                break
        return result


class DefaultNodeValue(Node, Value):
    node_outputs = ['output']

    def __init__(self, value_name='output', value_index=0, node_name=None, **kwargs):
        super().__init__(value_name=value_name,
                         value_index=value_index, node_name=node_name, **kwargs)

    @property
    def value_node(self):
        return self

    @property
    def value_index(self):
        return 0

    @property
    def output(self):
        return self


class SecondaryValue(Value):
    # For ops that have more than one output
    def __init__(self, value_node, value_name, value_index, optional=False, **kwargs):
        super().__init__(value_name=value_name, value_index=value_index, **kwargs)
        self._value_node = value_node

    @property
    def value_node(self):
        return self._value_node

    def __getattr__(self, name):
        return getattr(self.value_node, name)


class Placeholder(DefaultNodeValue):
    node_inputs = []
    node_outputs = ['output']

    def __init__(self, elt_type=np.float32, shape=None, **kwargs):
        super().__init__(op_type='Placeholder', **kwargs)
        self._elt_type = elt_type
        self._shape = shape

    @property
    def elt_type(self):
        return self._elt_type

    @property
    def shape(self):
        return self._shape


class Abs(DefaultNodeValue):
    node_inputs = ['X']
    node_outputs = ['Y']

    def __init__(self, X, **kwargs):
        super().__init__(op_type='Abs', value_name='Y', **kwargs)
        self._X = X

    # Inputs
    @property
    def X(self):
        return self._X

    # Outputs
    @property
    def Y(self):
        return self


class Add(DefaultNodeValue):
    node_inputs = ['A', 'B']
    node_outputs = ['C']

    def __init__(self, A, B, **kwargs):
        super().__init__(op_type='Add', value_name='C', **kwargs)
        self._A = A
        self._B = B

    # Inputs
    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    # Outputs
    @property
    def C(self):
        return self


class BatchNormalization(DefaultNodeValue):
    node_inputs = ['X', 'scale', 'B', 'input_mean', 'input_var']
    node_attributes = ['epsilon', 'momentum', 'training_mode']
    node_outputs = ['Y', 'running_mean', 'running_var']

    def __init__(self, X, scale, B, input_mean, input_var, epsilon=None, momentum=None, training_mode=None, **kwargs):
        super().__init__(op_type='BatchNormalization', value_name='Y', **kwargs)
        self._X = X
        self._scale = scale
        self._B = B
        self._input_mean = input_mean
        self._input_var = input_var
        self._epsilon = epsilon
        self._momentum = momentum
        self._training_mode = training_mode

    # Inputs
    @property
    def X(self):
        return self._X

    @property
    def scale(self):
        return self._scale

    @property
    def B(self):
        return self._B

    @property
    def input_mean(self):
        return self._input_mean

    @property
    def input_var(self):
        return self._input_var

    # Attributes
    @property
    def epsilon(self):
        return self._epsilon

    @property
    def momentum(self):
        return self._momentum

    @property
    def training_mode(self):
        return self._training_mode

    # Outputs
    @property
    def Y(self):
        return self

    @property
    def running_mean(self):
        return SecondaryValue(self, 'running_mean', 1, value_optional=True)

    @property
    def running_var(self):
        return SecondaryValue(self, 'running_var', 2, value_optional=True)


class Constant(DefaultNodeValue):
    node_inputs = []

    def __init__(self, value, dtype=None, value_name=None, **kwargs):
        super().__init__(op_type='Constant', **kwargs)
        if not type(value) is np.ndarray:
            value = np.array(value, dtype)
        if type(value) is np.ndarray and value.dtype in {np.dtype('int64'), np.dtype('float32')}:
            self._value = value
        else:
            raise ValueError()

    def node_attribute_dict(self, exporter, node_name):
        value_name = exporter.exporter_value_name(self)
        value = self._value
        if value.dtype is np.dtype('int64'):
            return {'value': helper.make_tensor(name=value_name, data_type=TensorProto.INT64, dims=value.shape, vals=value.flatten().astype(np.int64))}
        elif value.dtype is np.dtype('float32'):
            return {'value': helper.make_tensor(name=value_name, data_type=TensorProto.FLOAT, dims=value.shape, vals=value.flatten().astype(np.float32))}


class LSTM(Node):
    node_inputs = ['X', 'W', 'R', 'B',
                   'sequence_lens', 'initial_h', 'initial_c', 'P']
    node_attributes = ['activation_alpha', 'activation_beta', 'activations',
                       'clip', 'direction', 'hidden_size', 'input_forget', 'layout']
    node_outputs = ['Y', 'Y_h', 'Y_c']

    def __init__(self, X, W, R, B=None, sequence_lens=None, initial_h=None, initial_c=None, P=None,
                 activation_alpha=None, activation_beta=None, activations=None, clip=None, direction=None, hidden_size=None,
                 input_forget=None, layout=None, **kwargs):
        super().__init__(op_type='LSTM')
        self.X = X
        self.W = W
        self.R = R
        self.B = B
        self.sequence_lens = sequence_lens
        self.initial_h = initial_h
        self.initial_c = initial_c
        self.P = P
        self.activation_alpha = activation_alpha
        self.activation_beta = activation_beta
        self.activations = activations
        self.clip = clip
        self.direction = direction
        self.hidden_size = hidden_size
        self.input_forget = input_forget
        self.layout = layout

    # Inputs exposed above
    # Attributes exposed above

    # Outputs
    @property
    def Y(self):
        return SecondaryValue(self, 'Y', 0, value_optional=True)

    @property
    def Y_h(self):
        return SecondaryValue(self, 'Y_h', 1, value_optional=True)

    @property
    def Y_c(self):
        return SecondaryValue(self, 'Y_c', 2, value_optional=True)


class MatMul(DefaultNodeValue):
    node_inputs = ['A', 'B']
    node_outputs = ['Y']

    def __init__(self, A, B, **kwargs):
        super().__init__(op_type='MatMul', value_name='Y', **kwargs)
        self._A = A
        self._B = B

    # Inputs
    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    # Outputs
    @property
    def Y(self):
        return self


class Mul(DefaultNodeValue):
    node_inputs = ['A', 'B']
    node_outputs = ['C']

    def __init__(self, A, B, **kwargs):
        super().__init__(op_type='Mul', value_name='C', **kwargs)
        self._A = A
        self._B = B

    # Inputs
    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    # Outputs
    @property
    def C(self):
        return self


class Pad(DefaultNodeValue):
    node_inputs = ['input', 'pads', 'constant_value']
    node_attributes = ['mode']

    def __init__(self, input, pads, constant_value=None, mode=None, **kwargs):
        super().__init__(op_type='Pad', **kwargs)
        self.input = input
        self.pads = pads
        self.constant_value = constant_value
        self.mode = mode


class Reshape(DefaultNodeValue):
    node_inputs = ['data', 'shape']
    node_attributes = ['allowzero']
    node_outputs = ['reshaped']

    def __init__(self, data, shape, allowzero=None, **kwargs):
        super().__init__(op_type='Reshape', value_name='reshaped', **kwargs)
        self.data = data
        self.shape = shape
        self.allowzero = allowzero

    @property
    def reshaped(self):
        return self


class Sigmoid(DefaultNodeValue):
    node_inputs = ['X']
    node_outputs = ['Y']

    def __init__(self, X, **kwargs):
        super().__init__(op_type='Sigmoid', **kwargs)
        self.X = X

    @property
    def Y(self):
        return self


class Slice(DefaultNodeValue):
    node_inputs = ['data', 'starts', 'ends', 'axes', 'steps']

    def __init__(self, data, starts, ends, axes=None, steps=None, **kwargs):
        super().__init__(op_type='Slice', **kwargs)
        self.data = data
        self.starts = starts
        self.ends = ends
        self.axes = axes
        self.steps = steps


class Sub(DefaultNodeValue):
    node_inputs = ['A', 'B']
    node_outputs = ['C']

    def __init__(self, A, B, **kwargs):
        super().__init__(op_type='Sub', value_name='C', **kwargs)
        self._A = A
        self._B = B

    # Inputs
    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    # Outputs
    @property
    def C(self):
        return self


class Tanh(DefaultNodeValue):
    node_inputs = ['input']
    node_outputs = ['output']

    def __init__(self, input, **kwargs):
        super().__init__(op_type='Tanh', **kwargs)
        self.input = input


class Tile(DefaultNodeValue):
    node_inputs = ['input', 'repeats']

    def __init__(self, input, repeats, **kwargs):
        super().__init__(op_type='Tile', **kwargs)
        this.input = input
        this.repeats = repeats


class Transpose(DefaultNodeValue):
    node_inputs = ['data']
    node_outputs = ['transposed']
    node_attributes = ['perm']

    def __init__(self, data, perm, **kwargs):
        super().__init__('Transpose', value_name='transposef', **kwargs)
        self.data = data
        self.perm = perm

    @property
    def transposed(self):
        return self
