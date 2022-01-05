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
    def __init__(self, value_name, value_index, **kwargs):
        super().__init__(**kwargs)
        self._value_name = value_name
        self._value_index = value_index

    @property
    def value_index(self):
        return self._value_index

    @property
    def value_name(self):
        return self._value_name

    @property
    def value_optional(self):
        return False

    def __add__(self, other):
        return Add(self, other)

    def __eq__(self, other):
        return self.value_node is other.value_node and self.value_index == other.value_index

    def __hash__(self):
        return hash((id(self.value_node), self.value_index))


class Node:
    def __init__(self, op_type, node_name=None, **kwargs):
        super().__init__(**kwargs)
        self._node_name = node_name
        self._op_type = op_type
        self._node_outputs_used = set()

    @property
    def op_type(self):
        return self._op_type

    @property
    def node_name(self):
        return self._node_name

    @property
    def node_outputs_used(self):
        return self._node_outputs_used

    def used_node_output_values(self, builder):
        saved_node_outputs_used = set(self._node_outputs_used)
        node_output_values = self.node_output_values(builder)
        self._node_outputs_used = saved_node_outputs_used
        return [None if value in saved_node_outputs_used else value for value in node_output_values]

    def build_node(self, builder):
        node_name = builder.builder_node_name(self)
        input_names = [builder.builder_value_name(node_input_value)
                       for node_input_value in self.node_input_values(builder)]
        output_names = [builder.builder_value_name(node_output_value)
                        for node_output_value in self.used_node_output_values(builder)]
        return helper.make_node(
            self.op_type, input_names, output_names, node_name, **self.node_attributes(builder, node_name))

    def node_output_values(self, builder):
        return [self.output]


class DefaultNodeValue(Node, Value):
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

    def node_attributes(self, builder, node_name):
        return {}

    def node_outputs(self, builder):
        return [self]


class SecondaryValue(Value):
    # For ops that have more than one output
    def __init__(self, value_node, value_name, value_index, optional=False, **kwargs):
        super().__init__(value_name=value_name, value_index=value_index, **kwargs)
        self._value_node = value_node
        self._optional = optional
        if optional:
            self.value_node.node_outputs_used.add(self.value_index)

    @property
    def value_node(self):
        return self._value_node

    @property
    def value_optional(self):
        return self._optional

    def __getattr__(self, name):
        return getattr(self.value_node, name)


class Placeholder(DefaultNodeValue):
    def __init__(self, elt_type, shape, **kwargs):
        super().__init__(op_type='Placeholder', value_name='output', **kwargs)
        self._elt_type = elt_type
        self._shape = shape

    @property
    def elt_type(self):
        return self._elt_type

    @property
    def shape(self):
        return self._shape

    @property
    def value_type(self):
        return self.elt_type, self.shape

    # Node protocol
    def node_input_values(self, builder):
        return []


class Abs(DefaultNodeValue):
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

    # Node protocol
    def node_input_values(self, builder):
        return [self.X]


class Add(DefaultNodeValue):
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

    # Node protocol
    def node_input_values(self, builder):
        return [self.A, self.B]


class BatchNormalization(DefaultNodeValue):
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
        return SecondaryValue(self, 'running_mean', 1, optional=True)

    @property
    def running_var(self):
        return SecondaryValue(self, 'running_var', 2, optional=True)

    # Node protocol
    def node_input_values(self, builder):
        if builder.opset >= 15:
            return [self.X, self.scale, self.B, self.input_mean, self.input_var]
        else:
            raise OpNotSupportedError()

    def node_attributes(self, builder, node_name):
        node_attributes = {}
        if builder.opset >= 15:
            if self.epsilon:
                node_attributes['epsilon'] = self.epsilon
            if self.momentum:
                node_attributes['momentum'] = self.momentum
            if self.training_mode:
                node_attributes['self.training_mode'] = self.training_mode
        return node_attributes

    def node_output_values(self, builder):
        if builder.opset >= 15:
            return [self.Y, self.running_mean, self.running_var]
        else:
            raise OpNotSupportedError()


class Constant(DefaultNodeValue):
    def __init__(self, value, dtype=None, value_name=None, **kwargs):
        super().__init__(op_type='Constant', **kwargs)
        if not type(value) is np.ndarray:
            value = np.array(value, dtype)
        if type(value) is np.ndarray and value.dtype in {np.dtype('int64'), np.dtype('float32')}:
            self._value = value
        else:
            raise ValueError()

    def node_input_values(self, builder):
        return []

    def node_attributes(self, builder, node_name):
        value_name = self.value_name or node_name
        value = self._value
        if value.dtype is np.dtype('int64'):
            return {'value': helper.make_tensor(name=value_name, data_type=TensorProto.INT64, dims=value.shape, vals=value.flatten().astype(np.int64))}
        elif value.dtype is np.dtype('float32'):
            return {'value': helper.make_tensor(name=value_name, data_type=TensorProto.FLOAT, dims=value.shape, vals=value.flatten().astype(np.float32))}


class LSTM(Node):
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
        return SecondaryValue(self, 'Y', 0, optional=True)

    @property
    def Y_h(self):
        return SecondaryValue(self, 'Y_h', 1, optional=True)

    @property
    def Y_c(self):
        return SecondaryValue(self, 'Y_c', 2, optional=True)

    def node_input_values(self, builder):
        if builder.opset >= 14:
            inputs = [self.X, self.W, self.R]
            return _append_if(inputs, self.B, self.sequence_lens,
                              self.initial_h, self.initial_c, self.P)
        else:
            raise OpNotSupportedError()

    def node_attributes(self, builder, node_name):
        if builder.opset >= 14:
            return _update_if({}, {'activation_alpha': self.activation_alpha,
                                   'activation_beta': self.activation_beta,
                                   'activations': self.activations,
                                   'clip': self.clip,
                                   'direction': self.direction,
                                   'hidden_size': self.hidden_size,
                                   'input_forget': self.input_forget,
                                   'layout': self.layout})

    def node_output_values(self, builder):
        return [self.Y, self.Y_h, self.Y_c]


class Pad(DefaultNodeValue):
    def __init__(self, input, pads, constant_value=None, mode=None, **kwargs):
        super().__init__(op_type='Pad', **kwargs)
        self.input = input
        self.pads = pads
        self.constant_value = constant_value
        self.mode = mode

    def node_input_values(self, builder):
        node_input_values = []
        if builder.opset >= 12:
            node_input_values = [self.input, self.pads]
            if self.constant_value:
                return node_input_values.append(self.constant_value)
        else:
            raise OpNotSupportedError()
        return node_input_values

    def node_attributes(self, builder, node_name):
        node_attributes = {}
        if builder.opset >= 12:
            if self.mode:
                node_attributes['mode'] = self.mode
        else:
            raise OpNotSupportedError()
        return node_attributes
