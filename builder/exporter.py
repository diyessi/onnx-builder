import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper

_type_conversion = {
    None: TensorProto.UNDEFINED,
    np.float32: TensorProto.FLOAT,
    np.uint8: TensorProto.UINT8,
    np.int8: TensorProto.INT8,
    np.uint16: TensorProto.UINT16,
    np.int16: TensorProto.INT16,
    np.int32: TensorProto.INT32,
    np.int64: TensorProto.INT64,
    str: TensorProto.STRING,
    bool: TensorProto.BOOL,
    np.float16: TensorProto.FLOAT16,
    np.float64: TensorProto.DOUBLE,
    np.uint32: TensorProto.UINT32,
    np.uint64: TensorProto.UINT64,
    np.complex64: TensorProto.COMPLEX64,
    np.complex128: TensorProto.COMPLEX128,
    # np.bfloat16 : TensorProto.BFLOAT16,
    float: TensorProto.DOUBLE,
    int: TensorProto.INT64,
    complex: TensorProto.COMPLEX128,
    np.dtype('int8'): TensorProto.INT8,
    np.dtype('uint8') : TensorProto.UINT8,
    np.dtype('int16'): TensorProto.INT16,
    np.dtype('uint16') : TensorProto.UINT16,
    np.dtype('int32'):  TensorProto.INT32,
    np.dtype('int64'):  TensorProto.INT64,
    np.dtype('float32'):  TensorProto.FLOAT,
    np.dtype('float64'): TensorProto.DOUBLE,
}


def onnx_type(typ):
    return _type_conversion.get(typ, TensorProto.UNDEFINED)


class Exporter:
    def __init__(self, opset=16):
        self.opset = opset

        # Provided inputs/outputs
        self._inputs = []
        self._outputs = []

        # lists of value info
        self._inputs_vi = []
        self._outputs_vi = []


        # value -> value name map
        self._name_for_value = {}
        # node -> node name map
        self._name_for_node = {}

        self._node_names = set()
        self._value_names = set()
        self._node_count = 0
        self.used_values = set()

    def export(self, model_name):
        # Add nodes down to input values
        self.used_values.update(self._outputs)
        self.used_values.update(self._inputs)
        nodes = []
        todo = [output.value_node for output in self._outputs]
        done = set(self._inputs)
        while todo:
            last = todo[-1]
            if last in done:
                todo.pop()
                continue
            ready = True
            for node_input_value in last.node_input_values(self):
                if node_input_value is not None:
                    self.used_values.add(node_input_value)
                    if node_input_value.value_node in done:
                        continue
                    todo.append(node_input_value.value_node)
                    ready = False
            if not ready:
                continue
            nodes.append(last)
            done.add(last)
            todo.pop()

        node_defs = []
        for node in nodes:
            node_defs.append(node.build_node(self, self.used_values))

        graph_def = helper.make_graph(
            node_defs,
            model_name,
            self._inputs_vi,
            self._outputs_vi)

        op = onnx.OperatorSetIdProto()
        op.version = 16

        return helper.make_model(graph_def,
                                 opset_imports=[op],
                                 producer_name='ONNX Builder')

    def _take_node_name(self, node, node_name):
        if not node_name or node_name in self._node_names:
            return False
        self._node_names.add(node_name)
        self._name_for_node[node] = node_name
        return node_name

    def _next_op_name(self, node):
        node_name = self._take_node_name(
            node, node.op_type + "_" + str(self._node_count))
        self._node_count += 1
        return node_name

    def exporter_node_name(self, node):
        return (self._take_node_name(node, self._name_for_node.get(node, None)) or
                self._take_node_name(node, node.node_name) or
                self._next_op_name(node))

    def exporter_value_name(self, value, name=None):
        if not value:
            return ""

        name_for_value = self._name_for_value.get(value, None)
        if name_for_value:
            # Already registered
            return name_for_value

        if not name or name in self._value_names:
            # No name, or someone else has the requested name
            multi_output = len(value.node_output_values(self)) > 1
            node_name = self.exporter_node_name(value.value_node)
            if multi_output:
                if value.value_name:
                    name = node_name+":"+value.value_name
                else:
                    name = node_name + ":" + str(value.value_index)
            else:
                name = node_name
        self._name_for_value[value] = name
        self._value_names.add(name)
        return name

    def add_graph_input(self, name, input, elt_type=None, shape=None):
        """Add a graph input"""
        value_name = self.exporter_value_name(input, name)
        elt_type = elt_type or input.elt_type
        elt_type = _type_conversion.get(elt_type, elt_type)
        shape = input.shape if shape is None else shape
        tvi = helper.make_tensor_value_info(value_name, elt_type, shape)
        self._inputs.append(input)
        self._inputs_vi.append(tvi)
        return self

    def add_graph_output(self, name, output, elt_type=None, shape='*'):
        """Add a graph output"""
        # Capture the name
        value_name = self.exporter_value_name(output, name)
        elt_type = _type_conversion.get(elt_type, elt_type)
        tvi = helper.make_tensor_value_info(value_name, elt_type, shape)
        self._outputs.append(output)
        self._outputs_vi.append(tvi)
        return self
        
