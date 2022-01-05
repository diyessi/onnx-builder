import onnx
from onnx import TensorProto
from onnx import helper

class Builder:
    def __init__(self, name, opset=16):
        self.opset = opset
        self.name = name
        # list of input value info
        self.inputs_vi = []
        # list of output names
        self.outputs = []
        # List of node defs
        self.node_defs = []

        # value -> value name map
        self._name_for_value = {}
        # node -> node name map
        self._name_for_node = {}

        self._node_names = set()
        self._node_count = 0

        self._defined_value_names = {}

    def named(self, name, value):
        self._defined_value_names[value] = name
        return value

    def build(self, inputs, outputs):
        for input in inputs:
            self.add_input(input)
        # Add nodes down to input values
        todo = [output.value_node for output in outputs]
        done = set(inputs)
        while todo:
            last = todo[-1]
            if last in done:
                todo.pop()
                continue
            ready = True
            for node_input_value in last.node_input_values(self):
                if node_input_value.value_node in done:
                    continue
                todo.append(node_input_value.value_node)
                ready = False
            if not ready:
                continue
            self.node_defs.append(last.build_node(self))
            done.add(last)
            todo.pop()

        for output in outputs:
            self.add_output(output)

        graph_def = helper.make_graph(
            self.node_defs,
            self.name,
            self.inputs_vi,
            self.outputs)
        self.model_def = helper.make_model(graph_def,
                                           producer_name='Builder')

    def unique_node_name(self, root_name, use_count):
        if not use_count and root_name not in self._node_names:
            self._node_names.add(root_name)
            return root_name
        name = root_name + "_" + str(self._node_count)
        self._node_count += 1
        self._node_names.add(name)
        return name

    def builder_node_name(self, node):
        node_name = self._name_for_node.get(node, None)
        if node_name:
            return node_name
        if node.node_name:
            # Node has a preferred name
            node_name = self.unique_node_name(node.node_name, False)
        else:
            node_name = self.unique_node_name(node.op_type, True)
        self._name_for_node[node] = node_name
        return node_name

    def builder_value_name(self, value):
        if not value:
            return ""
        name = self._name_for_value.get(value, None)
        if name:
            return name
        defined_value_name = self._defined_value_names.get(value, None)
        if defined_value_name:
            return defined_value_name
        multi_output = len(value.used_node_output_values(self)) > 1
        node_name = self.builder_node_name(value.value_node)
        if multi_output:
            if value.value_name:
                name = node_name+":"+value.value_name
            else:
                name = node_name + ":" + str(value.value_index)
        else:
            name = node_name
        self._name_for_value[value] = name
        return name


    def add_input(self, input):
        value_name = self.builder_value_name(input)
        elt_type, shape = input.value_type
        tvi = helper.make_tensor_value_info(value_name, elt_type, shape)
        self.inputs_vi.append(tvi)

    def add_output(self, output):
        name = self.builder_value_name(output)
        tvi = helper.make_tensor_value_info(name, TensorProto.FLOAT, [])
        self.outputs.append(tvi)

    def add_node(self, node):
        self.node_defs.append(node.build_node(self))
