import onnx
import numpy as np
from builder import *


def run():
    input = Placeholder()

    for typ in (np.int8, np.uint8, np.int16, np.uint16):
        exporter = Exporter()
        exporter.add_graph_input('input', input, shape=[1, 2, 3], elt_type=typ)
        exporter.add_graph_output('output', Cast(
            input + Constant(np.ones([1, 2, 3], dtype=typ)), np.float32))
        md = exporter.export(f'{typ.__name__}_to_float')
        with open(f'{typ.__name__}_to_float.onnxtxt', 'w') as f:
            f.write(str(md))
        onnx.save(md, f'{typ.__name__}_to_float.onnx')


if __name__ == "__main__":
    run()
