import onnx
import numpy as np
from builder import *

_camelName = {
    np.int8: 'Int8',
    np.uint8: 'UInt8',
    np.int16: 'Int16',
    np.uint16: 'UInt16',
    np.int32: 'Int32',
    np.int64: 'Int64'
}


def camel_name(typ):
    return _camelName[typ]

def convert_tests():
    input = Placeholder()

    for typ in (np.int8, np.uint8, np.int16, np.uint16):
        exporter = Exporter()
        exporter.add_graph_input('data',
                                 input,
                                 shape=[1, 2, 2, 2],
                                 elt_type=typ)
        exporter.add_graph_output('output', Cast(input, np.float32))
        md = exporter.export(f'{typ.__name__}_to_float')
        fname = f'cast{camel_name(typ)}ToFloat'
        with open(f'{fname}.onnxtxt', 'w') as f:
            f.write(str(md))
        onnx.save(md, f'{fname}.onnx')

def onehot_tests():
    indices = Placeholder(shape=[2,3,5])
    values = Constant(np.asarray([0, 1], dtype=np.float32))
    depth = Constant(np.asarray(4, dtype=np.int64))

    exporter = Exporter()
    exporter.add_graph_input('indices', indices)
    exporter.add_graph_output('a', OneHot(indices, depth, values))
    exporter.add_graph_output('a0', OneHot(indices, depth, values, 0))
    exporter.add_graph_output('a1', OneHot(indices, depth, values, 1))
    exporter.add_graph_output('a2', OneHot(indices, depth, values, 2))
    md = exporter.export('OneHot Test')
    fname = 'oneHot'
    with open(f'{fname}.onnxtxt', 'w') as f:
        f.write(str(md))
    onnx.save(md, f'{fname}.onnx')


def resize_test():
    X = Placeholder()
    Y = Resize(X,
               scales=Constant(np.asarray([1, 1, 2, 2], dtype=np.float32)), coordinate_transformation_mode="half_pixel",
               mode="nearest", nearest_mode="round_prefer_ceil")
    b = Exporter()
    b.add_graph_input('in', X, [1, 1, 2, 2])
    b.add_graph_output('Y', Y)
    md = b.export('Resize test')
    fname = 'resizeHalfPixelNearestCeil'
    with open(f'{fname}.onnxtxt', 'w') as f:
        f.write(str(md))
    onnx.save(md, f'{fname}.onnx')


def run():
    resize_test()
    # onehot_tests()
    # convert_tests()

if __name__ == "__main__":
    run()
