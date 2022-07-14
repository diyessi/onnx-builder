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
    for typ in (np.int8, np.uint8, np.int16, np.uint16):
        exporter = Exporter()
        input = Input(exporter, 'data',
                                 typ,
                                 shape=[1, 2, 2, 2])
        exporter.add_graph_output('output', Cast(input, np.float32))
        md = exporter.export(f'{typ.__name__}_to_float')
        fname = f'cast{camel_name(typ)}ToFloat'
        with open(f'{fname}.onnxtxt', 'w') as f:
            f.write(str(md))
        onnx.save(md, f'{fname}.onnx')

def onehot_tests():
    exporter = Exporter()
    values = np.asarray([0, 1], dtype=np.float32)
    depth = np.asarray(4, dtype=np.int64)

    indices = Input(exporter, 'indices', np.int64, [2, 3, 5])
    exporter.add_graph_output('a', OneHot(indices, depth, values), np.float32)
    exporter.add_graph_output('a0', OneHot(indices, depth, values, 0), np.float32)
    exporter.add_graph_output('a1', OneHot(indices, depth, values, 1), np.float32)
    exporter.add_graph_output('a2', OneHot(indices, depth, values, 2), np.float32)
    md = exporter.export('OneHot Test')
    fname = 'oneHot'
    with open(f'{fname}.onnxtxt', 'w') as f:
        f.write(str(md))
    onnx.save(md, f'{fname}.onnx')


def add_test():
    b = Exporter()
    X = Input(b, 'x', np.float32, [4])
    Y = Input(b, 'y', np.float32, [4])
    b.add_graph_output('Z', X+Y, np.float32)
    md = b.export('Add test')
    fname = 'add'
    onnx.save(md, f'{fname}.onnx')


def resize_test():
    b = Exporter()
    X = Input(b, 'in', np.float32, [1, 1, 2, 2])
    Y = Resize(X,
               scales=np.asarray([1, 1, 2, 2], dtype=np.float32), coordinate_transformation_mode="half_pixel",
               mode="nearest", nearest_mode="round_prefer_ceil")
    b.add_graph_output('Y', Y, np.float32)
    md = b.export('Resize test')
    fname = 'resizeHalfPixelNearestCeil'
    with open(f'{fname}.onnxtxt', 'w') as f:
        f.write(str(md))
    onnx.save(md, f'{fname}.onnx')


def run():
    add_test()
    # resize_test()
    # onehot_tests()
    # convert_tests()

if __name__ == "__main__":
    run()
