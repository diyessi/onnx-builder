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


def run():
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


if __name__ == "__main__":
    run()
