import onnx
import numpy as np
from builder import *


def a_plus_b():
    # Create two sources of data. For now ONNX graph
    # inputs need to be Placeholder
    a = Placeholder()
    b = Placeholder()

    # The exporter will convert a graph to ONNX
    exporter = Exporter()
    # Add two inputs
    exporter.add_graph_input('a', a, [32, 32])
    exporter.add_graph_input('b', b, [32, 32])
    # Add one output
    exporter.add_graph_output('output', Abs(a)+b)
    # Export as ONNX
    md = exporter.export('a plus b')
    onnx.checker.check_model(md)
    onnx.save(md, 'a_plus_b.onnx')


def run():
    a_plus_b()
    N = 4
    T = 140
    X = Placeholder()
    mean = Placeholder()
    var = Placeholder()
    BX = BatchNormalization(X,
                            Constant([1.0, 1.0, 1.0], dtype=np.float32),
                            Constant([0.0, 0.0, 0.0], dtype=np.float32),
                            mean, var)
    vpad = Pad(BX, Constant([0, 2, 0, 0], dtype=np.int64))
    sum = Pad(vpad + Abs(vpad), Constant([3, 3, 3, 3], dtype=np.int64))

    S = Placeholder()
    lens = Placeholder(np.int32)
    CW = Constant(np.ones([1, 64, 128], dtype=np.float32))
    CR = Constant(np.ones([1, 64, 16], dtype=np.float32))
    CB = Constant(np.zeros([1, 128], dtype=np.float32))
    L = LSTM(S, CW, CR, CB, lens)

    b = Exporter()
    b.add_graph_input('input', X, [N, 32, 32, 3])
    b.add_graph_input('mean', mean, [32, 32, 3])
    b.add_graph_input('var', var, [32, 32, 3])
    b.add_graph_input('sentences', S, [N, T, 128])
    b.add_graph_input('seqlen', lens, [N, T])
    b.add_graph_output('output', sum)
    b.add_graph_output('running_mean', BX.running_mean)
    b.add_graph_output('Y_h', L.Y_h)
    md = b.export('test-model')
    print(f'The graph in model:\n{md.graph}')
    onnx.checker.check_model(md)
    print("Checked")
    onnx.save(md, 'test-model.onnx')
    return


if __name__ == "__main__":
    run()
