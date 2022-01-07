import onnx
import numpy as np
from builder import *


def make_lstm_cell(x_in, h_in, c_in, W, R, B=None):
    Wi, Wo, Wf, Wc = W
    Ri, Ro, Rf, Rc = R
    Bi, Bo, Bf, Bc = B or None, None, None, None

    i = MatMul(Wi, x_in)+MatMul(Ri, h_in)
    if Bi:
        i = i + Bi
    i = Sigmoid(i)

    o = MatMul(Wo, x_in) + MatMul(Ro, h_in)
    if Bo:
        o = o + Bo
    o = Sigmoid(o)

    f = MatMul(Wf, x_in) + MatMul(Rf, h_in)
    if Bf:
        f = f + Bf
    f = Sigmoid(f)

    c = MatMul(Wc, x_in) + MatMul(Rc, h_in)
    if Bc:
        c = c + BatchNormalization
    c = Tanh(c)
    c_out = f*c_in + i*c
    h_out = o*Tanh(c_out)

    return h_out, c_out


def lstm_cell():
    x_in = Placeholder()
    h_in = Placeholder()
    c_in = Placeholder()
    Wi = Placeholder()
    Wo = Placeholder()
    Wf = Placeholder()
    Wc = Placeholder()
    Ri = Placeholder()
    Ro = Placeholder()
    Rf = Placeholder()
    Rc = Placeholder()

    h_out, c_out = make_lstm_cell(
        x_in, h_in, c_in, (Wi, Wo, Wf, Wc), (Ri, Ro, Rf, Rc))

    exporter = Exporter()
    N = 4  # Batch size
    C = 32  # Input word size
    H = 16  # Hidden size

    exporter.add_graph_input('x_in', x_in, [N, C])
    exporter.add_graph_input('h_in', h_in, [N, H])
    exporter.add_graph_input('c_in', c_in, [N, H])
    exporter.add_graph_input('Wi', Wi, [H, C])
    exporter.add_graph_input('Wo', Wo, [H, C])
    exporter.add_graph_input('Wf', Wf, [H, C])
    exporter.add_graph_input('Wc', Wc, [H, C])
    exporter.add_graph_input('Ri', Ri, [H, H])
    exporter.add_graph_input('Ro', Ro, [H, H])
    exporter.add_graph_input('Rf', Rf, [H, H])
    exporter.add_graph_input('Rc', Rc, [H, H])
    exporter.add_graph_output('h_out', h_out)
    exporter.add_graph_output('c_out', c_out)

    md = exporter.export('LSTM cell')
    onnx.checker.check_model(md)
    onnx.save(md, 'lstm_cell.onnx')


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
    lstm_cell()
    return
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
