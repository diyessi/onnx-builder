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
    N = 4  # Batch size
    C = 32  # Input word size
    H = 16  # Hidden size

    exporter = Exporter()
    x_in = Input(exporter, 'x_in', np.float32, [N, C])
    h_in = Input(exporter,  'h_in', np.float32, [N, H])
    c_in = Input(exporter, 'c_in', np.float32, [N, H])
    Wi = Input(exporter, 'Wi', np.float32, [H, C])
    Wo = Input(exporter, 'Wo', np.float32, [H, C])
    Wf = Input(exporter, 'Wf', np.float32, [H, C])
    Wc = Input(exporter, 'Wc', np.float32, [H, C])
    Ri = Input(exporter, 'Ri', np.float32, [H, H])
    Ro = Input(exporter, 'Ro', np.float32, [H, H])
    Rf = Input(exporter, 'Rf', np.float32, [H, H])
    Rc = Input(exporter, 'Rc', np.float32, [H, H])

    h_out, c_out = make_lstm_cell(
        x_in, h_in, c_in, (Wi, Wo, Wf, Wc), (Ri, Ro, Rf, Rc))

    exporter.add_graph_output('h_out', h_out, np.float32)
    exporter.add_graph_output('c_out', c_out, np.float32)

    md = exporter.export('LSTM cell')
    onnx.checker.check_model(md)
    onnx.save(md, 'lstm_cell.onnx')


def a_plus_b():

    # The exporter will convert a graph to ONNX
    exporter = Exporter()
    # Add two inputs
    a = Input(exporter, 'a', np.float32, [32, 32])
    b = Input(exporter, 'b', np.float32, [32, 32])
    # Add one output
    exporter.add_graph_output('output', Abs(a)+b, np.float32)
    # Export as ONNX
    md = exporter.export('a plus b')
    onnx.checker.check_model(md)
    onnx.save(md, 'a_plus_b.onnx')

def test_concat():
    exporter = Exporter()
    # Add two inputs
    a = Input(exporter, 'a', np.float32, [32, 32])
    b = Input(exporter, 'b', np.float32, [32, 32])
    # Add one output
    exporter.add_graph_output('output', Concat(a, b, a, a, b, axis=1), np.float32)
    # Export as ONNX
    md = exporter.export('concat')
    onnx.checker.check_model(md)
    onnx.save(md, 'concat.onnx')


def run():
    a_plus_b()
    test_concat()
    lstm_cell()

    b = Exporter()

    N = 4
    T = 140
    X = Input(b, 'input', np.float32, [N, 32, 32, 3])
    mean = Input(b, 'mean', np.float32, [32, 32, 3])
    var = Input(b, 'var',np.float32, [32, 32, 3])

    BX = BatchNormalization(X,
                            np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                            mean, var)
    vpad = Pad(BX, np.asarray([0, 2, 0, 0], dtype=np.int64))
    sum = Pad(vpad + Abs(vpad), np.asarray([3, 3, 3, 3], dtype=np.int64))

    S = Input(b, 'sentences', np.float32, [N, T, 128])
    lens = Input(b, 'seqlen', np.int32, [N, T])
    CW = np.ones([1, 64, 128], dtype=np.float32)
    CR = np.ones([1, 64, 16], dtype=np.float32)
    CB = np.zeros([1, 128], dtype=np.float32)
    L = LSTM(S, CW, CR, CB, lens)

    b.add_graph_output('output', sum, np.float32)
    b.add_graph_output('running_mean', BX.running_mean, np.float32)
    b.add_graph_output('Y_h', L.Y_h, np.float32)
    md = b.export('test-model')
    print(f'The graph in model:\n{md.graph}')
    onnx.checker.check_model(md)
    print("Checked")
    onnx.save(md, 'test-model.onnx')
    return


if __name__ == "__main__":
    run()
