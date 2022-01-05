import onnx
import numpy as np
from builder import *


def a_plus_b():
    # Create two sources of data. For now ONNX graph
    # inputs need to be Placeholder
    a = Placeholder(onnx.TensorProto.FLOAT, [32, 32])
    b = Placeholder(onnx.TensorProto.FLOAT, [32, 32])

    # The builder will convert a graph to ONNX
    builder = Builder('a plus b')
    # build needs a list of inputs and a list of outputs.
    builder.build([builder.named('a', a),  # Export first input as 'a'
                   builder.named('b', b)],  # Export second input as 'b'
                  # A single output exported as 'output'
                  # We can inline this simple graph construction
                  [builder.named('output', Abs(a)+b)])
    onnx.save(builder.model_def, 'a_plus_b.onnx')


def run():
    a_plus_b()
    N = 4
    T = 140
    X = Placeholder(onnx.TensorProto.FLOAT, [N, 32, 32, 3])
    mean = Placeholder(onnx.TensorProto.FLOAT, [32, 32, 3])
    var = Placeholder(onnx.TensorProto.FLOAT, [32, 32, 3])
    BX = BatchNormalization(X,
                            Constant([1.0, 1.0, 1.0], dtype=np.float32),
                            Constant([0.0, 0.0, 0.0], dtype=np.float32),
                            mean, var)
    vpad = Pad(BX, Constant([0, 2, 0, 0], dtype=np.int64))
    sum = Pad(vpad + Abs(vpad), Constant([3, 3, 3, 3], dtype=np.int64))

    S = Placeholder(onnx.TensorProto.FLOAT, [N, T, 128])
    lens = Placeholder(onnx.TensorProto.INT32, [N, T])
    CW = Constant(np.ones([1, 64, 128], dtype=np.float32))
    CR = Constant(np.ones([1, 64, 16], dtype=np.float32))
    CB = Constant(np.zeros([1, 128], dtype=np.float32))
    L = LSTM(S, CW, CR, CB, lens)

    b = Builder('test-model')
    b.build([b.named('input', X),
             b.named('mean', mean),
             b.named('var', var),
             b.named('sentences', S),
             b.named('seqlen', lens)],
            [b.named('output', sum),
             b.named('running_mean', BX.running_mean),
             b.named('running_var', BX.running_var),
             b.named('Y_h', L.Y_h)])
    md = b.model_def
    print(f'The graph in model:\n{md.graph}')
    onnx.checker.check_model(md)
    print("Checked")
    onnx.save(md, 'test-model.onnx')
    return


if __name__ == "__main__":
    run()
