import onnx
from onnx import TensorProto
from builder.builder import Builder
from builder import *
import numpy as np


def run():
    N = 4
    T = 140
    X = Placeholder(TensorProto.FLOAT, [N, 32, 32, 3])
    mean = Placeholder(TensorProto.FLOAT, [32, 32, 3])
    var = Placeholder(TensorProto.FLOAT, [32, 32, 3])
    BX = BatchNormalization(X,
                            Constant([1.0, 1.0, 1.0], dtype=np.float32),
                            Constant([0.0, 0.0, 0.0], dtype=np.float32),
                            mean, var)
    vpad = Pad(BX, Constant([0, 2, 0, 0], dtype=np.int64))
    sum = Pad(vpad + Abs(vpad), Constant([3, 3, 3, 3], dtype=np.int64))

    S = Placeholder(TensorProto.FLOAT, [N, T, 128])
    lens = Placeholder(TensorProto.INT32, [N, T])
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
