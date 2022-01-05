# ONNX Builder
Easy generation of ONNX files.

ONNX ops are treated as a DSL embedded within Python. Graphs can be created and then converted to ONNX IR where they can be exported as ONNX files.

```python
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
```

## Installation
```
pip install -r requirements.txt
# For Visual Studio Code support
pip install -r vsc_requirements.txt
```
