# ONNX Builder
Easy generation of ONNX files.

ONNX ops are treated as a DSL embedded within Python. Graphs can be created and then converted to ONNX IR where they can be exported as ONNX files.

```python
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
```

## Installation
```
pip install -r requirements.txt
# For Visual Studio Code support
pip install -r vsc_requirements.txt
```
