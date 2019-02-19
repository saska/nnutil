# nnutil
Tool for creating numpy neural nets

A little module for quickly and simply creating numpy neural nets. Found useful for demo purposes where e.g. TensorFlow might break when switching systems. 

Quick training example:

```python
df = pd.read_csv('example.csv')
costs = []
it = minibatch_gen_from_pddf(df, "Y (Target Label)", 1024)
for X, Y in it:
    if not costs:
        layers = [[16, "relu"], [8,"relu"], [1, "sigmoid"]]
        nn = NN(layers=layers, data=X.T, labels=Y, learning_rate=0.002)
    else:
        nn.data = X.T
        nn.labels = Y
    costs.append(nn.train(10000)[0])
```

After training classification can be done simply by calling model_forward of the instance, passing new data.
