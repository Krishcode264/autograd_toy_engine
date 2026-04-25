import random
from value import Value

class Neuron:
    def __init__(self, inputs):
        # w * x + b
        self.weights = [Value(random.uniform(-1, +1)) for i in range(inputs)]
        self.bias = Value(random.uniform(-1, +1))

    def __call__(self, x):
        act = sum((w * x for w, x in zip(self.weights, x)), self.bias)
        return act.tanh()

    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, no_of_neurons, input_per_neurons):
        self.neurons = [Neuron(input_per_neurons) for i in range(no_of_neurons)]

    def __call__(self, list_of_inputs):
        outs = [neuron(list_of_inputs) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for n in self.neurons:
            for p in n.parameters():
                params.append(p)
        return params

class MLP:
    # 3 inputs => layer of 4 neurons => layer of 4 neurons => one neron layer => output
    def __init__(self, inputs, list_of_neurons_in_each_layer):
        sz = [inputs] + list_of_neurons_in_each_layer  # [3, 4, 4, 1]
        self.layers = [Layer(sz[i], sz[i - 1]) for i in range(1, len(sz))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
