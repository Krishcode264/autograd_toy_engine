import math

class Value:
    def __init__(self, a, _child=(), _op="", label="") -> None:
        self.data = a
        self.child = set(_child)
        self.op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")

        def backward():
            self.grad += 1.0 * out.grad
            other.grad -= 1.0 * out.grad

        out._backward = backward
        return out

    def __rsub__(self, other):
        return Value(other) - self

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(
            (math.exp(self.data) - math.exp(-self.data))
            / (math.exp(self.data) + math.exp(-self.data)),
            [self],
            "tanh",
        )

        def backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.child:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
