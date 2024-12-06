import haiku as hk

class DummyModel(hk.Module):
    def __init__(self, name: str = "SimpleRNN", **kwargs):
        super().__init__(name)
        self.kwargs = kwargs

    def __call__(self, x):
        return x

model = DummyModel