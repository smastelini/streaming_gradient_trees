from river import synth
from river import metrics

from streaming_gradient_tree import StreamingGradientTreeRegressor

metric = metrics.MAE()

dataset = iter(synth.Friedman.take(5000))

tree = StreamingGradientTreeRegressor()

for x, y in dataset:
    metric.update(y, tree.predict_one(x))

    tree.learn_one(x, y)

print(metric)