from river import synth
from river import metrics

from sgt import StreamingGradientTreeRegressor

metric = metrics.MAE()

dataset = iter(synth.Friedman(seed=42).take(5000))

tree = StreamingGradientTreeRegressor()

for x, y in dataset:
    metric.update(y, tree.predict_one(x))

    tree.learn_one(x, y)

print(metric)