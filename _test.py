from river import synth
from river import metrics

from sgt import StreamingGradientTreeRegressor

metric = metrics.MAE()

dataset = iter(synth.Friedman(seed=42).take(1000))

tree = StreamingGradientTreeRegressor(delta=0.1)

for x, y in dataset:
    metric.update(y, tree.predict_one(x))

    tree.learn_one(x, y)

print(metric)
print('Tree depth:', tree.depth)
print('Number of nodes:', tree.n_nodes)

# from river import datasets
# from river import evaluate
# from river import linear_model
# from river import metrics
# from river import optim
# from river import preprocessing
#
# from sgt import StreamingGradientTreeClassifier
#
# dataset = datasets.Phishing()
# model = (
#     preprocessing.StandardScaler() |
#     # linear_model.LogisticRegression(optimizer=optim.SGD(.1))
#     StreamingGradientTreeClassifier(delta=0.1)
# )
# metric = metrics.Accuracy()
# print(evaluate.progressive_val_score(dataset, model, metric))
# # Accuracy: 88.96%