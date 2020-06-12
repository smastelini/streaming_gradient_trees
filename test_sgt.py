import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from skmultiflow.data import RegressionGenerator, RandomRBFGenerator, MultilabelGenerator
from skmultiflow.trees import HoeffdingTreeRegressor, iSOUPTreeRegressor
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import MultiOutputLearner
from streaming_gradient_tree import StreamingGradientTree


""" Basic test of SGTs concerning their capabilities to handle different learning tasks. """


# Some regression metrics
def MAE(y, y_pred):
    return np.mean(np.abs(y - y_pred))


def aMAE(Y, Y_hat):
    return np.mean(np.mean(np.abs(Y - Y_hat), axis=0))


def aRMSE(Y, Y_hat):
    return np.mean(np.sqrt(np.mean((Y - Y_hat) ** 2, axis=0)))


def aRRMSE(Y, Y_hat):
    numerator = np.sum((Y - Y_hat) ** 2, axis=0) + 1e-6
    denominator = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0) + 1e-6

    return np.mean(np.sqrt(numerator / denominator))


RANDOM_SEED = 2020
N_SAMPLES = 5001
# Build SGT
sgt = StreamingGradientTree(quantization_strategy='stddiv3', mode='current')

###################################################################################################
#                                           STR                                                   #
###################################################################################################
print('\n-------------------------')
print('Single-target regression:')
stream = RegressionGenerator(n_samples=N_SAMPLES, n_features=10, n_informative=8, n_targets=1,
                             random_state=RANDOM_SEED)
sgt.reset()
htr = HoeffdingTreeRegressor(random_state=RANDOM_SEED, leaf_prediction='mean')

# Warm-start
X, y = stream.next_sample()
sgt.partial_fit(X[0], float(y[0]))
htr.partial_fit(X, y)

ys = []
ys_sgt = []
ys_htr = []

while stream.has_more_samples():
    X, y = stream.next_sample()
    ys.append(y[0])
    ys_sgt.append(sgt.predict(X[0])[0])
    ys_htr.append(htr.predict(X)[0])

    sgt.partial_fit(X[0], y[0])
    htr.partial_fit(X, y)

print('MAE SGT:', MAE(np.asarray(ys), np.asarray(ys_sgt)))
print('MAE HTR:', MAE(np.asarray(ys), np.asarray(ys_htr)))

print()
print('SGT:')
print('Number of Nodes:', sgt.n_nodes)
print('Node updates:', sgt.n_node_updates)
print('Number of splits:', sgt.n_splits)
print('Max depth:', sgt.max_depth)
print()

###################################################################################################
#                                           MTR                                                   #
###################################################################################################
print('\n------------------------')
print('Multi-target regression:')
stream = RegressionGenerator(n_samples=N_SAMPLES, n_features=10, n_informative=8, n_targets=5,
                             random_state=RANDOM_SEED)
sgt.reset()
isoup = iSOUPTreeRegressor(random_state=RANDOM_SEED, leaf_prediction='adaptive')
meta = MultiOutputLearner(base_estimator=HoeffdingTreeRegressor(
    random_state=RANDOM_SEED, leaf_prediction='mean'))


# Warm-start
X, y = stream.next_sample()
sgt.partial_fit(X[0], y[0])
isoup.partial_fit(X, y)
meta.partial_fit(X, y)


ys = []
ys_sgt = []
ys_isoup = []
ys_meta = []

while stream.has_more_samples():
    X, y = stream.next_sample()
    ys.append(y[0])
    ys_sgt.append(sgt.predict(X[0])[0])
    ys_isoup.append(isoup.predict(X)[0])
    ys_meta.append(meta.predict(X)[0])

    sgt.partial_fit(X[0], y[0])
    isoup.partial_fit(X, y)
    meta.partial_fit(X, y)

print('aMAE SGT:', aMAE(np.asarray(ys), np.asarray(ys_sgt)))
print('aMAE iSOUP-Tree:', aMAE(np.asarray(ys), np.asarray(ys_isoup)))
print('aMAE Meta-HTR:', aMAE(np.asarray(ys), np.asarray(ys_meta)))
print()
print('aRMSE SGT:', aRMSE(np.asarray(ys), np.asarray(ys_sgt)))
print('aRMSE iSOUP-Tree:', aRMSE(np.asarray(ys), np.asarray(ys_isoup)))
print('aRMSE Meta-HTR:', aRMSE(np.asarray(ys), np.asarray(ys_meta)))
print()
print('aRRMSE SGT:', aRRMSE(np.asarray(ys), np.asarray(ys_sgt)))
print('aRRMSE iSOUP-Tree:', aRRMSE(np.asarray(ys), np.asarray(ys_isoup)))
print('aRRMSE Meta-HTR:', aRRMSE(np.asarray(ys), np.asarray(ys_meta)))

print()
print('SGT:')
print('Number of Nodes:', sgt.n_nodes)
print('Node updates:', sgt.n_node_updates)
print('Number of splits:', sgt.n_splits)
print('Max depth:', sgt.max_depth)
print()

###################################################################################################
#                                   Binary Classification                                         #
###################################################################################################
print('\n----------------------')
print('Binary classification:')

stream = RandomRBFGenerator(model_random_state=RANDOM_SEED, sample_random_state=RANDOM_SEED)
sgt.reset()
ht = HoeffdingTreeClassifier(leaf_prediction='mc')


# Warm-start
X, y = stream.next_sample()
sgt.partial_fit(X[0], int(y[0]))
ht.partial_fit(X, y)


ys = []
ys_sgt = []
ys_ht = []

for i in range(N_SAMPLES):
    X, y = stream.next_sample()
    ys.append(y[0])
    ys_sgt.append(sgt.predict(X[0])[0])
    ys_ht.append(ht.predict(X)[0])

    sgt.partial_fit(X[0], int(y[0]))
    ht.partial_fit(X, y)

print('Acc SGT:', accuracy_score(np.asarray(ys), np.asarray(ys_sgt)))
print('Acc HT:', accuracy_score(np.asarray(ys), np.asarray(ys_ht)))

print()
print('F1 SGT:', f1_score(np.asarray(ys), np.asarray(ys_sgt)))
print('F1 HT:', f1_score(np.asarray(ys), np.asarray(ys_ht)))

print()
print('SGT:')
print('Number of Nodes:', sgt.n_nodes)
print('Node updates:', sgt.n_node_updates)
print('Number of splits:', sgt.n_splits)
print('Max depth:', sgt.max_depth)
print()


###################################################################################################
#                                   Multi-class Classification                                    #
###################################################################################################
print('\n---------------------------')
print('Multi-class classification:')

stream = RandomRBFGenerator(model_random_state=RANDOM_SEED, sample_random_state=RANDOM_SEED,
                            n_classes=3)
sgt.reset()
ht = HoeffdingTreeClassifier(leaf_prediction='mc')


# Warm-start
X, y = stream.next_sample()
sgt.partial_fit(X[0], int(y[0]))
ht.partial_fit(X, y)


ys = []
ys_sgt = []
ys_ht = []

for i in range(N_SAMPLES):
    X, y = stream.next_sample()
    ys.append(y[0])
    ys_sgt.append(sgt.predict(X[0])[0])
    ys_ht.append(ht.predict(X)[0])

    sgt.partial_fit(X[0], int(y[0]))
    ht.partial_fit(X, y)

print('Acc SGT:', accuracy_score(np.asarray(ys), np.asarray(ys_sgt)))
print('Acc HT:', accuracy_score(np.asarray(ys), np.asarray(ys_ht)))

print()
print('F1 SGT:', f1_score(np.asarray(ys), np.asarray(ys_sgt), average='weighted'))
print('F1 HT:', f1_score(np.asarray(ys), np.asarray(ys_ht), average='weighted'))

print()
print('SGT:')
print('Number of Nodes:', sgt.n_nodes)
print('Node updates:', sgt.n_node_updates)
print('Number of splits:', sgt.n_splits)
print('Max depth:', sgt.max_depth)
print()

###################################################################################################
#                                   Multi-class Classification                                    #
###################################################################################################
print('\n---------------------------')
print('Multi-label classification:')

stream = MultilabelGenerator(n_samples=N_SAMPLES, n_features=10, n_targets=5, n_labels=2,
                             random_state=RANDOM_SEED)
sgt.reset()
meta = MultiOutputLearner(base_estimator=HoeffdingTreeClassifier(leaf_prediction='mc'))


# Warm-start
X, y = stream.next_sample()
sgt.partial_fit(X[0], y[0].astype('int'))
meta.partial_fit(X, y)

ys = []
ys_sgt = []
ys_meta = []

while stream.has_more_samples():
    X, y = stream.next_sample()
    if len(y) == 0:
        break

    ys.append(y[0])
    ys_sgt.append(sgt.predict(X[0])[0])
    ys_meta.append(meta.predict(X)[0])

    sgt.partial_fit(X[0], y[0])
    meta.partial_fit(X, y)

print('Acc SGT:', accuracy_score(np.asarray(ys), np.asarray(ys_sgt)))
print('Acc Meta-HT:', accuracy_score(np.asarray(ys), np.asarray(ys_meta)))

print()
print('F1 SGT:', f1_score(np.asarray(ys), np.asarray(ys_sgt), average='weighted'))
print('F1 Meta-HT:', f1_score(np.asarray(ys), np.asarray(ys_meta), average='weighted'))

print()
print('SGT:')
print('Number of Nodes:', sgt.n_nodes)
print('Node updates:', sgt.n_node_updates)
print('Number of splits:', sgt.n_splits)
print('Max depth:', sgt.max_depth)
print()
