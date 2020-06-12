# Streaming Gradient Trees

Streaming Gradient Trees (SGT) to appear in skmultiflow

To do a basic test, just run `python test_sgt.py`

The folder structure deserves some refactoring and there are some "tricky" imports messing with the `sys.path`. These imports are going to be fixed when the implementation is added to skmultiflow.

### TODO:

- [ ] Add option to control manually the task type (?)
- [ ] Implement high-level predict_proba
- [ ] Benchmark classification and multi-label tasks (just basic tests were performed so far)
