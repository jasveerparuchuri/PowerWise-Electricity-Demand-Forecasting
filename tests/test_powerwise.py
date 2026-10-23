import numpy as np

from powerwise.activations import relu, sigmoid, tanh
from powerwise.data import Standardizer, chronological_split, make_features, make_synthetic_demand
from powerwise.network import NumpyMLP, mse
from powerwise.perceptron import Perceptron


def test_activations():
    np.testing.assert_allclose(relu([-1, 0, 2]), [0, 0, 2])
    np.testing.assert_allclose(sigmoid([0]), [.5])
    np.testing.assert_allclose(tanh([0]), [0])


def test_perceptron_and():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert np.array_equal(Perceptron().fit(X, [0, 0, 0, 1]).predict(X), [0, 0, 0, 1])


def test_feedforward_shapes_and_mse():
    model = NumpyMLP(3, 5, seed=1)
    pred, _ = model.forward(np.ones((4, 3)))
    assert pred.shape == (4, 1)
    assert mse(np.array([1, 2]), np.array([1, 4])) == 2.0


def test_finite_difference_gradient():
    model = NumpyMLP(2, 3, seed=2)
    X, y = np.array([[.2, -.3], [.5, .7]]), np.array([.4, -.2])
    analytic = model.gradients(X, y)[2][0, 0]
    old, eps = model.W2[0, 0], 1e-6
    model.W2[0, 0] = old + eps; plus = mse(y, model.forward(X)[0])
    model.W2[0, 0] = old - eps; minus = mse(y, model.forward(X)[0])
    model.W2[0, 0] = old
    assert abs(analytic - (plus - minus) / (2 * eps)) < 1e-5


def test_training_reproducible_and_reduces_loss_with_partial_batch():
    X = np.linspace(-1, 1, 41).reshape(-1, 1); y = 3 * X[:, 0] + 1
    a, b = NumpyMLP(1, 8, 5), NumpyMLP(1, 8, 5)
    ha = a.fit(X, y, epochs=150, learning_rate=.03, batch_size=8, seed=9)
    hb = b.fit(X, y, epochs=150, learning_rate=.03, batch_size=8, seed=9)
    assert ha.train_loss[-1] < ha.train_loss[0] * .1
    assert ha.train_loss == hb.train_loss


def test_time_split_and_training_only_scaling():
    X, y = make_features(make_synthetic_demand(240))
    xt, yt, xv, yv, xe, ye = chronological_split(X, y)
    assert len(xt) + len(xv) + len(xe) == len(X)
    assert xt.index.max() < xv.index.min() < xe.index.min()
    scaler = Standardizer().fit(xt.to_numpy())
    assert np.allclose(scaler.mean_, xt.to_numpy().mean(axis=0))
    assert len(yt) + len(yv) + len(ye) == len(y)
