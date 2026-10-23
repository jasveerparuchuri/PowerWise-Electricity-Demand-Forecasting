# PowerWise: Deep Learning Based Electricity Demand Forecasting

PowerWise is a small educational project for learning artificial neural networks by implementing the core calculations with NumPy. Its main task is **regression**: estimate an hourly electricity demand value from past demand and available context. The repository was empty when this project was started, so there was no prior model, dataset, or architecture to retain.

## Dataset and honest results

No measured dataset was present in the GitHub repository. The runnable demo therefore creates a deterministic, illustrative synthetic hourly series in `powerwise/data.py` (demand in arbitrary kWh-like units with daily/weekly patterns, temperature, and noise). It is useful to exercise the code, but it is **not real utility data** and its metrics must not be represented as real-world results. Replace it with a documented public dataset before drawing conclusions about actual electricity demand. Input rows need `timestamp`, `demand`, and `temperature`; features are derived using only the previous observed demand and calendar/weather columns.

## Architecture

`timestamp`/demand/weather → chronological 70/15/15 split → standardizer fitted only on training features → vectorized NumPy MLP (ReLU hidden layer, linear output) → test metrics and plots. A persistence baseline predicts the previous hour's demand. The model's `W1`, `b1`, `W2`, and `b2` are parameters. Learning rate, epochs, hidden size, batch size, and seed are hyperparameters.

The MLP computes `Z1 = X @ W1 + b1`, `A1 = ReLU(Z1)`, and `prediction = A1 @ W2 + b2`. MSE gradients are derived manually and propagated through ReLU with the chain rule; mini-batches are shuffled per epoch and a final partial batch is retained. The linear output is appropriate for continuous regression. A sigmoid is shown separately for binary classification and its derivative is `a * (1-a)`.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
python -m powerwise.education
python -m powerwise.forecast --epochs 300 --seed 7
pytest
```

The forecast command writes `outputs/metrics.txt`, `loss.png`, `predictions.png`, and `errors.png`. Metrics are calculated on the final chronological 15% holdout, in original demand units. MAE is mean absolute error, RMSE penalizes large errors more, and R² compares residual error with predicting the test mean. The baseline and network are reported side by side; no superiority is assumed.

## Learning notes

An artificial neuron is a simplified mathematical model inspired by (but not equivalent to) a biological neuron: numeric inputs are multiplied by trainable weights, summed with a bias, then passed through an activation. For example, inputs `[2, 1]`, weights `[1.5, -1]`, bias `0.5` give `z = 2*1.5 + 1*(-1) + 0.5 = 2.5`; ReLU outputs `2.5`.

A single perceptron can learn linearly separable AND/OR, but cannot learn XOR because XOR classes cannot be separated by one straight decision boundary. Binary classification has two outcomes; multiclass classification selects among several output scores (the educational example uses `argmax`). Those are separate from demand forecasting regression.

Activation examples are provided for step, sigmoid, ReLU, tanh, and linear functions. Sigmoid is evaluated stably for large positive/negative values. A grayscale 8×8 image has 64 pixels; flattening its matrix produces a 64-value input vector. This is a shape concept only, not an image classifier.

For an illustrative six-feature network with eight hidden neurons and one output: `X (batch,6) @ W1 (6,8) + b1 (1,8) → A1 (batch,8) @ W2 (8,1) + b2 (1,1) → output (batch,1)`. Matrix inner dimensions must match so each output neuron has one weighted sum per example.

Training changes weights and biases to reduce MSE. Gradient descent updates `W_new = W_old - learning_rate * dLoss/dW`; very small learning rates progress slowly, while very large values can overshoot. Stochastic gradient descent updates one example at a time, batch gradient descent uses the entire set for one update, and mini-batch training updates on chunks. Smaller chunks usually mean more updates per epoch. A fixed seed makes the shuffle and initialization reproducible.

## Syllabus coverage

| Topic | Demonstration |
|---|---|
| Biological/artificial neuron, inputs, weights, bias, activation, numeric example | README “Learning notes”; `powerwise/education.py` |
| Perceptron formula, AND/OR learning, update, XOR limitation | `powerwise/perceptron.py`; README “Learning notes” |
| Binary vs multiclass classification, output scores | `powerwise/education.py` |
| Inputs, layers, regression vs classification, assumptions | `powerwise/network.py`; README “Architecture” |
| Parameters and hyperparameters | README “Architecture” |
| Step, sigmoid, ReLU, tanh, linear, stable sigmoid | `powerwise/activations.py`; `powerwise/education.py` |
| Layer flow and matrix multiplication | `powerwise/network.py`; README shape example |
| Image pixels, dimensions, flattened inputs | `powerwise/education.py` |
| Matrix dimensions and vectorized batches | `powerwise/network.py`; README shape example |
| MSE, loss history, gradient descent, chain rule, backprop | `powerwise/network.py` |
| Sigmoid derivative and weight update | `powerwise/network.py:sigmoid_neuron_step` |
| SGD/batch/mini-batch concepts, shuffle, partial batch, seed | `powerwise/network.py`; README “Learning notes” |
| Chronological split, training-only scaling, persistence baseline, MAE/RMSE/R² | `powerwise/data.py`; `powerwise/forecast.py` |
| Loss, prediction and error plots | `powerwise/forecast.py` |
| Shape, loss, gradient-check, training, reproducibility, leakage checks | `tests/test_powerwise.py` |

## Viva questions

- **What is a perceptron?** A linear weighted sum plus bias followed by a threshold activation.
- **Why can’t a single perceptron learn XOR?** XOR is not linearly separable.
- **Why ReLU in the hidden layer?** It is simple, nonlinear, and has a straightforward derivative.
- **Why linear output for demand?** Demand is a continuous numeric value, not a class probability.
- **What are weights and biases?** Trainable parameters controlling each neuron's weighted sum and offset.
- **What does feedforward do?** Computes layer outputs from inputs toward the prediction.
- **What is backpropagation?** Applies the chain rule from loss back through the layers to calculate parameter gradients.
- **Why use mini-batches?** They provide more frequent updates than full-batch training while averaging gradients over multiple examples.
- **Why split time chronologically?** A forecast should not train on observations from after its evaluation period.

## Limitations and next steps

The generated series is synthetic; forecast features assume weather for the target hour is known. The small one-hidden-layer MLP is a teaching tool and does not model complex seasonality as well as a carefully validated operational system. For a real study, add a cited public load dataset, document units and missing values, confirm forecast-time weather availability, tune hyperparameters on validation only, and reserve the latest period for one final test.
