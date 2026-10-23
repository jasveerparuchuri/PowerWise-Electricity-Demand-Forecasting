# PowerWise Electricity Forecasting Lab

PowerWise is an educational electricity demand forecasting project organized as a small, runnable Python application. It follows a modular research-project layout with `src/`, `tests/`, `data/`, `models/`, `reports/`, a notebook, command-line training, and an optional Streamlit dashboard. Its primary neural network uses NumPy for feedforward calculations and manual backpropagation.

## Dataset status

The original GitHub repository was empty and contained no dataset or forecasting code. The default demonstration therefore generates a deterministic **synthetic** hourly dataset. It is not measured utility data, and its metrics are not evidence of real-world forecasting skill. You can instead provide a CSV with `timestamp`, `demand`, and `temperature` columns. Demand and temperature should use consistent numeric units, timestamps should be hourly and sortable, and the target-hour temperature must be information available at forecast time. The first row is dropped because its previous-hour demand is unavailable.

## Architecture

```text
data (CSV or synthetic) → validation and causal features → chronological 70/15/15 split
  → standardize using training rows only → NumPy MLP (ReLU hidden, linear output)
  → test metrics vs persistence baseline → reports and dashboard
```

The forecast is a **regression** problem: predict a continuous demand value. The six input features are previous-hour demand, cyclical hour and weekday encodings, and temperature. The persistence baseline predicts previous-hour demand. The model computes `Z1 = X @ W1 + b1`, `A1 = ReLU(Z1)`, and `prediction = A1 @ W2 + b2`. Backpropagation, MSE gradients, parameter updates, and mini-batch training are implemented directly in `src/network.py`.

## Project structure

```text
PowerWise-Electricity-Demand-Forecasting/
├── app.py                 # optional Streamlit training/results dashboard
├── train.py               # command-line training and evaluation
├── data/
│   └── cache/              # reserved for local input/cache files
├── models/                 # reserved for saved model artifacts
├── notebooks/
│   └── 01_forecasting_and_neural_networks.ipynb
├── reports/                # generated reports/latest/ is git-ignored
├── src/
│   ├── activations.py      # step, sigmoid, ReLU, tanh, linear
│   ├── config.py           # default hyperparameters and split fractions
│   ├── data.py             # synthetic series, features, split, scaling, metrics
│   ├── data_collection.py  # CSV loading and validation
│   ├── education.py        # perceptron, logic gates, image and sigmoid demos
│   ├── forecast.py         # end-to-end training, evaluation, and plots
│   ├── model.py            # public model API
│   ├── network.py          # vectorized NumPy MLP and manual gradients
│   ├── perceptron.py       # binary perceptron from scratch
│   ├── pipeline.py         # run orchestration
│   └── visualization.py    # plot helpers
├── tests/                  # pytest suite
├── pytest.ini
├── requirements.txt
└── .github/workflows/ci.yml
```

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.education
python train.py --epochs 300 --seed 7
pytest -q
```

To forecast from a CSV:

```bash
python train.py --csv data/my_hourly_demand.csv --epochs 300 --seed 7
```

The CSV needs these headers: `timestamp,demand,temperature`. The CLI writes `reports/latest/metrics.txt`, `loss.png`, `predictions.png`, and `errors.png`. It prints MAE, RMSE, and R² for the final chronological test segment for both the MLP and persistence baseline. The default synthetic run has no external data dependency.

Run the interactive dashboard with:

```bash
streamlit run app.py
```

The dashboard is an optional front end to the same educational pipeline; it does not deploy a service or connect to a utility provider.

## Neural network lessons

An artificial neuron is a simplified mathematical model inspired by the brain. For inputs `[2, 1]`, weights `[1.5, -1]`, and bias `0.5`, the weighted sum is `z = 2(1.5) + 1(-1) + 0.5 = 2.5`; ReLU returns `2.5`. Weights and biases are learned parameters. Learning rate, epochs, hidden width, mini-batch size, and random seed are hyperparameters.

The network uses matrix operations across a batch. With six features, eight hidden units, and one output, `X` is `(batch, 6)`, `W1` is `(6, 8)`, `b1` is `(1, 8)`, `W2` is `(8, 1)`, and `b2` is `(1, 1)`. Matching inner dimensions allow each layer to compute one weighted sum per neuron and example.

The perceptron learns AND and OR using a threshold. It cannot learn XOR because XOR is not linearly separable. The educational examples also show binary and multiclass classification, the step/sigmoid/ReLU/tanh/linear activations, a sigmoid derivative update, and how an 8×8 grayscale image becomes 64 input values. These are separate from the demand regression problem.

Training minimizes mean squared error. Mini-batch training shuffles only the training period at each epoch, uses the final incomplete batch, and records training and validation loss. Splits remain chronological, and the standardizer learns its mean and scale from training features only. Test metrics are reported in the original demand units. Do not claim the neural network is better than the baseline unless the measured test results support that conclusion.

## Syllabus coverage

| Syllabus topic | Project location |
|---|---|
| Artificial neuron, brain analogy, weighted sum and numerical example | `src/education.py`, this README |
| Perceptron learning, AND/OR, XOR limitation | `src/perceptron.py`, `src/education.py` |
| Binary and multiclass classification | `src/education.py` |
| Inputs, layers, regression vs classification, simplifying assumptions | `src/network.py`, this README |
| Parameters and hyperparameters | `src/config.py`, this README |
| Step, sigmoid, ReLU, tanh, linear and stable sigmoid | `src/activations.py` |
| Layer-to-layer information flow and matrix dimensions | `src/network.py`, notebook |
| Image pixels and flattened feature counts | `src/education.py`, notebook |
| Feedforward and vectorized calculations | `src/network.py` |
| MSE, gradient descent and manual backpropagation | `src/network.py` |
| Sigmoid derivative and chain-rule weight update | `src/network.py:sigmoid_neuron_step` |
| SGD, batch and mini-batch concepts; shuffle, partial batch and seed | `src/network.py`, notebook |
| Time split, training-only scaling, leakage-aware features | `src/data.py`, `src/forecast.py` |
| Persistence comparison, MAE, RMSE, R² | `src/forecast.py`, `src/data.py` |
| Loss, prediction and residual plots | `src/forecast.py` |
| Perceptron, shapes, finite-difference gradients, training and reproducibility tests | `tests/test_powerwise.py` |

## Common viva questions

- **Why is demand forecasting regression?** Its output is a continuous numerical quantity.
- **Why use a linear output activation?** It allows continuous predictions without restricting them to a probability range.
- **Why can’t a single perceptron learn XOR?** A single linear boundary cannot separate XOR classes.
- **What does feedforward propagation do?** It computes activations from input toward the output layer.
- **What does backpropagation do?** It applies the chain rule to compute how each parameter affected the loss.
- **Why use mini-batches?** They average gradients over a small group while making more frequent updates than full-batch training.
- **Why split chronologically?** Future observations must not be used to train a model evaluated on earlier times.

## Limitations and future work

The built-in data is synthetic and simplified. For a college study, use a documented public electricity dataset, explain its units and missing values, and verify that weather features would have been known at forecast time. This small MLP is intended to make the mathematics clear, not to represent an operational utility forecasting system. Possible extensions include richer lag features, seasonal datasets, rolling-origin evaluation, and an optional PyTorch comparison model.
