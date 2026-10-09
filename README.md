# Rosenblatt's Perceptron from Scratch

A single-layer perceptron implemented in NumPy and trained with the classic perceptron update rule on a synthetic, linearly separable 2D dataset.

**Interactive demo:** [Run it in your browser](https://laiba-khawar-portfolio.vercel.app/work/perceptron#demo)

## How it works

All code is in `Rosenblatt’s Perceptron.ipynb`.

- **Data.** 500 points drawn from a standard normal distribution in 2D (`np.random.seed(42)`). The label is `+1` when `x1 + x2 > 0` and `-1` otherwise, so the classes are separated by the line `x1 + x2 = 0`. The data is split 80/20 (`train_test_split`, `random_state=42`): 400 training and 100 test points.
- **Model.** Weights start at zero (stored as a 2x1 matrix), bias at zero. The output is a step function: `+1` if `w·x + b >= 0`, else `-1`.
- **Training.** For each sample in each epoch: `error = y - y_hat`, then `w += lr * error * x` and `b += lr * error`. Learning rate `0.01`, `20` epochs. The notebook prints the summed absolute error per epoch (each mistake adds 2, because labels are ±1).
- **Evaluation.** A decision boundary plot over the training data and the accuracy on the 100 held-out points.

## Results

Recorded output of `Rosenblatt’s Perceptron.ipynb`, cell 3:

- Test accuracy: **0.96**

The per-epoch error printed in the same cell does not reach zero within 20 epochs, so the perceptron had not fully separated the training set when training stopped.

## Repository contents

- `Rosenblatt’s Perceptron.ipynb`: the notebook (note the curly apostrophe in the file name).
- `requirements.txt`: jupyter, matplotlib, numpy, scikit-learn.

## Running it

The notebook was last run with Python 3.9.

```bash
git clone https://github.com/LaibaKhawar/Rosenblatt-s-Perceptron-from-Scratch.git
cd Rosenblatt-s-Perceptron-from-Scratch
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

No external data is needed; the dataset is generated in the notebook.

## Known limitations

- Training runs for a fixed 20 epochs with no convergence check, and the training error is still non-zero at the end.
- Only one synthetic dataset and one learning rate are tested.
- scikit-learn is used only for the train/test split.

## Author

[Laiba Khawar](https://github.com/LaibaKhawar) · [LinkedIn](https://www.linkedin.com/in/laiba-k-00b2b1249/)
