"""
Implementation of gradeint descent to learn a simple linear function: f(x) = 3x+12.

Written for understandability.
"""


import numpy as np

def real_function(X):
    """The real function we are trying to learn: f(x)=3x+12"""
    return 3 * X + 12


def loss(b_0, b_1, x, y):
    """The mean squared error"""
    y_pred = b_0 + b_1 * x
    return sum((y - y_pred) ** 2) / len(x)


def grad_b_1(b_0, b_1, x, y):
    """
    returns the gradient of the loss function with respect to b_1
    b_0 and b_1 are our current guesses for the coefficients
    x and y are vectors for our training data

    L = sum((y_pred - y_real)**2) / n
      = sum((b_0 + b_1 * x - y_real)**2) / n
    dL/d[b_1] = 2 * sum((b_0 + b_1 * x - y)(x)) / n # chain rule
    """
    return 2 * np.sum((b_0 + b_1 * x - y) * x) / len(x)


def grad_b_0(b_0, b_1, x, y):
    """
    returns the gradient of the loss function with respect to b_0
    b_0 and b_1 are our current guesses for the coefficients
    x and y are vectors for our training data

    L = sum((y_pred - y_real)**2) / n
      = sum((b_0 + b_1 * x - y_real)**2) / n
    dL/d[b_0] = 2 * sum(b_0 + b_1 * x - y) / n # chain rule
    """
    return 2 * np.sum(b_0 + b_1 * x - y) / len(x)


def main():
    # 5000 numbers from -100 to 100
    X = np.linspace(-100, 100, 5000)
    y = real_function(X)

    # start off with f(x) = b_0 + b_1 * x
    # our coefficients are random guesses to start
    b_0 = np.random.normal()
    b_1 = np.random.normal()

    # multiplier for how much we update the weights each iteration
    learning_rate = 0.0001

    while loss(b_0, b_1, X, y) > 0:
        # calculate gradeint
        dL_db0 = grad_b_0(b_0, b_1, X, y)
        dL_db1 = grad_b_1(b_0, b_1, X, y)

        # update weights
        b_0 -= learning_rate * dL_db0
        b_1 -= learning_rate * dL_db1
        print(f"guess: {b_0:.5f} + {b_1:.5f} * x \tLoss: {loss(b_0, b_1, X, y):.5f}")


if __name__ == "__main__":
    main()
