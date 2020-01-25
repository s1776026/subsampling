import numpy as np


# Define interaction functions
def zero(x_i):
    """No interaction between particles"""
    return np.zeros_like(x_i)


def uniform(x_i):
    """All particles interact with every other particle equally"""
    return -2 * np.ones_like(x_i)


def exp_product(x_i):
    return -np.exp(-x_i)


def neg_exp(x_i):
    return -(2 - np.exp(-(x_i ** 2)))


def arctan(x_i):
    return np.arctan(x_i)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.arange(0, np.pi, 0.01)
    for function_str in dir():
        phi_function = eval(function_str)
        if callable(phi_function):
            plt.plot(x, phi_function(x), label=phi_function.__name__)
    plt.legend()
    plt.suptitle("Interaction Functions")
    plt.show()
