import numpy as np


def sphere(x):
    return np.sum(np.square(x), axis=-1)


def rastrigin(x):
    dim = x.shape[-1]
    return 10.0 * dim + np.sum(np.square(x) - 10.0 * np.cos(2.0 * np.pi * x), axis=-1)


def rosenbrock(x):
    return np.sum(100.0 * np.square(x[..., 1:] - np.square(x[..., :-1])) + np.square(1.0 - x[..., :-1]), axis=-1)


def ackley(x):
    dim = x.shape[-1]
    sq = np.sum(np.square(x), axis=-1)
    cos = np.sum(np.cos(2.0 * np.pi * x), axis=-1)
    return -20.0 * np.exp(-0.2 * np.sqrt(sq / dim)) - np.exp(cos / dim) + 20.0 + np.e


OBJECTIVES = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
}


DEFAULT_BOUNDS = {
    "sphere": (-5.12, 5.12),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-5.0, 10.0),
    "ackley": (-32.768, 32.768),
}
