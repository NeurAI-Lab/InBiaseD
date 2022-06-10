import numpy as np
import matplotlib.pyplot as plt


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


lst_sig = [sigmoid_rampup(i, rampup_length=0) for i in range(200)]
lst_lin = [linear_rampup(i, rampup_length=0) for i in range(200)]


plt.plot(list(range(200)), lst_sig)
plt.plot(list(range(200)), lst_lin)

