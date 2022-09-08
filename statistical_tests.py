# Implementing Smooth Test from Neyman's Paper
from numpy import ndarray
import numpy as np
import math
import scipy.special as sc
from scipy.stats import norm


class smooth_tests:
    def __init__(
        self,
        number_of_degree: int,
        epsilon: float,
    ):
        if number_of_degree > 4:
            raise ValueError("Degree cannot be larger than 4!")
        self.degree = number_of_degree
        self.threshold = sc.gammainccinv(self.degree / 2, epsilon) * 2
        self.function_list_complete = [
            self.legendre_1,
            self.legendre_2,
            self.legendre_3,
            self.legendre_4,
        ]

    def legendre_1(self, x: ndarray):
        return math.sqrt(12) * (x - 0.5)

    def legendre_2(self, x: ndarray):
        return math.sqrt(5) * (6 * (x - 0.5) ** 2 - 0.5)

    def legendre_3(self, x: ndarray):
        return math.sqrt(7) * (20 * (x - 0.5) ** 3 - 3 * (x - 0.5))

    def legendre_4(self, x: ndarray):
        return 210 * (x - 0.5) ** 4 - 45 * (x - 0.5) ** 2 + 9 / 8

    def smooth_test(self, x: ndarray):
        score = np.zeros(x.shape[0])
        size = x.shape[1]
        detection_result = np.zeros(x.shape[0])
        for i in range(self.degree):
            f = self.function_list_complete[i]
            u = 1 / math.sqrt(size) * np.sum(f(x), axis=1)
            score += u**2

        detection_result[score >= self.threshold] = 1

        return detection_result


def runs_up_and_down(test: ndarray):
    # test = z_train.numpy()
    # test = test.flatten()
    # test = np.reshape(test, (len(test), 1))
    # print(test.shape)
    print("run tests", test.shape[0])
    run_up = 0
    run_down = 0
    for i in range(test.shape[0] - 1):
        if test[i] > test[i - 1] and test[i] > test[i + 1]:
            run_down = run_down + 1
        if test[i] < test[i - 1] and test[i] < test[i + 1]:
            run_up = run_up + 1
    runs = run_up + run_down
    n = test.shape[0]
    mu_r = (2 * n - 1) / 3
    sigma_r = np.sqrt((16 * n - 29) / 90)
    if runs > mu_r:
        z_ud = (runs - mu_r - 0.5) / sigma_r
    else:
        z_ud = (runs - mu_r + 0.5) / sigma_r
    prob = 2 * norm.cdf(-abs(z_ud))
    return prob
