import numpy as np
from numpy import ndarray
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from statistical_tests import smooth_tests
import tensorflow as tf
from tensorflow import Tensor


def get_training_samples(dataSet, nI, k_size):
    """Reshape and prepare 1D array for WGAN input
    nI: number of sampler per row
    k_size: filter size of the CNN, determines the step size of the slide window
    """

    x = np.zeros((dataSet.size // (nI - k_size + 1) - 1, nI))
    for i in range(dataSet.size // (nI - k_size + 1) - 1):
        j = i * (nI - k_size + 1)
        x[i] = np.reshape(dataSet[j : j + nI], (nI,))
    return x


def loss_function(
    real_uniform_samples: Tensor,
    innovations: Tensor,
    real_original_samples: Tensor,
    reconstructed_original_samples: Tensor,
    equivalence_decoder_weight: float,
):
    indepedence_discriminator_loss = tf.reduce_mean(
        real_uniform_samples
    ) - tf.reduce_mean(innovations)
    equivalence_discriminator_loss = tf.reduce_mean(
        real_original_samples
    ) - tf.reduce_mean(reconstructed_original_samples)
    loss = (
        indepedence_discriminator_loss
        + equivalence_decoder_weight * equivalence_discriminator_loss
    )

    return loss


def empirical_cdf_transform(x: ndarray, x_fit: ndarray):
    """
    Calculate the transformation via empirical CDF (ECDF)
    x:the numpy array for which we shall calculate ECDF
    x_fit: the numpy array for which we shall perform the transformation
    return: the transformed array
    """

    ecdf = ECDF(x)
    ecdf_transformed = ecdf(x_fit)
    return ecdf_transformed


def reshape_innovations(x: ndarray, block_size: int, stride: int):
    """
    Reshape the 1D transformed innovations to 2D array with given shape and strides
    x: the given 1D array
    block_size: the number of element in a row (block)
    stride: strides of each column
    return: a new array
    """
    if stride == block_size:
        length = int(np.ceil(x.shape[0] // block_size))
        effective_length = length * block_size
        x_new = np.reshape(x[0:effective_length], (length, block_size))
    else:
        shape = (int(np.ceil((x.shape[0] - block_size) // stride)) + 1, block_size)
        strides = (stride * x.itemsize, x.itemsize)
        x_new = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_new


def ROC_curve_plotting(x_test: ndarray, x_bad: ndarray, degree: int, figname: str):
    true_positive = []
    false_positive = []
    set_of_epsilon = np.linspace(0, 1, 1001)
    for i in range(len(set_of_epsilon)):
        epsilon = set_of_epsilon[i]
        smooth_test = smooth_tests(epsilon=epsilon, number_of_degree=degree)
        true_positive.append(np.mean(smooth_test.smooth_test(x_bad)))
        false_positive.append(np.mean(smooth_test.smooth_test(x_test)))

    plt.plot(false_positive, true_positive)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.show()
    plt.savefig(figname)

    return true_positive, false_positive


# np.random.seed(sx_test_original = np.random.uniform(low=0,high=1,size=(1,1000))
# # x_test = np.repeat(x_test_original,200,axis=0)
# # x_test = x_test.flatten()
# # x_bad = np.random.normal(loc=0.5,scale=0.2,size=x_test.shape)
# # x_test_new = reshape_innovations(x_test,1000,500)
# # x_bad_new = reshape_innovations(x_bad,1000,1000)
# # ROC_curve_plotting(x_test_new,x_bad_new,4)eed=130)
#
