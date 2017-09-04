import mxnet as mx


def diagonal_gaussian_kl(mean, std):
    """
    Compute the KL divergence between a diagonal and standard Gaussian.

    :param mean: The mean of the diagonal Gaussian.
    :param std: The standard deviation of the diagonal Gaussian.
    :return: The KL divergence.
    """
    var = std ** 2
    return 0.5 * (mx.sym.sum(1 + mx.sym.log(var) - mean ** 2 - var))