import mxnet as mx
from abc import ABC
from typing import List, Tuple, Callable
from .kl_divergences import diagonal_gaussian_kl


class Generator(ABC):
    """
    Generator network.

    :param data_dims: Dimensionality of the generated data.
    :param layer_sizes: Size of each layer in the network.
    :param act_type: The activation after each layer.
    """

    def __init__(self, data_dims: int, layer_sizes: List[int], act_type: str) -> None:
        self.data_dims = data_dims
        self.layer_sizes = layer_sizes
        self.act_type = act_type

    def generate_sample(self, latent_state: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Generate a data sample from a latent state.

        :param latent_state: The latent input state.
        :return: A data sample.
        """
        raise NotImplementedError()

    def train(self, latent_state: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Train the generator from a given latent state.

        :param latent_state: The latent input state
        :return: The loss symbol used for training.
        """
        raise NotImplementedError()


class ProductOfBernoullisGenerator(Generator):
    """
    A generator that produces binary vectors whose entries are independent Bernoulli draws.

    :param data_dims: Dimensionality of the generated data.
    :param layer_sizes: Size of each layer in the network.
    :param act_type: The activation after each layer.
    """
    def __init__(self, data_dims: int, layer_sizes=List[int], act_type=str) -> None:
        super().__init__(data_dims, layer_sizes, act_type)
        self.output_act = "sigmoid"

    def _preactivation(self, latent_state: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Computes the pre-activation of the generator, i.e. the hidden state before the final output activation.

        :param latent_state: The input latent state
        :return: The pre-activation before output activation
        """
        prev_out = None
        for i, hidden in enumerate(self.layer_sizes):
            fc_i = mx.sym.FullyConnected(data=latent_state, num_hidden=hidden, name="gen_fc_{}".format(i))
            prev_out = mx.sym.Activation(data=fc_i, act_type=self.act_type, name="gen_act_{}".format(i))

        # The output layer that gives pre_activations for multiple Bernoulli softmax between 0 and 1
        fc_out = mx.sym.FullyConnected(data=prev_out, num_hidden=self.data_dims, name="gen_fc_out")

        return fc_out

    def generate_sample(self, latent_state: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Generates a data sample.

        :param latent_state: The input latent state.
        :return: A vector of Bernoulli draws.
        """
        act = mx.sym.Activation(data=self._preactivation(latent_state=latent_state), act_type=self.output_act,
                                name="gen_act_out")
        out = act > 0.5
        return out

    def train(self, latent_state=mx.sym.Symbol, label=mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Train the generator from a given latent state

        :param latent_state: The input latent state
        :param label: A binary vector (same as input for inference module)
        :return: The loss symbol used for training
        """
        output = mx.sym.Activation(data=self._preactivation(latent_state=latent_state), act_type=self.output_act,
                                   name="output_act")
        return mx.sym.sum(label * mx.sym.log(output) + (1-label) * mx.sym.log(1-output), axis=[1])


class InferenceNetwork(ABC):
    """
    A network to infer distributions over latent states.

    :param latent_variable_size: The dimensionality of the latent variable.
    :param layer_sizes: Size of each layer in the network.
    :param act_type: The activation after each layer.
    """
    def __init__(self, latent_variable_size: int, layer_sizes: List[int], act_type: str) -> None:
        self.latent_var_size = latent_variable_size
        self.layer_sizes = layer_sizes
        self.act_type = act_type

    def inference(self, data: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, ...]:
        """
        Infer the parameters of the distribution over latent values.

        :param data: A data sample.
        :return: The parameters of the distribution.
        """
        raise NotImplementedError()


class GaussianInferenceNetwork(InferenceNetwork):
    """
    An inference network that predicts the parameters of a diagonal Gaussian.

    :param latent_variable_size: The dimensionality of the latent variable.
    :param layer_sizes: Size of each layer in the network.
    :param act_type: The activation after each layer.
    """
    def __init__(self, latent_variable_size: int, layer_sizes: List[int], act_type: str):
        super().__init__(latent_variable_size, layer_sizes, act_type)

    def inference(self, data: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        """
        Infer the mean and standard deviation.

        :param data: A data sample.
        :return: The mean and standard deviation.
        """
        shared_layer = mx.sym.FullyConnected(data=data, num_hidden=self.layer_sizes[0], name="inf_joint_fc")
        shared_layer = mx.sym.Activation(data=shared_layer, act_type=self.act_type, name="inf_joint_act")

        prev_out = shared_layer
        for i, size in enumerate(self.layer_sizes[1:]):
            mean_fc_i = mx.sym.FullyConnected(data=prev_out, num_hidden=size, name="inf_mean_fc_{}".format(i))
            mean_act_i = mx.sym.Activation(data=mean_fc_i, act_type=self.act_type, name="inf_mean_act_{}".format(i))
            prev_out = mean_act_i
        mean = mx.sym.FullyConnected(data=prev_out, num_hidden=self.latent_var_size, name="inf_mean_compute")

        prev_out = shared_layer
        for i, size in enumerate(self.layer_sizes[1:]):
            var_fc_i = mx.sym.FullyConnected(data=prev_out, num_hidden=size, name="rec_var_fc_{}".format(i))
            var_act_i = mx.sym.Activation(data=var_fc_i, act_type=self.act_type, name="rec_var_act_{}".format(i))
            prev_out = var_act_i
        # soft-relu maps std onto non-negative real line
        std = mx.sym.Activation(
            mx.sym.FullyConnected(data=prev_out, num_hidden=self.latent_var_size, name="inf_var_compute"),
            act_type="softrelu")

        return mean, std

    def sample_latent_state(self, mean: mx.sym.Symbol, std: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Sample a latent Gaussian variable

        :param mean: The mean of the Gaussian
        :param std: The standard deviation of the Gaussian
        :return: A Gaussian sample
        """
        # TODO sampling needs to be adjusted once correlations are introduced
        return mean + std * mx.sym.random_normal(loc=0, scale=1, shape=(0, self.latent_var_size))


# TODO allow specification of prior
class VAE(ABC):
    """
    A variational autoencoding model (Kingma and Welling, 2013).

    :param generator: A generator network that specifies the likelihood of the model.
    :param inference_net: An inference network that specifies the distribution over latent values.
    """
    def __init__(self, generator: Generator, inference_net: InferenceNetwork) -> None:
        self.generator = generator
        self.inference_net = inference_net

    def train(self, data: mx.sym.Symbol, label: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Train the generator and inference network jointly by optimising the ELBO.

        :param data: The training data.
        :param label: Copy of the training data.
        :param: A list of loss symbols.
        """
        raise NotImplementedError()

    def generate_reconstructions(self, data:mx.sym.Symbol, n: int) -> mx.sym.Symbol:
        """
        Generate a number of reconstructions of input data points.

        :param data: The input data.
        :param n: Number of reconstructions per data point.
        :return: The reconstructed data.
        """
        raise NotImplementedError()

    def phantasize(self, n: int) -> mx.sym.Symbol:
        """
        Generate data by randomly sampling from the prior.

        :param n: Number of sampled data points.
        :return: Randomly generated data points.
        """
        raise NotImplementedError()


class GaussianVAE(VAE):
    """
    A VAE with Gaussian latent variables. It assumes a standard normal prior on the latent values.

    :param generator: A generator network that specifies the likelihood of the model.
    :param inference_net: An inference network that specifies the Gaussian over latent values.
    """
    def __init__(self,
                 generator: Generator,
                 inference_net: GaussianInferenceNetwork,
                 kl_divergence: Callable) -> None:
        self.generator = generator
        self.inference_net = inference_net
        self.kl_divergence = kl_divergence

    def train(self, data: mx.sym.Symbol, label: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Train the generator and inference network jointly by optimising the ELBO.

        :param data: The training data.
        :param label: Copy of the training data.
        :return: A list of loss symbols.
        """
        mean, std = self.inference_net.inference(data=data)
        latent_state = self.inference_net.sample_latent_state(mean, std)
        kl_term = self.kl_divergence(mean, std)
        log_likelihood = self.generator.train(latent_state=latent_state, label=label)
        return mx.sym.MakeLoss(kl_term - log_likelihood, name="Neg_Elbo")

    def generate_reconstructions(self, data: mx.sym.Symbol, n: int) -> mx.sym.Symbol:
        """
        Generate a number of reconstructions of input data points.

        :param data: The input data.
        :param n: Number of reconstructions per data point.
        :return: The reconstructed data.
        """
        mean, std = self.inference_net.inference(data=data)
        mean = mx.sym.tile(data=mean, reps=(n, 1))
        std = mx.sym.tile(data=std, reps=(n, 1))
        latent_state = self.sample_latent_state(mean, std, n)
        return self.generator.generate_sample(latent_state=latent_state)

    def phantasize(self, n: int) -> mx.sym.Symbol:
        """
        Generate data by randomly sampling from the prior.

        :param n: Number of sampled data points.
        :return: Randomly generated data points.
        """
        latent_state = mx.sym.random_normal(loc=0, scale=1, shape=(n, self.inference_net.latent_var_size))
        return self.generator.generate_sample(latent_state=latent_state)


def construct_vae(latent_type: str,
                  likelihood: str,
                  generator_layer_sizes: List[int],
                  infer_layer_size: List[int],
                  latent_variable_size: int,
                  data_dims: int,
                  generator_act_type: str = "tanh",
                  infer_act_type: str = "tanh") -> VAE:
    """
    Construct a variational autoencoder

    :param latent_type: Distribution of latent variable.
    :param likelihood: Type of likelihood.
    :param generator_layer_sizes: Sizes of generator hidden layers.
    :param infer_layer_size: Sizes of inference network hidden layers.
    :param latent_variable_size: Size of the latent variable.
    :param data_dims: Dimensionality of the data.
    :param generator_act_type: Activation function for generator hidden layers.
    :param infer_act_type: Activation function for inference network hidden layers.
    :return: A variational autoencoder.
    """
    if likelihood == "bernoulliProd":
        generator = ProductOfBernoullisGenerator(data_dims=data_dims, layer_sizes=generator_layer_sizes,
                                                 act_type=generator_act_type)
    else:
        raise Exception("{} is an invalid likelihood type.".format(likelihood))

    if latent_type == "gaussian":
        inference_net = GaussianInferenceNetwork(latent_variable_size=latent_variable_size,
                                                 layer_sizes=infer_layer_size,
                                                 act_type=infer_act_type)
        return GaussianVAE(generator=generator, inference_net=inference_net, kl_divergence=diagonal_gaussian_kl)
    else:
        raise Exception("{} is an invalid latent variable type.".format(latent_type))
