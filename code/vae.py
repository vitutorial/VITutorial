import mxnet as mx
from abc import ABC
from typing import List, Tuple
from kl_divergences import diagonal_gaussian_kl


class Generator(ABC):
    def __init__(self, data_dims: int, layer_sizes: List[int], act_type: str) -> None:
        self.data_dims = data_dims
        self.layer_sizes = layer_sizes
        self.act_type = act_type

    def generate_sample(self, latent_state: mx.sym.Symbol) -> mx.sym.Symbol:
        raise NotImplementedError()

    def train(self, latent_state: mx.sym.Symbol) -> None:
        raise NotImplementedError()


class ProductOfBernoullisGenerator(Generator):
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
            act_i = mx.sym.Activation(data=fc_i, act_type="relu", name="gen_act_{}".format(i))
            prev_out = act_i

        # The output layer that gives pre_activations for multiple Bernoulli softmax between 0 and 1
        fc_out = mx.sym.FullyConnected(data=prev_out, num_hidden=2 * self.data_dims, name="gen_fc_out")

        return fc_out

    def generate_sample(self, latent_state: mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Generates a data sample.

        :param latent_state: The input latent state
        :return: A vector of Bernoulli draws
        """
        act = mx.sym.Activation(data=self._generate(latent_state=latent_state), act_type=self.output_act,
                                name="gen_act_out")
        act = mx.ndarray(mx.sym.split(data=act, num_outputs=self.data_dims))
        out = mx.sym.maximum(data=act, axis=0)

        return out

    def train(self, latent_state=mx.sym.Symbol, label=mx.sym.Symbol) -> mx.sym.Symbol:
        """
        Train the generator from a given latent state
        :param latent_state: The input latent state
        :param label: A binary vector (same as input for inference module)
        """
        output = self._preactivation(latent_state=latent_state)
        print('generator has computed preactivation')
        output = mx.sym.reshape(data=output, shape=(-1, 2, self.data_dims))
        print('starting to compute loss')
        return mx.sym.SoftmaxOutput(data=output, label=label, multi_output=True)


class InferenceNetwork(ABC):
    def __init__(self, latent_variable_size, layer_sizes: List[int], act_type: str) -> None:
        self.latent_var_size = latent_variable_size
        self.layer_sizes = layer_sizes
        self.act_type = act_type

    def inference(self, data: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        raise NotImplementedError()


class GaussianInferenceNetwork(InferenceNetwork):
    def __init__(self, latent_variable_size: int, layer_sizes: List[int], act_type: str):
        super().__init__(latent_variable_size, layer_sizes, act_type)

    def inference(self, data: mx.sym.Symbol) -> Tuple[mx.sym.Symbol, mx.sym.Symbol]:
        shared_layer = mx.sym.FullyConnected(data=data, num_hidden=self.layer_sizes[0], name="inf_joint_fc")
        shared_layer = mx.sym.Activation(data=shared_layer, act_type="relu", name="inf_joint_act")

        prev_out = shared_layer
        for i, size in enumerate(self.layer_sizes[1:]):
            mean_fc_i = mx.sym.FullyConnected(data=prev_out, num_hidden=size, name="inf_mean_fc_{}".format(i))
            mean_act_i = mx.sym.Activation(data=mean_fc_i, act_type="relu", name="inf_mean_act_{}".format(i))
            prev_out = mean_act_i
        mean = mx.sym.FullyConnected(data=prev_out, num_hidden=self.latent_var_size, name="inf_mean_compute")

        prev_out = shared_layer
        for i, size in enumerate(self.layer_sizes[1:]):
            var_fc_i = mx.sym.FullyConnected(data=prev_out, num_hidden=size, name="rec_var_fc_{}".format(i))
            var_act_i = mx.sym.Activation(data=var_fc_i, act_type="relu", name="rec_var_act_{}".format(i))
            prev_out = var_act_i
        # exp maps variance on non-negative real line
        # TODO rename to std
        var = mx.sym.exp(mx.sym.FullyConnected(data=prev_out, num_hidden=self.latent_var_size, name="inf_var_compute"))

        return mean, var

    def sample_latent_state(self, mu: mx.sym.Symbol, sigma: mx.sym.Symbol, batch_size: int) -> mx.sym.Symbol:
        """
        Sample a latent Gaussian variable

        :param mu: The mean of the Gaussian
        :param sigma: The standard deviation of the Gaussian
        :return: A Gaussian sample
        """
        # TODO sampling needs to be adjusted once correlations are introduced
        print('sampling')
        return mu + sigma * mx.sym.random_normal(loc=0, scale=1, shape=(batch_size, self.latent_var_size))


class VAE(ABC):
    def __init__(self, generator: Generator, inference_net: InferenceNetwork) -> None:
        self.generator = generator
        self.inference_net = inference_net

    def train(self, data: mx.sym.Symbol) -> None:
        """
        Train the generator and inference network jointly by optimising the ELBO.

        :param data: The training data
        """
        raise NotImplementedError()

    def generate_reconstructions(self, data:mx.sym.Symbol, n: int) -> mx.sym.Symbol:
        """
        Generate a number of reconstructions of input data points.

        :param data: The input data
        :param n: Number of reconstructions per data point
        :return: mx.sym.Symbol
        """
        raise NotImplementedError()

    def phantasize(self, n: int) -> mx.sym.Symbol:
        """
        Generate data by randomly sampling from the prior (currently standard normal).

        :param n: Number of sampled data points
        :return: Randomly generated data points
        """
        raise NotImplementedError()


class GaussianVAE(VAE):
    def __init__(self,
                 generator: Generator,
                 inference_net: GaussianInferenceNetwork) -> None:
        self.generator = generator
        self.inference_net = inference_net

    def train(self, data: mx.sym.Symbol, label: mx.sym.Symbol, batch_size: int) -> None:
        """
        Train the generator and inference network jointly by optimising the ELBO.

        :param data: The training data
        :param label: Copy of the training data
        """
        mu, sigma = self.inference_net.inference(data=data)
        latent_state = self.inference_net.sample_latent_state(mu, sigma, batch_size)
        mx.sym.MakeLoss(diagonal_gaussian_kl(mu, sigma))
        return self.generator.train(latent_state=latent_state, label=label)

    def generate_reconstructions(self, data:mx.sym.Symbol, n: int) -> mx.sym.Symbol:
        """
        Generate a number of reconstructions of input data points.

        :param data: The input data
        :param n: Number of reconstructions per data point
        :return: mx.sym.Symbol
        """
        mu, sigma = self.inference_net.inference(data=data)
        mu = mx.sym.tile(data=mu, reps=(n,1))
        sigma = mx.sym.tile(data=sigma, reps=(n,1))
        latent_state = self.sample_latent_state(mu, sigma, n)
        return self.generator.generate_sample(latent_state=latent_state)

    def phantasize(self, n: int) -> mx.sym.Symbol:
        """
        Generate data by randomly sampling from the prior (currently standard normal).

        :param n: Number of sampled data points
        :return: Randomly generated data points
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

    :param latent_type: Distribution of latent variable
    :param likelihood: Type of likelihood
    :param generator_layer_sizes: Sizes of generator hidden layers
    :param infer_layer_size: Sizes of inference network hidden layers
    :param latent_variable_size: Size of the latent variable
    :param data_dims: Dimensionality of the data
    :param generator_act_type: Activation function for generator hidden layers
    :param infer_act_type: Activation function for inference network hidden layers
    :return: A variational autoencoder
    """
    if likelihood == "bernoulliProd":
        generator = ProductOfBernoullisGenerator(data_dims=data_dims, layer_sizes=generator_layer_sizes,
                                                 act_type=generator_act_type)
    else:
        raise Exception("{} is an invalid likelihood type.".format(likelihood))

    if latent_type == "gaussian":
        inference_net = GaussianInferenceNetwork(latent_variable_size=latent_variable_size,
                                                 layer_sizes=infer_layer_size, act_type=infer_act_type)
        return GaussianVAE(generator=generator, inference_net=inference_net)
    else:
        raise Exception("{} is an invalid latent variable type.".format(latent_type))



