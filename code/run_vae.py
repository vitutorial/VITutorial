import mxnet as mx
import urllib.request
import os, logging, sys
from typing import Optional, List
from argparse import ArgumentParser
from os.path import join, exists
from numpy import genfromtxt
from .vae import construct_vae

# from matplotlib import pyplot as plt

DEFAULT_LEARNING_RATE = 0.0003

data_names = ['train', 'valid', 'test']
train_set = ['train', 'valid']
test_set = ['test']
data_dir = join(os.curdir, "binary_mnist")


# def plot_digit(digit: np.array) -> None:
#     '''
#     Plots an mnist digit encoded in a pixel array.
#
#     :param digit: An array of pixels.
#     '''
#     size = np.sqrt(digit.shape[0])
#     digit.reshape((size, size))
#
#     plt.imshow(digit, cmap='gray')
#     plt.show()


def load_data(train: bool = True, logger: Optional[logging.Logger] = logging) -> dict:
    '''
    Download binarised mnist data set and load it into memory.

    :param: Whether to load training or test data.
    :param: A logger for the data loading process.
    :return: Binarised mnist data.
    '''
    if not exists(data_dir):
        os.mkdir(data_dir)
    for data_set in data_names:
        file_name = "binary_mnist.{}".format(data_set)
        goal = join(data_dir, file_name)
        if exists(goal):
            logger.info("Data file {} exists".format(file_name))
        else:
            logger.info("Downloading {}".format(file_name))
            link = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat".format(
                data_set)
            urllib.request.urlretrieve(link, goal)
            logger.info("Finished")

    data = {}
    data_sets = train_set if train else test_set
    for data_set in data_sets:
        file_name = join(data_dir, "binary_mnist.{}".format(data_set))
        logger.info("Reading {} into memory".format(file_name))
        data[data_set] = mx.nd.array(genfromtxt(file_name))
        print("{} contains {} data points".format(file_name, data[data_set].shape)[0])

    return data


def train_model(generator_layers: List[int],
                inference_layers: List[int],
                latent_size: int,
                batch_size: int,
                epochs: int = 10,
                learning_rate: float = DEFAULT_LEARNING_RATE,
                optimiser: str = "adam",
                ctx: mx.context = mx.cpu(),
                logger: Optional[logging.logger] = logging):
    """
    Train a variational autoencoder model.

    :param generator_layers:
    :param inference_layers:
    :param latent_size:
    :param batch_size:
    :param ctx:
    :param logger:
    :return:
    """
    mnist = load_data(train=True, logger=logger)
    train_iter = mx.io.NDArrayIter(data=mnist['train'], data_name="data", label=mnist["train"], label_name="label",
                                   batch_size=batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(data=mnist['valid'], data_name="data", label=mnist['valid'], label_name="label",
                                 batch_size=batch_size)

    vae = construct_vae(latent_type="gaussian", likelihood="bernoulliProd", generator_layer_sizes=generator_layers,
                        infer_layer_size=inference_layers, latent_variable_size=latent_size,
                        data_dims=mnist['train'].shape[1], generator_act_type='tanh', infer_act_type='tanh')

    module = mx.module.Module(vae.train(mx.sym.Variable("data"), mx.sym.Variable('label')),
                              data_names=[train_iter.provide_data[0][0]],
                              label_names=["label"], context=ctx,
                              logger=logger)

    logger.info("Starting to train")
    module.fit(train_data=train_iter, optimizer=optimiser, force_init=True, force_rebind=True, num_epoch=epochs,
               optimizer_params={'learning_rate': learning_rate},
               # validation_metric=mx.metric.Perplexity(None),
               # eval_data=val_iter,
               batch_end_callback=mx.callback.Speedometer(frequent=1, batch_size=batch_size),
               epoch_end_callback=mx.callback.do_checkpoint('vae'))


def load_model(model_file: str):
    # TODO
    pass


def main():
    command_line_parser = ArgumentParser("Train a VAE on binary mnist and generate images of random digits.")

    command_line_parser.add_argument('-b', '--batch-size', default=500, type=int,
                                     help="Training batch size. Default: %(default)s.")
    command_line_parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="adam",
                                     help="The optimizer to use during training. "
                                          "Choices: %(choices)s. Default: %(default)s")
    command_line_parser.add_argument('-e', '--epochs', default=20, type=int,
                                     help="Number of epochs to run during training. Default: %(default)s.")
    command_line_parser.add_argument('--latent-dim', type=int, default=300,
                                     help='Dimensionality of the latent variable. Default: %(default)s.')
    command_line_parser.add_argument('--num-gpus', type=int, default=0,
                                     help="Number of GPUs to use. CPU is used if set to 0. Default: %(default)s.")
    command_line_parser.add_argument('-s', '--sample-random-digits', action='store_true',
                                     help="Load parameters of a previously trained VAE and randomly generate "
                                          "image digits from it.")

    args = command_line_parser.parse_args()

    ctx = mx.cpu() if args.num_gpus <= 0 else [mx.gpu(i) for i in range(args.num_gpus)]
    opt = args.opt
    epochs = args.epochs
    batch_size = args.batch_size

    generator_layers = [400, 600]
    inference_layers = [600, 400]
    latent_size = args.latent_dim

    training = not args.sample_random_digits

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    if training:
        train_model(generator_layers=generator_layers, inference_layers=inference_layers, latent_size=latent_size,
                    batch_size=batch_size, epochs=epochs, optimiser=opt, ctx=ctx)
    else:
        mnist = load_data(False, logger)
        test_iter = mx.io.NDArrayIter(data=mnist['test'], label=mnist['test'], label_name="label")




#     # set up training
#     module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label, for_training=True,
#                 force_rebind=True)
#     module.init_params(force_init=True)
#     module.init_optimizer(optimizer=opt, optimizer_params={('learning_rate', learning_rate)})
#
#     eval_metric = mx.metric.Accuracy()
#
#     callback_func = mx.callback.do_checkpoint('vae',1)
#
# #     for epoch in range(epochs):
#         tic = time.time()
#         eval_metric.reset()
#         nbatch = 0
#         data_iter = iter(train_iter)
#         end_of_batch = False
#         next_data_batch = next(data_iter)
#         while not end_of_batch:
#             data_batch = next_data_batch
#             module.forward_backward(data_batch)
#             module.update()
#             try:
#                 # pre fetch next batch
#                 next_data_batch = next(data_iter)
#                 module.prepare(next_data_batch)
#             except StopIteration:
#                 end_of_batch = True
#
#             module.update_metric(eval_metric, data_batch.label)
#
#             batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
#                                                       eval_metric=eval_metric,
#                                                       locals=locals())
#             callback_func(batch_end_params)
#             nbatch += 1
#
#             print('{} data points processed'.format(nbatch * batch_size))
#
#         # one epoch of training is finished
#         for name, val in eval_metric.get_name_value():
#             module.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
#         toc = time.time()
#         module.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))
#
#         # sync aux params across devices
#         arg_params, aux_params = module.get_params()
#         module.set_params(arg_params, aux_params)
#
#         # ----------------------------------------
#         # evaluation on validation set
#         # if eval_data:
#         #     res = module.score(eval_data, validation_metric,
#         #                        score_end_callback=eval_end_callback,
#         #                        batch_end_callback=eval_batch_end_callback, epoch=epoch)
#         #     # TODO: pull this into default
#         #     for name, val in res:
#         #         module.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
#
#         # end of 1 epoch, reset the data-iter for another epoch
#         print('finished epoch')
#         train_iter.reset()
#
#
# # module.fit(train_data=train_iter, eval_data=val_iter, optimizer=opt, num_epoch=epochs)


if __name__ == "__main__":
    main()
