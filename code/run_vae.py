import mxnet as mx
import urllib.request
import os
from os.path import join, exists
from numpy import genfromtxt
from vae import construct_vae

data_sets = ['train', 'valid', 'test']
data_dir = join(os.curdir, "binary_mnist")


def load_data() -> dict:
    '''
    Download binarised mnist data set and load it into memory.

    :return: binarised mnist data
    '''
    if not exists(data_dir):
        os.mkdir(data_dir)
    for data_set in data_sets:
        file_name = "binary_mnist.{}".format(data_set)
        goal = join(data_dir, file_name)
        if exists(goal):
            print("Data file {} exists".format(file_name))
        else:
            print("Downloading {}".format(file_name))
            link = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat".format(
                data_set)
            urllib.request.urlretrieve(link, goal)
            print("Finished")

    data = {}
    for data_set in data_sets:
        file_name = join(data_dir, "binary_mnist.{}".format(data_set))
        print("Reading {} into memory".format(file_name))
        data[data_set] = mx.nd.array(genfromtxt(file_name))
        print("Data shape = {}".format(data[data_set].shape))

    return data


def main():
    ctx = mx.cpu(0)
    opt = "adam"
    learning_rate = 0.0003
    epochs = 20
    batch_size = 100

    generator_layers = [400, 600]
    inference_layers = [600, 400]
    latent_size = 300

    mnist = load_data()

    train_iter = mx.io.NDArrayIter(data=mnist['train'], data_name="data", label=mnist["train"], label_name="label",
                                   batch_size=batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['valid'], data_name="data")
    test_iter = mx.io.NDArrayIter(data=mnist['test'], label_name="label")

    vae = construct_vae(latent_type="gaussian", likelihood="bernoulliProd", generator_layer_sizes=generator_layers,
                        infer_layer_size=inference_layers, latent_variable_size=latent_size,
                        data_dims=mnist['train'].shape[1], generator_act_type='tanh', infer_act_type='tanh')

    module = mx.module.Module(vae.train(mx.sym.Variable("data"), mx.sym.Variable('label')),
                              data_names=[train_iter.provide_data[0][0]],
                              label_names=["label"], context=ctx)

    print("starting to train")
    module.fit(train_data=train_iter, optimizer=opt,force_init=True, force_rebind=True, num_epoch=epochs,
               optimizer_params={'learning_rate': learning_rate},
               batch_end_callback=mx.callback.Speedometer(frequent=1, batch_size=batch_size),
               epoch_end_callback=mx.callback.do_checkpoint('vae'))

    # # set up training
    # module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label, for_training=True,
    #             force_rebind=True)
    # module.init_params(force_init=True)
    # module.init_optimizer(optimizer=opt, optimizer_params={('learning_rate', learning_rate)})
    #
    # eval_metric = mx.metric.Accuracy()
    #
    # callback_func = mx.callback.do_checkpoint('params',1)

#     for epoch in range(epochs):
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
