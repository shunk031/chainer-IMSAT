import chainer

from chainer import training
from chainer.training import extensions

from sklearn.datasets import fetch_20newsgroups

from util.args import parse_args
from util.updater import IMSATClusterUpdater
from util.wrapper import ClusterWrapper
from model import Encoder


def main():

    args = parse_args()

    encoder = ClusterWrapper(Encoder(), prop_eps=0.25)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(encoder)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        encoder.to_gpu()

    dataset, _ = chainer.datasets.get_mnist(withlabel=False)
    train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)
    _, test = chainer.datasets.get_mnist()
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = IMSATClusterUpdater(train_iter, optimizer, mu=args.mu, lam=args.lam, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    trainer.extend(extensions.Evaluator(
        test_iter, encoder, device=args.gpu, eval_func=encoder.compute_accuracy),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'main/entropy', 'main/conditional_entropy', 'main/Rsat',
        'validation/main/accuracy',
    ]), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        encoder, 'model_epoch_{.updater.epoch}.npz'), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    main()
