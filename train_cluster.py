import numpy as np
import chainer

from chainer import training
from chainer.training import extensions

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from util.args import parse_args
from util.args import DATASETS
from util.updater import IMSATClusterUpdater
from util.wrapper import ClusterWrapper
from model import Encoder


def main():

    args = parse_args()

    if args.dataset == 'mnist':
        dataset, _ = chainer.datasets.get_mnist(withlabel=False)
        train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

        _, test = chainer.datasets.get_mnist()
        test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
        n_class = 10

    elif args.dataset == '20news':
        newsgroups_train = fetch_20newsgroups(subset='train')

        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        dataset = vectorizer.fit_transform(
            newsgroups_train.data).todense().astype(np.float32)
        train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

        newsgroups_test = fetch_20newsgroups(subset='test')
        test_vectors = vectorizer.transform(
            newsgroups_test.data).todense().astype(np.float32)
        test = [(test_vector, target) for test_vector, target in zip(test_vectors, newsgroups_test.target)]
        test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
        n_class = 20

    else:
        raise NotImplementedError('Please select dataset from: {}'.format(DATASETS))

    encoder = ClusterWrapper(
        Encoder(in_size=dataset.shape[1], out_size=n_class), prop_eps=0.25)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        encoder.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(encoder)

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
