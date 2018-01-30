import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--out', default='result',
                        help='Output directory')
    parser.add_argument('--mu', type=float, default=4.0)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=1.0)

    return parser.parse_args()
