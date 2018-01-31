import chainer
import chainer.functions as F
import chainer.links as L


class Encoder(chainer.Chain):

    def __init__(self, in_size=784, h0=1200, h1=1200, out_size=10):

        initialW = chainer.initializers.HeNormal(scale=0.1)
        super(Encoder, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(in_size, h0, initialW=initialW)
            self.bn1 = L.BatchNormalization(h0)
            self.fc2 = L.Linear(h0, h1, initialW=initialW)
            self.bn2 = L.BatchNormalization(h1)
            self.fc3 = L.Linear(h1, out_size)

        self.n_class = out_size

    def __call__(self, x):

        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)

        return h
