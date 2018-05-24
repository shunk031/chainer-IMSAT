import chainer
import chainer.functions as F

from munkres import Munkres


class ClusterWrapper(chainer.Chain):

    def __init__(self, encoder, prop_eps):
        self.prop_eps = prop_eps

        super(ClusterWrapper, self).__init__()
        with self.init_scope():
            self.encoder = encoder

    def classify(self, x):
        y = F.softmax(self.encoder(x))
        return y

    def compute_entropy(self, p):
        if p.ndim == 2:
            return -F.sum(p * F.log(p + 1e-16), axis=1)

        return -F.sum(p * F.log(p + 1e-16))

    def compute_marginal_entropy(self, p_batch):
        return self.compute_entropy(F.mean(p_batch, axis=0))

    def compute_KLd(self, p, q):
        assert p.shape[0] == q.shape[0]

        return F.reshape(F.sum(p * (F.log(p + 1e-16) - F.log(q + 1e-16)), axis=1), (-1, 1))

    def get_unit_vector(self, v):
        xp = chainer.cuda.get_array_module(v)

        if v.ndim == 4:
            return v / (xp.sqrt(xp.sum(v ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
        return v / (xp.sqrt(xp.sum(v ** 2, axis=1)).reshape((-1, 1)) + 1e-16)

    def compute_lds(self, x, xi=10, Ip=1):
        xp = chainer.cuda.get_array_module(x)

        y1 = self.classify(x)
        y1.unchain_backward()

        d = chainer.Variable(self.get_unit_vector(xp.random.normal(size=x.shape)).astype(xp.float32))
        for ip in range(Ip):
            y2 = self.classify(x + xi * d)
            kld = F.sum(self.compute_KLd(y1, y2))
            kld.backward()
            d = self.get_unit_vector(d.grad)

        y2 = self.classify(x + self.prop_eps * d)
        return -self.compute_KLd(y1, y2)

    def compute_accuracy(self, x, t):
        xp = chainer.cuda.get_array_module(x)
        with chainer.using_config('train', False):
            prob = F.softmax(self.encoder(x)).data

        prediction = xp.argmax(prob, axis=1)

        if isinstance(t, chainer.Variable):
            tt = t.data
        else:
            tt = t

        m = Munkres()
        mat = xp.zeros(shape=(self.encoder.n_class, self.encoder.n_class))
        for i in range(self.encoder.n_class):
            for j in range(self.encoder.n_class):
                mat[i, j] = xp.sum(xp.logical_and(prediction == i, tt == j))

        indices = m.compute(-mat)
        corresp = [indices[i][1] for i in range(self.encoder.n_class)]
        pred_corresp = xp.asarray([corresp[int(predicted)] for predicted in prediction])
        acc = xp.sum(pred_corresp == tt) / len(tt)

        chainer.report({'accuracy': acc}, self)
