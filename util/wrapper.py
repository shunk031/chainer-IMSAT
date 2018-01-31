import numpy as np
import chainer
import chainer.functions as F

from munkres import Munkres


class ClusterWrapper(chainer.Chain):

    def __init__(self, encoder, prop_eps):

        super(ClusterWrapper, self).__init__()
        with self.init_scope():
            self.encoder = encoder

        self.prop_eps = prop_eps

    def aux(self, x):
        return self.encoder(x)

    def entropy(self, p):
        if not isinstance(p, chainer.Variable):
            p = chainer.Variable(p)

        if p.data.ndim == 1:
            return -F.sum(p * F.log(p + 1e-8))
        elif p.data.ndim == 2:
            return -F.sum(p * F.log(p + 1e-8)) / len(p.data)
        else:
            raise NotImplementedError()

    def KL(self, p, q):

        if not isinstance(p, chainer.Variable):
            p = chainer.Variable(p)
            q = chainer.Variable(q)

        kl_d = F.sum(p * F.log((p + 1e-8) / (q + 1e-8))) / len(p.data)
        return kl_d

    def KL_distance(self, y0, y1):
        s_y0 = F.softmax(y0)
        s_y1 = F.softmax(y1)

        return self.KL(s_y0, s_y1)

    def convert_unit_vector(self, v):
        xp = chainer.cuda.get_array_module(v)
        v = v / (xp.sqrt(xp.sum(v ** 2, axis=1)).reshape((-1, 1)) + 1e-16)
        return v

    def VAT(self, x, xi=10, Ip=1):
        xp = chainer.cuda.get_array_module(x)

        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)

        with chainer.using_config('train', False):
            y1 = self.encoder(x)
        y1.unchain_backward()

        d = xp.random.normal(size=x.shape)
        d = d / xp.sqrt(xp.sum(d ** 2, axis=1)).reshape((x.shape[0], 1))

        for ip in range(Ip):
            d_var = chainer.Variable(d.astype(np.float32))

            with chainer.using_config('train', False):
                y2 = self.encoder(x + xi * d_var)

            kl_d = self.KL(y1, y2)
            kl_d.backward()
            d = d_var.grad
            d = self.convert_unit_vector(d)
        d_var = chainer.Variable(d.astype(np.float32))

        with chainer.using_config('train', False):
            y2 = self.encoder(x + self.prop_eps * d_var)

        vat_value = self.KL_distance(y1, y2)
        return vat_value

    def loss_equal(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x.astype(np.float32))

        p_logit = self.encoder(x)
        p = F.softmax(p_logit)
        p_ave = F.sum(p, axis=0) / x.data.shape[0]
        entropy = self.entropy(p)

        return entropy, -F.sum(p_ave * F.log(p_ave + 1e-8))

    def loss_unlabeled(self, x):
        vat_loss = self.VAT(x)
        return vat_loss

    def loss_test(self, x, t):
        xp = chainer.cuda.get_array_module(x)
        with chainer.using_config('train', False):
            prob = F.softmax(self.encoder(x)).data

        p_margin = xp.sum(prob, axis=0) / len(prob)
        entropy = xp.sum(-p_margin * xp.log(p_margin + 1e-8))
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
        pred_corresp = [corresp[int(predicted)] for predicted in prediction]
        acc = xp.sum(pred_corresp == tt) / len(tt)

        chainer.report({'accuracy': acc,
                        'entropy': entropy}, self)
