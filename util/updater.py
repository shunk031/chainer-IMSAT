import chainer

from chainer import training
from chainer.dataset import convert


class IMSATClusterUpdater(training.StandardUpdater):

    def __init__(self, iterator, optimizer,
                 mu, lam, converter=convert.concat_examples, device=None, loss_func=None):
        super(IMSATClusterUpdater, self).__init__(
            iterator, optimizer, converter=converter, device=device, loss_func=loss_func)

        self.mu = mu
        self.lam = lam

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        x = chainer.Variable(in_arrays)
        optimizer = self._optimizers['main']

        model = optimizer.target
        loss_eq1, loss_eq2 = model.loss_equal(x)
        loss_eq = loss_eq1 - self.mu * loss_eq2

        loss_ul = model.loss_unlabeled(x)
        model.cleargrads()

        loss = loss_eq + loss_ul + self.lam
        loss.backward()

        optimizer.update()
        chainer.report({
            'max_entropy': loss_eq1,
            'min_entropy': loss_eq2,
            'vat': loss_ul,
        }, model)
