import chainer
import chainer.functions as F
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
        batch_size = self._iterators['main'].batch_size

        optimizer = self._optimizers['main']
        model = optimizer.target
        p = model.classify(in_arrays)
        hy = model.compute_marginal_entropy(p)
        hy_x = F.sum(model.compute_entropy(p)) / batch_size
        Rsat = -F.sum(model.compute_lds(in_arrays)) / batch_size

        loss = Rsat - self.lam * (self.mu * hy - hy_x)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        chainer.report({
            'loss': loss,
            'entropy': hy,
            'conditional_entropy': hy_x,
            'Rsat': Rsat,
        }, model)
