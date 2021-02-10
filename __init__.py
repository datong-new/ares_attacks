import numpy as np
import random
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph
from ares.loss import CrossEntropyLoss
from ares.loss.base import Loss


class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        """ Based on ares.attack.bim.BIM, numpy version. """
        self.model, self.batch_size, self._session = model, batch_size, session
        # dataset == "imagenet" or "cifar10"
        #loss = CrossEntropyLoss(self.model)
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)
        self.xs_var = tf.Variable(tf.zeros(self.xs_ph.shape, dtype=self.model.x_dtype))
        self.ys_var = tf.Variable(tf.zeros(shape=(batch_size,), dtype=self.model.y_dtype))
        self.setup = [self.xs_var.assign(self.xs_ph),self.ys_var.assign(self.ys_ph)]

        #self.loss, self.stop_mask = loss(self.xs_var, self.ys_var)
        #self.grad = tf.gradients(self.loss, self.xs_var)[0]
        self.grad_ods, self.loss_ods, self.stop_mask_ods = self._get_gradients(loss_type="ods")
        self.grad_ce, self.loss_ce, self.stop_mask_ce = self._get_gradients(loss_type="ce")
        self.grad_cw, self.loss_cw, self.stop_mask_cw = self._get_gradients(loss_type="cw")

        self.grad, self.loss, self.stop_mask = self._get_gradients(loss_type="cw")

        self.iteration = 80

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            self.alpha = self.eps / 7


    def init_delta(self):
        return (2*np.random.uniform(size=self.xs_ph.shape)-1) * self.eps

    def _get_gradients(self, loss_type="ce"):
        logits, label = self.model._logits_and_labels(self.xs_var)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)

 #       loss += tf.reduce_sum((2*tf.random.uniform(logits.shape)-1)*logits, axis=-1)

        mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
        label_score = tf.reduce_sum(mask*logits, axis=1)
        second_scores = tf.reduce_max((1- mask) * logits,  axis=1)
        loss += -(label_score - second_scores)
        mean = (label_score+second_scores)/2

        L = label_score - second_scores
        c=10

        """
        for i in range(2, 20, 2):
            mean_1 = label_score -  L / float(i)
            mean_2 = second_scores + L /float(i)

            loss -= 1 / (1+tf.exp(-c*(label_score-mean_1))) + 1 / (1+tf.exp(-c*(mean_2-second_scores)))
        """

        mean_1 = label_score -  L / 5.0
        mean_2 = second_scores + L / 5.0

        loss -= 1 / (1+tf.exp(-c*(label_score-mean_1))) + 1 / (1+tf.exp(-c*(mean_2-second_scores)))
        loss -= 1 / (1+tf.exp(-c*(label_score-mean))) + 1 / (1+tf.exp(-c*(mean-second_scores)))


        grad = tf.gradients(loss, self.xs_var)[0]
        stop_mask = tf.cast(tf.equal(label, self.ys_var), dtype=tf.float32)
        return grad, loss, stop_mask



    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        xs_adv = xs

        prev_grad = np.zeros(xs.shape)
        m = np.zeros(xs.shape)
        v = np.zeros(xs.shape)
        stop_mask = None

        for i in range(self.iteration):

            """
            if i%20==0 or i%20==1:
                osd_flag = True
                #if stop_mask is not None:
                #    xs_adv = xs_adv * (1-stop_mask[:, None, None, None]) + (xs+self.init_delta()) * (stop_mask[:, None, None, None])
                self._session.run(self.setup,  feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
                grad = self._session.run(self.grad_ods)
                loss, stop_mask = self.loss_ods, self.stop_mask_ods
                prev_grad = np.zeros(xs.shape)
                m = np.zeros(xs.shape)
                v = np.zeros(xs.shape)

            else: 
                osd_flag = False
                self._session.run(self.setup,  feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
                #grad = self._session.run(self.grad_ce)
                grad = self._session.run(self.grad_cw)
                loss, stop_mask = self.loss_ce, self.stop_mask_ce
            """

            self._session.run(self.setup,  feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})

            grad = self._session.run(self.grad)
            loss, stop_mask = self.loss, self.stop_mask

            grad = grad.reshape(self.batch_size, *self.model.x_shape)
            loss, stop_mask = loss.eval(session=self._session), stop_mask.eval(session=self._session)
            print(i, "stop_mask", stop_mask.sum())

            # MI
            """
            grad = 0.75 * grad + 0.25 * prev_grad
            prev_grad = grad

            """
            m_ = m
            m = 0.9*m + 0.1* grad
            v = 0.99*v + 0.01 * (grad**2)
            grad = m / (np.sqrt(v)+1e-8)
            prev_grad = grad

            grad_sign = np.sign(grad)

            #grad_sign = grad / grad.mean(axis=(1,2,3))[:, None, None, None]
#            print("grad mean:{}, grad.max:{}, grad.min:{}, grad.abs.mean:{}, grad.std:{}, grad.abs.std:{}".format(grad.mean(), grad.max(), grad.min(), np.abs(grad).mean(), grad.std(), np.abs(grad).std()))

            xs_adv = np.clip(xs_adv + (self.alpha * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
        return xs_adv
