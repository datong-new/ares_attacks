import numpy as np
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph
from ares.loss import CrossEntropyLoss


class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        """ Based on ares.attack.bim.BIM, numpy version. """
        self.model, self.batch_size, self._session = model, batch_size, session
        # dataset == "imagenet" or "cifar10"
        if dataset == "imagenet":
            self.num_classes = 1000
        else:
            self.num_classes = 10

        # placeholder for batch_attack's inputvar
        self.xs_var = get_xs_ph(model, batch_size)
        self.ys_var = get_ys_ph(model, batch_size)
        self.visited_logits = tf.placeholder(self.model.x_dtype, [batch_size, None, self.num_classes])
        self.tf_w = tf.placeholder(self.model.x_dtype, [batch_size, self.num_classes])

        self.grad_ods, self.loss_ods, self.stop_mask_ods, self.logits_ods = self._get_gradients(loss_type="ods")
        self.grad_ce, self.loss_ce, self.stop_mask_ce, self.logits_ce = self._get_gradients(loss_type="ce")
        self.grad_cw, self.loss_cw, self.stop_mask_cw, self.logits_cw = self._get_gradients(loss_type="cw")
        #self.grad_kl, self.loss_kl, self.stop_mask_kl, self.logits_kl = self._get_gradients(loss_type="kl")
        self.grad_zmax, self.loss_zmax, self.stop_mask_zmax, self.logits_zmax = self._get_gradients(loss_type="md_zmax")
        self.grad_zy, self.loss_zy, self.stop_mask_zy, self.logits_zy = self._get_gradients(loss_type="md_zy")
        self.grad_y, self.loss_y, self.stop_mask_y, self.logits_y = self._get_gradients(loss_type="md_y")

        self.iteration = 100

    def init_delta(self):
        return (2 * np.random.uniform(size=self.xs_ph.shape) - 1) * self.eps

    def _get_gradients(self, loss_type="ce"):
        logits, label = self.model._logits_and_labels(self.xs_var)
        if loss_type == 'ce':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)
        elif loss_type == 'ods':
            loss = tf.reduce_sum(logits * self.tf_w, axis=-1)
        elif loss_type == 'cw':
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            label_score = tf.reduce_sum(mask * logits, axis=1)
            second_scores = tf.reduce_max((1 - mask) * logits - 1e4 * mask, axis=1)
            loss = -(label_score - second_scores)
            # ce
            # loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)

        elif loss_type =='md_zy':
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            loss = tf.reduce_sum(mask * logits, axis=1)
            #loss = tf.reduce_sum(mask * logits, axis=1)
        elif loss_type == 'md_y':
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            loss = -tf.reduce_sum(mask * logits, axis=1)
            # loss = tf.reduce_sum(mask * logits, axis=1)
        elif loss_type =='md_zmax':
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            loss = tf.reduce_max((1-mask) * logits - mask * 1e4, axis=1)
            #loss = tf.reduce_max((1 - mask) * logits, axis=1)


        grad = tf.gradients(loss, self.xs_var)[0]
        stop_mask = tf.cast(tf.equal(label, self.ys_var), dtype=tf.float32)
        return grad, loss, stop_mask, logits

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            self.alpha = self.eps / 7

    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        xs_adv = xs
        visted_logits = self._session.run(self.logits_ce, feed_dict={self.xs_var: xs_adv, self.ys_var: ys})
        visted_logits = visted_logits[:, None, :]
        self.alpha = self.eps
        self.num_restart=2

        for r in range(self.num_restart):
            xs_adv = xs
            k=int(self.iteration//self.num_restart)
            for i in range(k):
                if i<2 :
                    grad, loss, stop_mask, logits = self._session.run(
                        (self.grad_ods, self.loss_ods, self.stop_mask_ods, self.logits_ods),
                        feed_dict={self.xs_var: xs_adv, self.ys_var: ys,
                                   self.visited_logits: visted_logits,
                                   self.tf_w: 2 * np.random.uniform(size=(self.batch_size, self.num_classes)) - 1})
                elif i<3 :
                    grad, loss, stop_mask, logits = self._session.run(
                        (self.grad_zy, self.loss_zy, self.stop_mask_zy, self.logits_zy),
                        feed_dict={self.xs_var: xs_adv, self.ys_var: ys})
                elif i < k/2:
                    if r%2==0:
                        grad, loss, stop_mask, logits = self._session.run(
                            (self.grad_zmax, self.loss_zmax, self.stop_mask_zmax, self.logits_zmax),
                            feed_dict={self.xs_var: xs_adv, self.ys_var: ys})
                    elif r%2==1:
                        grad, loss, stop_mask, logits = self._session.run(
                            (self.grad_y, self.loss_y, self.stop_mask_y, self.logits_y),
                            feed_dict={self.xs_var: xs_adv, self.ys_var: ys})
                else:
                    #v1
                    grad, loss, stop_mask, logits = self._session.run(
                        (self.grad_cw, self.loss_cw, self.stop_mask_cw, self.logits_cw),
                        feed_dict={self.xs_var: xs_adv, self.ys_var: ys, self.visited_logits: visted_logits})

                if stop_mask[0] == 0: return xs_adv

                grad = grad.reshape(self.batch_size, *self.model.x_shape)
                #print(i, "stop_mask", stop_mask.sum())

                # MI
                #            grad = 0.75 * grad + 0.25 * prev_grad
                #            prev_grad = grad

                grad_sign = np.sign(grad)
                if i<3:
                    xs_adv = np.clip(xs_adv + (self.eps * 2 * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
                elif i<k/2:
                    xs_adv = np.clip(xs_adv + (self.alpha * 2 * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
                else:
                    xs_adv = np.clip(xs_adv + (self.alpha / 4 * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
                xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)

        return xs_adv