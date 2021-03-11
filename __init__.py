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
        self.grad_kl, self.loss_kl, self.stop_mask_kl, self.logits_kl = self._get_gradients(loss_type="kl")
        self.grad_zmax, self.loss_zmax, self.stop_mask_zmax, self.logits_zmax = self._get_gradients(loss_type="md_zmax")
        self.grad_zy, self.loss_zy, self.stop_mask_zy, self.logits_zy = self._get_gradients(loss_type="md_zy")

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
            label_score = tf.reduce_sum(mask * logits-(1-mask)*1e5, axis=1)
            second_scores = tf.reduce_max((1 - mask) * logits - 1e5 * mask, axis=1)
            loss = -(label_score - second_scores)
            # ce
            # loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)
        elif loss_type == "kl":
            log_nature_logits = tf.nn.log_softmax(self.visited_logits, axis=-1)
            log_logits = tf.nn.log_softmax(logits, axis=-1)
            exp_nature_logits = tf.exp(log_nature_logits)
            neg_ent = tf.reduce_sum(exp_nature_logits * log_nature_logits, axis=-1)
            neg_cross_ent = tf.reduce_sum(exp_nature_logits * log_logits[:, None, :], axis=-1)
            kl_loss = neg_ent - neg_cross_ent
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)
            loss = kl_loss
        elif loss_type =='md_zy':
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            loss = tf.reduce_sum(mask * logits-(1-mask)*1e5, axis=1)
        elif loss_type =='md_zmax':
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            loss = tf.reduce_max((1-mask) * logits - mask * 1e5, axis=1)



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
        restart_count = 0
        prev_loss = -1e8
        self.alpha = self.eps

        for i in range(self.iteration):
            # print("visted_logits.shape", visted_logits.shape)
            # if i%30==0 or i%30==10: prev_grad=0
            if restart_count == 0:
                self.alpha, prev_grad = self.eps, 0

            if restart_count == 30:
                xs_prev = xs_adv
                self.alpha, prev_grad = self.eps / 7, 0
            if restart_count < 30:
                if i%30< 1:  # do ods first
                    grad, loss, stop_mask, logits = self._session.run(
                        (self.grad_zy, self.loss_zy, self.stop_mask_zy, self.logits_zy),
                        feed_dict={self.xs_var: xs_adv, self.ys_var: ys,
                                   self.visited_logits: visted_logits})
                elif i%30 < 20:
                    grad, loss, stop_mask, logits = self._session.run(
                        (self.grad_zmax, self.loss_zmax, self.stop_mask_zmax, self.logits_zmax),
                        feed_dict={self.xs_var: xs_adv, self.ys_var: ys, self.visited_logits: visted_logits})
                else:
                    #v1
                    grad, loss, stop_mask, logits = self._session.run(
                        (self.grad_cw, self.loss_cw, self.stop_mask_cw, self.logits_cw),
                        feed_dict={self.xs_var: xs_adv, self.ys_var: ys, self.visited_logits: visted_logits})
                    #v2
                    '''
                    grad, loss, stop_mask, logits = self._session.run(
                        (self.grad_kl, self.loss_kl, self.stop_mask_kl, self.logits_kl),
                        feed_dict={self.xs_var: xs_adv, self.ys_var: ys, self.visited_logits: visted_logits})
                    '''

            #                print("loss_kl", loss[:10])

            else:
                grad, loss, stop_mask, logits = self._session.run(
                    (self.grad_cw, self.loss_cw, self.stop_mask_cw, self.logits_cw),
                    feed_dict={self.xs_var: xs_adv, self.ys_var: ys, self.visited_logits: visted_logits})
            #                print("loss_cw", loss[:10])
            # if i%30==29: # save visited logits
            #    visted_logits = np.concatenate((visted_logits, logits[:,None,:]), axis=1)

            if stop_mask[0] == 0: return xs_adv

            if loss[0] < prev_loss + 1e-3 and restart_count >= 30:
                xs_adv = xs_prev
                #print(i, "decay step size, loss:{}, prev_loss:{}, alpha:{}".format(loss[0], prev_loss, self.alpha))
                self.alpha /= 2
                if self.alpha < self.eps / 32:
                    restart_count = 0
                    self.alpha = self.eps
                    prev_loss = -1e8
                    visted_logits = np.concatenate((visted_logits, logits[:, None, :]), axis=1)
                continue
            #print(i, "loss:{}, prev_loss:{}, alpha:{}".format(loss[0], prev_loss, self.alpha))
            if restart_count >= 30: prev_loss = loss[0]

            grad = grad.reshape(self.batch_size, *self.model.x_shape)
            print(i, "stop_mask", stop_mask.sum())

            # MI
            #            grad = 0.75 * grad + 0.25 * prev_grad
            #            prev_grad = grad

            grad_sign = np.sign(grad)
            xs_prev = xs_adv
            if i%30<1:
                xs_adv = np.clip(xs_adv + (self.eps*2 * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
            elif i % 30 < 20:
                xs_adv = np.clip(xs_adv + (self.alpha * 2 * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
            else:
                xs_adv = np.clip(xs_adv + (self.alpha * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
            restart_count += 1

        return xs_adv