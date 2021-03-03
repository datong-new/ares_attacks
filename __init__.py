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
        self.restart_mask = tf.placeholder(self.model.x_dtype, [batch_size,])

        self.tf_w = tf.placeholder(self.model.x_dtype, [batch_size, self.num_classes])
        logits, label = self.model._logits_and_labels(self.xs_var)

        self.grad_ods, self.loss_ods= self._get_gradients(logits, label, loss_type="ods")
        self.grad_ce, self.loss_ce= self._get_gradients(logits, label, loss_type="ce")
        self.grad_cw, self.loss_cw= self._get_gradients(logits, label, loss_type="cw")
        self.grad_cos, self.loss_cos= self._get_gradients(logits, label, loss_type="cos")
        self.grad_dlr, self.loss_dlr= self._get_gradients(logits, label, loss_type="dlr")

        self.stop_mask, self.logits = tf.cast(tf.equal(label, self.ys_var), dtype=tf.float32), logits

        self.loss = self.loss_cos * self.restart_mask + self.loss_dlr * (1-self.restart_mask)
        self.grad = tf.gradients(self.loss, self.xs_var)[0]

        self.iteration = 100

    def init_delta(self):
        return (2 * np.random.uniform(size=self.xs_var.shape) - 1) * self.eps

    def _get_gradients(self, logits, label, loss_type="ce"):
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
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)
        elif loss_type == "kl":

            log_nature_logits = tf.nn.log_softmax(self.visited_logits, axis=-1)
            log_logits = tf.nn.log_softmax(logits, axis=-1)

            exp_nature_logits = tf.exp(log_nature_logits)
            neg_ent = tf.reduce_sum(exp_nature_logits * log_nature_logits, axis=-1)
            neg_cross_ent = tf.reduce_sum(exp_nature_logits * log_logits[:, None, :], axis=-1)
            kl_loss = neg_ent - neg_cross_ent
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)
            loss = kl_loss
        elif loss_type == "cos":
            # visited_scores = tf.nn.softmax(self.visited_logits, axis=-1)
            visited_scores = self.visited_logits
            visited_directions = visited_scores[:, 1:, :] - visited_scores[:, 0:1, :]
            visited_directions /= (tf.sqrt(tf.reduce_sum(visited_directions ** 2, axis=-1))[:, :, None] + 1e-8)

            # scores = tf.nn.softmax(logits, axis=-1)[:,None,:]
            scores = logits[:, None, :]
            current_direction = scores - visited_scores[:, 0:1, :]
            current_direction /= (tf.sqrt(tf.reduce_sum(current_direction ** 2, axis=-1))[:, :, None] + 1e-8)

            cos_dis = 1 - tf.reduce_mean(tf.reduce_sum(visited_directions * current_direction, axis=-1), axis=-1)
            loss = cos_dis

            log_nature_logits = tf.nn.log_softmax(self.visited_logits, axis=-1)
            log_logits = tf.nn.log_softmax(logits, axis=-1)
            exp_nature_logits = tf.exp(log_nature_logits)
            neg_ent = tf.reduce_sum(exp_nature_logits * log_nature_logits, axis=-1)
            neg_cross_ent = tf.reduce_sum(exp_nature_logits * log_logits[:, None, :], axis=-1)
            kl_loss = neg_ent - neg_cross_ent
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)
            loss += kl_loss
        elif loss_type == 'dlr':

            '''
            logits_sorted, ind_sorted = logits.sort(dim=1)
            ind = (ind_sorted[:, -1] == self.ys_var).float()
            dlr_loss = -(logits[np.arange(logits.shape[0]), self.ys_var] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (1. - ind)) / (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-12)
            '''
            logits_sort = tf.contrib.framework.sort(logits, axis=1)
            #y_onehot = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            dlr_loss = -(logits_sort[:, -1] - logits_sort[:, -2]) / (logits_sort[:, -1] - logits_sort[:, -3] + 1e-12)
            loss=dlr_loss

            #ce
        #    loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)
        grad = tf.gradients(loss, self.xs_var)[0]

        return grad, loss

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            self.alpha = self.eps / 7

    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        xs_adv = xs
        visted_logits = self._session.run(self.logits_ce, feed_dict={self.xs_var: xs_adv, self.ys_var: ys})
        visited_logits_list = []
        for i in range(self.batch_size): visited_logits_list+=[[visted_logits[i,:]]]
        max_len = 1
        prev_loss = np.ones((self.batch_size,)) * (-1e8)
        restart_count = np.zeros((self.batch_size,))

        def pad_list2numpy(visited_logits_list, max_len):
            new_list = []
            for i in range(self.batch_size):
                item = [[]+visited_logits_list[i]]
                while len(item)<max_len:
                    item += [np.array([0.]*self.num_classes)]
                new_list += [item]
            return np.array(new_list)

        for i in range(self.iteration):
            restart_mask = restart_count<5
            visted_logits_np = pad_list2numpy(visited_logits_list, max_len)

            if max_len<2:
                grad, loss, stop_mask, logits = self._session.run(
                    (self.grad_kl, self.loss_ods, self.stop_mask_ods, self.logits_ods),
                    feed_dict={self.xs_var: xs_adv, self.ys_var: ys,
                               self.visited_logits: visted_logits_np,
                               self.tf_w: 2 * np.random.uniform(size=(self.batch_size, self.num_classes)) - 1})
            else:
                grad, loss, stop_mask, logits = self._session.run(
                    (self.grad, self.loss, self.stop_mask, self.logits),
                    feed_dict={self.xs_var: xs_adv, self.ys_var: ys,
                               self.visited_logits: visted_logits_np,
                               self.restart_mask: restart_mask,
                               self.tf_w: 2 * np.random.uniform(size=(self.batch_size, self.num_classes)) - 1})

            loss_mask = (loss>prev_loss+1e-6)
            grad = grad.reshape(self.batch_size, *self.model.x_shape)
            grad_sign = np.sign(grad)
            #            print(i, "stop_mask", stop_mask.sum())
            # MI
            """
            grad = 0.75 * grad + 0.25 * prev_grad
            prev_grad = grad
            """
            # update step size for those with decreasing loss
            self.alpha = self.alpha * loss_mask + (self.alpha /2) * (1-loss_mask)
            ## self.alpha = self.eps for restart stage
            self.alpha = self.alpha * (1-restart_mask) + (self.eps)*restart_mask

            ## only update xs_adv with increasing loss
            xs_adv = np.clip(xs_adv + (self.alpha * stop_mask * loss_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
            prev_loss = loss * loss_mask + prev_loss * (1-loss_mask)

            # update restart count
            restart_count += 1
            alpha_mask = self.alpha < self.eps/16
            restart_count = restart_count*alpha_mask

            # update prev_loss if start to restart
            restart_count_equal_0 = (restart_count==0)
            prev_loss = prev_loss * (1-restart_count_equal_0) + (np.ones(prev_loss.shape)*(-1e8))

            for idx in alpha_mask.nonzero()[0]:
                visited_logits_list[idx] += [logits[idx, :]]
                if (visited_logits_list[idx])>max_len: max_len += 1

        return xs_adv
