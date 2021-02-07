import numpy as np
import random
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph
from ares.loss import CrossEntropyLoss
from ares.loss.base import Loss

class MyLoss(Loss):
    def __init__(self, model):
        self.model = model

    def __call__(self, xs, ys):
        logits, label = self.model._logits_and_labels(xs)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys, logits=logits)
        #print("loss", loss)

        mask = tf.one_hot(ys, depth=tf.shape(logits)[1])
        label_score = tf.reduce_sum(mask*logits, axis=1)
        second_scores = tf.reduce_max((1- mask) * logits,  axis=1)

        top_scores = tf.math.top_k((1-mask)*logits, 2)[0]
        second = tf.reduce_max(top_scores,  axis=1)
        third = tf.reduce_min(top_scores,  axis=1)


        #loss_1 = -(label_score - second) / (label_score - third)
        loss_1 = -(label_score - second)
        #loss_2 = -(label_score - second_scores)

        stop_mask = tf.cast(tf.equal(label, ys), dtype=tf.float32)

        return loss_1, stop_mask



class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        """ Based on ares.attack.bim.BIM, numpy version. """
        self.model, self.batch_size, self._session = model, batch_size, session
        # dataset == "imagenet" or "cifar10"
        #loss = CrossEntropyLoss(self.model)
        loss = MyLoss(self.model)
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)
        self.xs_var = tf.Variable(tf.zeros(self.xs_ph.shape, dtype=self.model.x_dtype))
        self.ys_var = tf.Variable(tf.zeros(shape=(batch_size,), dtype=self.model.y_dtype))
        self.setup = [self.xs_var.assign(self.xs_ph),self.ys_var.assign(self.ys_ph)]

        self.loss, self.stop_mask = loss(self.xs_var, self.ys_var)
        self.grad = tf.gradients(self.loss, self.xs_var)[0]

        self.iteration = 100
        self.checkpoints = self.init_checkpoints()

    def init_checkpoints(self):
        prev_k = 0
        k = 0.22
        checkpoints = set()
        while k<1:
            checkpoints.add(int(k*self.iteration))
            next_k = k+max(k-prev_k-0.03, 0.06)
            prev_k = k
            k = next_k
        return checkpoints

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            self.alpha = self.eps * 2 * np.ones((self.batch_size,))
#            self.alpha = self.eps /7 * np.ones((self.batch_size,))

    def init_delta(self, batch_size):
        x = np.zeros(self.model.x_shape)
        for i in range(x.shape[0]):
            c = [1,1,1]
            for j in range(3):
                if random.uniform(0,1)<0.5: c[j] = -c[j]
            c = np.array(c)
            x[i, :, :] += c
        x = np.ones((batch_size, 1, 1, 1)) * x

        return x * self.eps



    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        xs_adv = xs
        adv_best = xs_adv

        loss_prev = np.zeros((self.batch_size, ))
        loss_best = np.zeros((self.batch_size, ))
        total = 0
        loss_inc_num = np.zeros((self.batch_size, ))

        prev_grad = np.zeros(xs.shape)
        self.alpha = self.eps * np.ones((self.batch_size,))

        for i in range(self.iteration):

            """
            if i in self.checkpoints:
                self.alpha, xs_adv=self.update_alpha(total, loss_inc_num, adv_best, xs_adv)

                adv_best = xs_adv
                loss_best = np.zeros((self.batch_size, ))
                total = 0
                loss_inc_num = np.zeros((self.batch_size, ))
            """
            if i%10==9:
                print("stop_mask", stop_mask)
                xs_adv = xs_adv * ((1-stop_mask)[:, None, None, None]) + (xs+self.init_delta(self.batch_size)) * stop_mask[:, None, None, None]

            self._session.run(self.setup,  feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
            #grad = self._session.run(self.grad, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
            grad = self._session.run(self.grad)
            grad = grad.reshape(self.batch_size, *self.model.x_shape)
            loss, stop_mask = self.loss.eval(session=self._session), self.stop_mask.eval(session=self._session)
            print(i, "stop_mask", stop_mask.sum())

            #
            #loss_inc_num += loss>loss_prev
            #loss_prev = loss
            #total += 1

            # update loss_best and adv_best
            #mask = loss>loss_best
            #loss_best = loss_best * (1-mask) + loss * mask
            #adv_best = adv_best * (1-mask)[:, None, None, None] + xs_adv * mask[:, None, None, None]

            grad = 0.75 * grad + 0.25 * prev_grad
            prev_grad = grad
            grad_sign = np.sign(grad)

            xs_adv = np.clip(xs_adv + (self.alpha * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
        return xs_adv
