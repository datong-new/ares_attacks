import numpy as np
import random
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array
from ares.loss.base import Loss 
from ares.loss import CrossEntropyLoss

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
        loss_1 = -(label_score - second_scores)

        stop_mask = tf.cast(tf.equal(label, ys), dtype=tf.float32)

        return loss_1, stop_mask

class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        ''' Based on ares.attack.bim.BIM '''
        self.model, self.batch_size, self._session = model, batch_size, session
        # dataset == "imagenet" or "cifar10"
        #loss = CrossEntropyLoss(self.model)
        loss = MyLoss(self.model)
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.xs_adv_var_ph = get_xs_ph(model, batch_size)
        self.prev_grad_ph = get_xs_ph(model, batch_size)


        self.ys_ph = get_ys_ph(model, batch_size)
        # flatten shape of xs_ph
        xs_flatten_shape = (batch_size, np.prod(self.model.x_shape))
        # store xs and ys in variables to reduce memory copy between tensorflow and python
        # variable for the original example with shape of (batch_size, D)
        self.xs_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        self.prev_grad = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))

        # variable for labels
        self.ys_var = tf.Variable(tf.zeros(shape=(batch_size,), dtype=self.model.y_dtype))
        # variable for the (hopefully) adversarial example with shape of (batch_size, D)
        self.xs_adv_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        # magnitude
        self.eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # step size
        self.alpha_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.alpha_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # expand dim for easier broadcast operations
        eps = tf.expand_dims(self.eps_var, 1)
        alpha = tf.expand_dims(self.alpha_var, 1)
        # calculate loss' gradient with relate to the adversarial example
        # grad.shape == (batch_size, D)
        xs_lo, xs_hi = self.xs_var - eps, self.xs_var + eps


        self.xs_adv_model = tf.reshape(self.xs_adv_var, (batch_size, *self.model.x_shape))

        self.loss, self.stop_mask = loss(self.xs_adv_model, self.ys_var)
        self.stop_mask = tf.cast(self.stop_mask, dtype=tf.float32)
        self.stop_mask = tf.expand_dims(self.stop_mask, axis=1)



        self.grad = tf.gradients(self.loss, self.xs_adv_var)[0]
        self.update_prev_grad = self.prev_grad.assign(self.grad)
        # update the adversarial example
        grad_sign = tf.sign(self.grad)
        # clip by max l_inf magnitude of adversarial noise

        xs_adv_next = tf.clip_by_value(self.xs_adv_var + alpha * self.stop_mask * grad_sign, xs_lo, xs_hi)
        #xs_adv_next = tf.clip_by_value(self.xs_adv_var + alpha * grad_sign, xs_lo, xs_hi)


        # clip by (x_min, x_max)
        xs_adv_next = tf.clip_by_value(xs_adv_next, self.model.x_min, self.model.x_max)

        self.update_xs_adv_step = self.xs_adv_var.assign(xs_adv_next)
        self.config_eps_step = self.eps_var.assign(self.eps_ph)
        self.config_alpha_step = self.alpha_var.assign(self.alpha_ph)

        self.delta = self.init_delta(batch_size)
        self.setup_xs = [self.xs_var.assign(tf.reshape(self.xs_ph, xs_flatten_shape)),
                         self.xs_adv_var.assign(tf.reshape(self.xs_adv_var_ph, xs_flatten_shape))]

        self.setup_prev_grad = self.prev_grad.assign(tf.reshape(self.prev_grad_ph, xs_flatten_shape))
        self.setup_ys = self.ys_var.assign(self.ys_ph)
        self.iteration = 30

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            eps = maybe_to_array(self.eps, self.batch_size)
            self._session.run(self.config_eps_step, feed_dict={self.eps_ph: eps})
            self._session.run(self.config_alpha_step, feed_dict={self.alpha_ph: eps / 7})

    def init_delta(self, batch_size):
        x = np.zeros(self.model.x_shape)
        for i in range(x.shape[0]):
            c = [1,1,1]
            for j in range(3):
                if random.uniform(0,1)<0.5: c[j] = -c[j]
            c = np.array(c)
            x[i, :, :] += c
        x = np.ones((batch_size, 1, 1, 1)) * x

        return x


    def batch_attack(self, xs, ys=None, ys_target=None):
        self._session.run(self.setup_xs, feed_dict={self.xs_ph: xs, self.xs_adv_var_ph: xs+self.delta})
        self._session.run(self.setup_ys, feed_dict={self.ys_ph: ys})
        self._session.run(self.setup_prev_grad, feed_dict={self.prev_grad_ph: np.zeros(xs.shape)})
        for i in range(self.iteration):
            #print("self.prev_grad", self.prev_grad)
            #print("self.prev_grad", tf.reduce_sum(self.prev_grad).eval(session=self._session))

            self._session.run(self.update_xs_adv_step)

            #print("self.prev_grad1", self.prev_grad)
            #print("self.prev_grad1", tf.reduce_sum(self.prev_grad).eval(session=self._session))

            self._session.run(self.update_prev_grad)
            #print("self.prev_grad2", self.prev_grad)
            #print("self.prev_grad2", tf.reduce_sum(self.prev_grad).eval(session=self._session))
            # for test
            """

            xs_ph = tf.placeholder(model.x_dtype, shape=(None, *model.x_shape))
            labels_op = model.labels(xs_ph)
            xs_adv = self._session.run(self.xs_adv_model)
            xs_adv = np.clip(xs_adv, xs - self.eps, xs+self.eps)
            xs_adv = np.clip(xs_adv, model.x_min, model.x_max)
            assert not np.any(np.isnan(xs_adv))
            labels = self._session.run(labels_op, feed_dict={xs_ph: xs_adv})
            print("in score", np.sum(np.logical_not(np.equal(labels, ys))))
            """


            print(i,"stop_mask", tf.reduce_sum(self.stop_mask).eval(session=self._session))
            #print("self.grad", self.grad)
            #print("self.grad", tf.reduce_sum(self.grad).eval(session=self._session))


        return self._session.run(self.xs_adv_model)
