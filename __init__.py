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
        if dataset=="imagenet": 
            self.num_classes=1000
        else:
            self.num_classes=10


        self.universal_perturbation_ph = tf.placeholder(self.model.x_dtype, self.xs_ph.shape[1:])
        self.universal_perturbation = tf.Variable(tf.zeros(shape=self.xs_ph.shape[1:], dtype=self.model.x_dtype))
        self.setup_universal_perturbation = [self.xs_var.assign(self.xs_ph),
                self.ys_var.assign(self.ys_ph), 
                self.universal_perturbation.assign(self.universal_perturbation_ph)]
        self.grad_perturbation = self._get_gradient_universal_perturbation()


        self.tf_w_ph = tf.placeholder(self.model.x_dtype, (batch_size, self.num_classes))
        self.tf_w = tf.Variable(tf.zeros(shape=(batch_size, self.num_classes), dtype=self.model.x_dtype))

        self.setup = [self.xs_var.assign(self.xs_ph),self.ys_var.assign(self.ys_ph)]
        self.setup_tf_w = self.tf_w.assign(self.tf_w_ph)

        self.grad_ods, self.loss_ods, self.stop_mask_ods = self._get_gradients(loss_type="ods")
        self.grad_ce, self.loss_ce, self.stop_mask_ce = self._get_gradients(loss_type="ce")
        self.grad_cw, self.loss_cw, self.stop_mask_cw = self._get_gradients(loss_type="cw")

        self.iteration = 80

        self.perturbations=None

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            self.alpha = self.eps /7 

    def init_delta(self):
        return (2*np.random.uniform(size=self.xs_ph.shape)-1) * self.eps

    def _get_gradients(self, loss_type="ce"):
        logits, label = self.model._logits_and_labels(self.xs_var)
        if loss_type=='ce':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)
        elif loss_type=='ods':
            loss = tf.reduce_sum(logits*self.tf_w, axis=-1)
        elif loss_type=='cw':
            logits *= 10
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            label_score = tf.reduce_sum(mask*logits, axis=1)
            second_scores = tf.reduce_max((1- mask) * logits,  axis=1)
            loss = -(label_score - second_scores)

            # ce
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)


        grad = tf.gradients(loss, self.xs_var)[0]
        stop_mask = tf.cast(tf.equal(label, self.ys_var), dtype=tf.float32)
        return grad, loss, stop_mask

    def _get_gradient_universal_perturbation(self):
        logits, label = self.model._logits_and_labels(self.xs_var+self.universal_perturbation[None, :,:,:])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)
        grad = tf.gradients(loss, self.universal_perturbation)[0]
        return grad




    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps

        prev_grad = np.zeros(xs.shape)
        stop_mask = None

        if self.perturbations is None:
            print("self.perturbations is None")
            self.perturbations = np.zeros(xs.shape[1:])

        for i in range(10):
            self._session.run(self.setup_universal_perturbation,  feed_dict={self.xs_ph: xs, self.ys_ph: ys,
                self.universal_perturbation_ph:self.perturbations})
            grad = self._session.run(self.grad_perturbation)
            self.perturbations = self.perturbations+np.sign(grad)
            self.perturbations = np.clip(self.perturbations, -self.eps, self.eps)

        xs_adv = xs + self.perturbations[None, :, :, :] 
        for i in range(self.iteration):
            #if i%20==0 or i%20==1:
            if i%80<3:
                if i%80==0:
                    self._session.run(self.setup_tf_w, feed_dict={self.tf_w_ph: 2*np.random.uniform(size=(self.batch_size, self.num_classes))-1})
                    if stop_mask is not None:
                        xs_adv = xs_adv * (1-stop_mask[:, None, None, None]) + (xs+self.init_delta()) * (stop_mask[:, None, None, None])

                self._session.run(self.setup,  feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
                grad = self._session.run(self.grad_ods)
                loss, stop_mask = self.loss_ods, self.stop_mask_ods
            else: 
                self._session.run(self.setup,  feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
                grad = self._session.run(self.grad_cw)
                #grad = self._session.run(self.grad_ce)
                loss, stop_mask = self.loss_cw, self.stop_mask_cw

            grad = grad.reshape(self.batch_size, *self.model.x_shape)
            loss, stop_mask = loss.eval(session=self._session), stop_mask.eval(session=self._session)
            print(i, "stop_mask", stop_mask.sum())

            # MI
            """
            grad = 0.75 * grad + 0.25 * prev_grad
            prev_grad = grad
            """

            grad_sign = np.sign(grad)
            xs_adv = np.clip(xs_adv + (self.alpha * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
        return xs_adv
