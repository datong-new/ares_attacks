import numpy as np
import random
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph
from ares.loss import CrossEntropyLoss


class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        """ Based on ares.attack.bim.BIM, numpy version. """
        self.model, self.batch_size, self._session = model, batch_size, session
        # dataset == "imagenet" or "cifar10"
        if dataset=="imagenet":
            self.num_classes=1000
        else:
            self.num_classes=10
        
        # placeholder for batch_attack's inputvar
        self.xs_var = get_xs_ph(model, batch_size)
        self.ys_var = get_ys_ph(model, batch_size)
        self.restart_mask = tf.placeholder(self.model.x_dtype, [batch_size, ])
        self.visited_logits = tf.placeholder(self.model.x_dtype, [batch_size, None, self.num_classes])
        self.tf_w = tf.placeholder(self.model.x_dtype, [batch_size, self.num_classes])

        self.lambda_ph = tf.placeholder(self.model.x_dtype, [batch_size, ])
        self.logits, self.label = self.model._logits_and_labels(self.xs_var)
        # restart loss
        self.loss_ods = self._get_loss(self.logits, self.label, loss_type="ods")
        self.grad_ods = tf.gradients(self.loss_ods, self.xs_var)[0]

        self.loss_zy = self._get_loss(self.logits, self.label, loss_type="z_y")
        self.loss_zmax = self._get_loss(self.logits, self.label, loss_type="z_max")
        self.loss_kl = self._get_loss(self.logits, self.label, loss_type="kl")
        self.grad_kl = tf.gradients(self.loss_kl, self.xs_var)[0]

        # attack loss
        #self.loss_attack = self.lambda_ph * self.loss_zy + (1-self.lambda_ph) * self.loss_zmax + self.loss_kl
        self.loss_attack = self.lambda_ph * self.loss_zy + (1-self.lambda_ph) * self.loss_zmax
        self.grad_attack = tf.gradients(self.loss_attack, self.xs_var)[0]

        self.loss = self.restart_mask * self.loss_ods + (1-self.restart_mask) * self.loss_attack
        self.grad = tf.gradients(self.loss, self.xs_var)[0]

        self.stop_mask = tf.cast(tf.equal(self.label, self.ys_var), dtype=tf.float32)

    def init_delta(self):
        return (2*np.random.uniform(size=self.xs_var.shape)-1) * self.eps

    def _get_loss(self, logits, label, loss_type="ce"):
        if loss_type=='ce':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)
        elif loss_type=='ods':
            loss = tf.reduce_sum(logits*self.tf_w, axis=-1)
        elif loss_type=='cw':
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            label_score = tf.reduce_sum(mask*logits, axis=1)
            second_scores = tf.reduce_max((1- mask) * logits - 1e4*mask,  axis=1)
            loss = -(label_score - second_scores)
            # ce
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_var, logits=logits)
        elif loss_type=="kl":
            log_nature_logits=tf.nn.log_softmax(self.visited_logits, axis=-1)
            log_logits=tf.nn.log_softmax(logits, axis=-1)

            exp_nature_logits=tf.exp(log_nature_logits)
            neg_ent = tf.reduce_sum(exp_nature_logits* log_nature_logits, axis=-1)
            neg_cross_ent = tf.reduce_sum(exp_nature_logits * log_logits[:, None, :], axis=-1)
            kl_loss = neg_ent - neg_cross_ent
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)
            loss = kl_loss
        elif loss_type=="emd":

            tmp = tf.square((tf.cumsum(tf.nn.softmax(self.visited_logits, axis=-1), axis=-1) - tf.cumsum(tf.nn.softmax(logits[:, None, :], axis=-1), axis=-1)))
            loss = tf.reduce_mean(tmp, axis=[1,2])
            #tmp = self.visited_logits - logits[:, None, :] # B x L x num_classes
            #loss = tf.reduce_mean(tf.abs(tmp), axis=[1,2])
        elif loss_type=='z_y':
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            label_score = tf.reduce_sum(mask*logits, axis=1)
            second_scores = tf.reduce_max((1- mask) * logits - 1e4*mask,  axis=1)
            loss = -label_score
        elif loss_type=='z_max':
            mask = tf.one_hot(self.ys_var, depth=tf.shape(logits)[1])
            label_score = tf.reduce_sum(mask*logits, axis=1)
            second_scores = tf.reduce_max((1- mask) * logits - 1e4*mask,  axis=1)
            loss = second_scores

        return loss

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            self.alpha = self.eps / 7
            self.iteration = 20

    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        xs_adv = xs
        #visted_logits = self._session.run(self.logits, feed_dict={self.xs_var: xs_adv, self.ys_var: ys})
        #visted_logits = visted_logits[:, None, :]

        round_num = 25
        return_xs_adv = [None for _ in range(self.batch_size)]
        restart_count = np.zeros(self.batch_size)
        id2img = [i for i in range(self.batch_size)]
        img2ids = [[i] for i in range(self.batch_size)]
        tf_w = 2*np.random.uniform(size=(self.batch_size, self.num_classes))-1
        fail_set = set(list(range(self.batch_size)))

        for i in range(self.iteration):
            restart_count %= round_num
            restart_mask = ((restart_count<3) * 1.0).astype(np.float32)

            self.alpha = self.eps * 2 * restart_mask + self.eps / 4 * (1-restart_mask)

            grad, loss, stop_mask, logits  = self._session.run(
               (self.grad, self.loss, self.stop_mask, self.logits),
               #(self.grad_kl, self.loss_kl, self.stop_mask, self.logits),
               feed_dict={self.xs_var: xs_adv, self.ys_var: ys, 
                       #self.visited_logits:visted_logits, 
                       self.tf_w:tf_w,
                       self.restart_mask: restart_mask,
                       self.lambda_ph: np.random.uniform(size=(self.batch_size,)),

                       })

            free_ids = []

            for idx in stop_mask.nonzero()[0]:
                img = id2img[idx]
                print("img", img)
                if img in fail_set: fail_set.remove(img) # one img may successs attack in different ids
                free_ids += img2ids[img]
                return_xs_adv[img] = xs_adv[idx]

            grad_sign = np.sign(grad)

            xs_adv = np.clip(xs_adv + (self.alpha * stop_mask)[:, None, None, None] * grad_sign, xs_lo, xs_hi)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
            print("fali_set", len(fail_set))

            if len(fail_set)==0: break

            for free_id in free_ids:
                # random select a img
                rand_img = list(fail_set)[random.randint(0, len(fail_set)-1)]
                rand_copy_id = img2ids[rand_img][random.randint(0, len(img2ids[rand_img])-1)]
                xs_adv[free_id] = xs_adv[rand_copy_id]
                ys[free_id] = ys[rand_copy_id]
                xs_lo[free_id] = xs_lo[rand_copy_id]
                xs_hi[free_id] = xs_hi[rand_copy_id]


                id2img[free_id] = rand_img
                img2ids[rand_img] += [free_id]

                restart_count[free_id] = 0
                tf_w[free_id] = 2*np.random.uniform(size=(self.num_classes))-1
            restart_count += 1
        for i in range(len(return_xs_adv)):
            if return_xs_adv[i] is None:
                return_xs_adv[i] = xs_adv[i]

        print("return_xs_adv", np.array(return_xs_adv).max())
        print("return_xs_adv", np.array(return_xs_adv).min())

        return np.array(return_xs_adv)



