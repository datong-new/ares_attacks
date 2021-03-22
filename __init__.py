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
        self.ods_mask = tf.placeholder(self.model.x_dtype, [batch_size, ])
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
        self.loss_cw = self.loss_zmax + self.loss_zy
        self.grad_kl = tf.gradients(self.loss_kl, self.xs_var)[0]
        self.loss_restart = self.ods_mask * self.loss_ods + (1-self.ods_mask) * self.loss_kl

        # attack loss
        self.loss_attack = self.lambda_ph * self.loss_zy + (1-self.lambda_ph) * self.loss_zmax + self.loss_kl
        #self.loss_attack = self.lambda_ph * self.loss_zy + (1-self.lambda_ph) * self.loss_zmax
        self.grad_attack = tf.gradients(self.loss_attack, self.xs_var)[0]


        self.loss = self.restart_mask * self.loss_restart + (1-self.restart_mask) * self.loss_attack
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
            self.visited_logits_mask = tf.cast(
                    tf.not_equal(
                        tf.reduce_sum(self.visited_logits, axis=-1), 
                        tf.constant(self.num_classes, dtype=tf.float32)), 
                dtype=tf.float32)

            log_nature_logits=tf.nn.log_softmax(self.visited_logits, axis=-1)
            log_logits=tf.nn.log_softmax(logits, axis=-1)

            exp_nature_logits=tf.exp(log_nature_logits)
            neg_ent = tf.reduce_sum(exp_nature_logits* log_nature_logits, axis=-1)
            neg_cross_ent = tf.reduce_sum(exp_nature_logits * log_logits[:, None, :], axis=-1)
            kl_loss = neg_ent - neg_cross_ent
            loss = tf.reduce_sum(kl_loss*self.visited_logits_mask, axis=-1) / tf.reduce_sum(self.visited_logits_mask, axis=-1)
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
            self.iteration = 100

    def batch_attack(self, xs, ys=None, ys_target=None):

        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        ys_cp = ys.copy()
        xs_lo_cp, xs_hi_cp = xs_lo.copy(), xs_hi.copy()
        xs_adv = xs.copy()
        #visited_logits = self._session.run(self.logits, feed_dict={self.xs_var: xs_adv, self.ys_var: ys})
        #visited_logits = visted_logits[:, None, :]

        round_num = 20
        return_xs_adv = xs.copy()
        restart_count = np.zeros(self.batch_size)
        id2img = [i for i in range(self.batch_size)]
        img2ids = [[i] for i in range(self.batch_size)]
        tf_w = 2*np.random.uniform(size=(self.batch_size, self.num_classes))-1
        fail_set = set(list(range(self.batch_size)))

        m = np.zeros(xs.shape)
        v = np.zeros(xs.shape)
        prev_grad = np.zeros(xs.shape)
        self.alpha = np.ones(self.batch_size)

        original_logits = self._session.run(self.logits, feed_dict={self.xs_var: xs_adv, self.ys_var: ys})
        visited_logits = np.ones((self.batch_size, 20, self.num_classes))
        visited_logits_list = [[original_logits[i].copy()] for i in range(self.batch_size)]
        loss_prev = np.ones(self.batch_size) * -1e8

        for i in range(self.iteration):
            ods_mask = np.zeros(self.batch_size, dtype=np.float32)

            for k in range(self.batch_size):
                #if restart_count[k]==3 or restart_count[k]==0:
                if restart_count[k]<3:
                    ods_mask[k] = 1
                    if restart_count[k]==0:
                        tf_w[k] = 2*np.random.uniform(size=(self.num_classes))-1
                if (restart_count%round_num)[k]<=3:
                    m[k] = np.zeros(xs.shape[1:])
                    v[k] = np.zeros(xs.shape[1:])
                    prev_grad[k] = np.zeros(xs.shape[1:])
                    """
                    if restart_count[k]<3:
                        self.alpha[k] = self.eps * 2
                    else:
                        self.alpha[k] = self.eps / 2
                    """
                visited_logits[k, :len(visited_logits_list[k]), :] = np.array(visited_logits_list[k])
            visited_logits = visited_logits.astype(np.float32)

            restart_mask = ((restart_count%round_num<3) * 1.0).astype(np.float32)

            #self.alpha = self.eps * 2 * restart_mask + self.eps / 4 * (1-restart_mask)

            grad, loss, stop_mask, logits, loss_cw, logits_mask  = self._session.run(
               (self.grad, self.loss, self.stop_mask, self.logits, self.loss_cw, self.visited_logits_mask),
               #(self.grad_kl, self.loss_kl, self.stop_mask, self.logits),
               feed_dict={self.xs_var: xs_adv, self.ys_var: ys_cp, 
                       #self.visited_logits:visited_logits, 
                       self.tf_w:tf_w,
                       self.restart_mask: restart_mask,
                       self.ods_mask: ods_mask,
                       self.lambda_ph: np.random.uniform(size=(self.batch_size,)),
                       self.visited_logits:visited_logits, 
                       })

            #print("logits_mask", logits_mask)


            loss_delta = loss_cw - loss_prev
            prev_loss = loss_cw
            for k in range(self.batch_size):
                #if restart_count[k]>3 and loss_delta[k]<=0:
                #if restart_count[k]>3 and loss_delta[k]<=1e-4:
                if (restart_count[k]+1) % round_num==0:
                    visited_logits_list[k] += [logits[k]]

            free_ids = []
            for idx in (1-stop_mask).nonzero()[0]:
                img = id2img[idx]
                loss_cw[idx] = -1e-8
                if img in fail_set:
                    fail_set.remove(img) # one img may successs attack in different ids
                    return_xs_adv[img] = xs_adv[idx].copy()

                    free_ids += img2ids[img]


            #grad_sign = np.sign(grad)
            """
            grad_sign = np.sign(grad) * (3**(np.sign(prev_grad)*np.sign(grad)))
            prev_grad = grad
            """

            m = 0.9*m+0.1*grad
            m/=0.9
            v = 0.99*v + 0.01*(grad**2)
            v/=0.99
            grad = m / (np.sqrt(v)+1e-8)

            max_ = np.max(np.abs(grad), axis=(1,2,3))
            min_ = np.min(np.abs(grad), axis=(1,2,3))
            max_alpha, min_alpha = self.eps*2, self.eps/4
            a = (max_alpha-min_alpha) / (max_-min_+1e-6)
            b = (min_*max_alpha-max_*min_alpha) / (min_-max_-1e-6)
            self.alpha = 1
            grad_sign = a[:,None, None, None]*grad + b[:,None, None, None]



            xs_adv = np.clip(xs_adv + (self.alpha * stop_mask)[:, None, None, None] * grad_sign, xs_lo_cp, xs_hi_cp)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)

            if len(fail_set)==0: break

            sort_idx = np.argsort(-loss_cw)
            selected_idx = sort_idx[:3]

            for free_id in free_ids:
                # random select a img
                rand_copy_id = selected_idx[random.randint(0, len(selected_idx)-1)]
                rand_img = id2img[rand_copy_id]

                #rand_img = list(fail_set)[random.randint(0, len(fail_set)-1)]
                #rand_copy_id = img2ids[rand_img][random.randint(0, len(img2ids[rand_img])-1)]

                xs_adv[free_id] = xs_adv[rand_copy_id].copy() 
                #xs_adv[free_id] = xs[rand_img].copy()
                ys_cp[free_id] = ys[rand_img].copy()
                xs_lo_cp[free_id] = xs_lo[rand_img].copy()
                xs_hi_cp[free_id] = xs_hi[rand_img].copy()

                visited_logits_list[free_id] = [original_logits[rand_img].copy()]
                visited_logits[free_id] = np.ones((20, self.num_classes))

                id2img[free_id] = rand_img
                img2ids[rand_img] += [free_id]

                #restart_count[free_id] = round_num
                restart_count[free_id] = 0
            #print(i, "fail len", len(fail_set))

            """
            stop_mask_, labels =  self._session.run(
                       (self.stop_mask, self.label),  feed_dict={self.xs_var:return_xs_adv,
                           self.ys_var: ys}
                       )
            print("stop_mask_", stop_mask_.sum())
            print("score", np.sum(np.logical_not(np.equal(labels, ys))))
            """

            restart_count += 1


        return return_xs_adv

