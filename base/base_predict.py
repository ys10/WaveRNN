# coding=utf-8
import tensorflow as tf


class BasePredict:
    def __init__(self, sess, model, data_loader, config, logger):
        self.sess = sess
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.logger = logger
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def predict(self):
        """
        Predict whole data set(one epoch).
        """
        tf.logging.info('Predicting...')
        # initialize data set
        self.sess.run([self.data_loader.data_set_init_ops['predict']])
        self.predict_epoch()

    def predict_epoch(self):
        """
        Implement the logic of predict epoch:
        -loop over the number of iterations in the config and call the predict step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def predict_step(self):
        """
        Implement the logic of the predict step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
