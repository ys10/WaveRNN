# coding=utf-8
import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, data_loader, config, logger):
        self.sess = sess
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.logger = logger
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        """
        Train from current epoch to number of epochs in the config.
            Call train_epoch for each epoch, and increase cur_epoch_tensor.
        """
        tf.logging.info('Training...')
        # initialize data set
        self.sess.run([self.data_loader.data_set_init_ops['train']])
        for cur_epoch in range(self.config.epochs):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        Implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        Implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
