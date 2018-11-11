# coding=utf-8
import tensorflow as tf


class BaseValidate:
    def __init__(self, sess, model, data_loader, config, logger):
        self.sess = sess
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.logger = logger
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def validate(self):
        """
        Predict whole data set(one epoch).
        """
        tf.logging.info('Validating...')
        # initialize data set
        self.sess.run([self.data_loader.data_set_init_ops['validate']])
        self.validate_epoch()
        tf.logging.info('Validation done.')

    def validate_epoch(self):
        """
        Implement the logic of epoch:
        -loop over the number of iterations in the config and call the validate step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def validate_step(self):
        """
        Implement the logic of the validate step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
