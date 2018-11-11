# coding=utf-8
import os
import tensorflow as tf


class BaseModel:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.init_global_step()
        self.init_cur_epoch()

    def reset_data(self, data):
        """
        Reset input data.
        :param data: a nested structure of tf.Tensors representing the next element of data set.
            e.g.
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            model.set_data(next_element)
        """
        self.data = data

    def save(self, sess):
        """
        Save function that saves the checkpoint in the path defined in the config file.
        :param sess: current session
        """
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        tf.logging.info("Saving model to {}...".format(self.config.checkpoint_dir))
        global_step = self.global_step_tensor.eval(sess)
        tf.logging.info("  Global step was: {}".format(global_step))
        self.saver.save(sess, self.config.checkpoint_dir, global_step)
        tf.logging.info("Model saved")

    def load(self, sess):
        """
        Load latest checkpoint from the experiment path defined in the config file.
        :param sess: current session
        :return:
            global_step: int
        """
        tf.logging.info("Trying to restore saved checkpoints from {} ...".format(self.config.checkpoint_dir))
        ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
        if ckpt:
            tf.logging.info("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
            global_step = int(ckpt.model_checkpoint_path
                              .split('/')[-1]
                              .split('-')[-1])
            tf.logging.info("  Global step was: {}".format(global_step))
            tf.logging.info("  Restoring...")
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            tf.logging.debug('global_step_tensor_value: {}.'.format(self.global_step_tensor.eval(sess)))
            tf.logging.debug('cur_epoch_tensor_value: {}.'.format(self.cur_epoch_tensor.eval(sess)))
            tf.logging.info(" Done.")
            return global_step
        else:
            tf.logging.warning(" No checkpoint found.")
            return None

    def init_cur_epoch(self):
        """
        Initialize a tf.Variable to use it as epoch counter.
            When you call load function, this tensor will restore from saved data if exist.
        """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign_add(self.cur_epoch_tensor, 1)

    def init_global_step(self):
        """
        Initialize a tf.Variable to use it as global step counter.
            Remember to add the global step tensor to the trainer.
            When you call load function, this tensor will restore from saved data if exist.
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        """
        Initialize saver of model.
            Call this function in __init__ method of concrete class after you build all model.
            Otherwise tf.Variable defined after init_saver won't be added to checkpoint.
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        """
        Build model here.
        """
        raise NotImplementedError
