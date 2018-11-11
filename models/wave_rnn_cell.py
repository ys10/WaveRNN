# coding=utf-8
import tensorflow as tf
from base.base_model import BaseModel


class WaveRNN(BaseModel):
    """
    Concrete model template.
    """
    def __init__(self, config, data):
        super(WaveRNN, self).__init__(config, data)
        self.build_model()
        self.init_saver()

    def build_model(self):
        """
        Here you build the tf.Graph of any model you want and also define the loss.
        """

        pass

    def __placeholders__(self):
        """
        Define inputs & labels.
        :return:
        """
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.config.batch_size, 1])

    def __pre_processing__(self):
        """
        Pre-process
        :return:
        """
        self.seq_length = self.data['seq_length']
        self.samples = self.data['samples']
        # self.mel_frames = self.data['mel_frames']

        self.samples = tf.transpose(self.samples, [1, 0, 2])

        pass

