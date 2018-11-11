# coding=utf-8
import tensorflow as tf
from base.base_model import BaseModel


class FCNetModel(BaseModel):
    def __init__(self, config, data):
        super(FCNetModel, self).__init__(config, data)
        self.build_model()
        self.init_saver()

    def build_model(self):
        with tf.variable_scope('fully_connected_net', reuse=tf.AUTO_REUSE):
            '''read data'''
            features = tf.cast(self.data['features'], tf.float32)
            labels = tf.cast(self.data["labels"], tf.int32)

            '''build graph'''
            output = tf.layers.batch_normalization(features, training=self.config.training, name='bn_layer_0')
            output = tf.layers.dense(inputs=output, units=256, activation=tf.nn.relu, name='hidden_layer_1')
            output = tf.layers.batch_normalization(output, training=self.config.training, name='bn_layer_1')
            output = tf.layers.dense(inputs=output, units=256, activation=tf.nn.relu, name='hidden_layer_2')
            output = tf.layers.batch_normalization(output, training=self.config.training, name='bn_layer_2')
            output = tf.layers.dense(inputs=output, units=64, activation=tf.nn.relu, name='hidden_layer_3')
            output = tf.layers.batch_normalization(output, training=self.config.training, name='bn_layer_3')
            logits = tf.layers.dense(inputs=output, units=3, activation=tf.nn.softmax, name='output_layer')

            '''build loss, optimizer & metrics'''
            with tf.name_scope('loss'):
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
                if self.config.training:
                    self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(
                        self.loss, global_step=self.global_step_tensor)
                self.predictions = tf.argmax(logits, axis=-1)
                acc, self.acc_op = tf.metrics.accuracy(labels, self.predictions)
                rec, self.rec_op = tf.metrics.recall(labels, self.predictions)
                pre, self.pre_op = tf.metrics.precision(labels, self.predictions)
                self.f1_score = tf.divide(2 * tf.multiply(self.rec_op, self.pre_op), tf.add(self.rec_op, self.pre_op))
