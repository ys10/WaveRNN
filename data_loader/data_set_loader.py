# coding=utf-8
import tensorflow as tf


class DataSetLoader(object):
    def __init__(self, config, generators, default_set_name='train'):
        self.config = config
        self.generators = generators
        self.data_sets = dict()
        self.data_set_init_ops = dict()
        with tf.variable_scope("data"):
            for k in self.generators.keys():
                self.data_sets[k] = self.get_data_set_from_generator(self.generators[k].next, epochs=self.config.epochs,
                                                                     batch_size=self.config.batch_size)
            self.iterator = self.data_sets[default_set_name].make_one_shot_iterator()
            features, labels = self.iterator.get_next()
            self.next_data = {'features': features, 'labels': labels}
            for k in self.data_sets.keys():
                self.data_set_init_ops[k] = self.iterator.make_initializer(self.data_sets[k])

    @staticmethod
    def get_data_set_from_generator(generator_func, epochs=1, batch_size=16):
        data_set = tf.data.Dataset.from_generator(generator_func,
                                                  output_types=(tf.int32, tf.int32),
                                                  output_shapes=(tf.TensorShape([64]), tf.TensorShape([1])))
        data_set = data_set.repeat(epochs)
        data_set = data_set.batch(batch_size)
        return data_set
