# coding=utf-8
import tensorflow as tf
from models.fc_net_model import FCNetModel
from data_loader.data_set_loader import DataSetLoader
from data_loader.data_generator import DataGenerator
from utils.configs import process_config
from utils.logger import Logger
from operators.example_trainer import ExampleTrainer
from operators.example_validater import ExampleValidater


def train():
    def monkey_patched_train(self):
        """
        Train from current epoch to number of epochs in the config.
            Call train_epoch for each epoch, and increase cur_epoch_tensor.
        """
        tf.logging.info('Training...')
        begin_epoch = self.model.cur_epoch_tensor.eval(sess)
        # initialize training data set
        self.sess.run([self.data_loader.data_set_init_ops['train']])
        for cur_epoch in range(begin_epoch, begin_epoch + self.config.epochs):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
            '''validation'''
            if cur_epoch % self.config.epochs_per_validation == 0:
                '''validate'''
                validater.validate()
                '''recover training environment'''
                self.sess.run([self.data_loader.data_set_init_ops['train']])

    train_config = process_config("configs/train_with_validation.json")
    validate_config = process_config("configs/validate.json")
    g = tf.Graph()
    with g.as_default():
        # train data
        train_data_gen = DataGenerator()
        # validate data
        validate_data_gen = DataGenerator()
        data_loader = DataSetLoader(train_config,
                                    {'train': train_data_gen, 'validate': validate_data_gen}, default_set_name='train')
        next_data = data_loader.next_data
        # TODO
        # create an instance of the model you want
        model = FCNetModel(train_config, next_data)
        with tf.Session() as sess:
            # create tensorboard logger
            train_logger = Logger(sess, train_config)
            # create trainer and pass all the previous components to it
            trainer = ExampleTrainer(sess, model, data_loader, train_config, train_logger)
            # create tensorboard logger
            validate_logger = Logger(sess, validate_config)
            # create validater and pass all the previous components to it
            validater = ExampleValidater(sess, model, data_loader, validate_config, validate_logger)
            # load model if exists
            model.load(sess)
            # make a monkey patch to model
            trainer.train = monkey_patched_train.__get__(trainer, ExampleTrainer)
            # here you train your model
            trainer.train()
            # save model
            model.save(sess)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    '''train model'''
    train()
    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()
