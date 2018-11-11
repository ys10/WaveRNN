# coding=utf-8
import tensorflow as tf
from models.fc_net_model import FCNetModel
from data_loader.data_set_loader import DataSetLoader
from data_loader.data_generator import DataGenerator
from utils.configs import process_config
from utils.logger import Logger
from operators.example_trainer import ExampleTrainer


def train():
    train_config = process_config("configs/train.json")
    g = tf.Graph()
    with g.as_default():
        # load data
        train_data_gen = DataGenerator()
        data_loader = DataSetLoader(train_config, {'train': train_data_gen}, default_set_name='train')
        next_data = data_loader.next_data
        # create an instance of the model you want
        model = FCNetModel(train_config, next_data)
        with tf.Session() as sess:
            # create tensorboard logger
            logger = Logger(sess, train_config)
            # create trainer and pass all the previous components to it
            trainer = ExampleTrainer(sess, model, data_loader, train_config, logger)
            # load model if exists
            model.load(sess)
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
