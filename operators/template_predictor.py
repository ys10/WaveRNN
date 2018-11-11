# coding=utf-8
from tqdm import tqdm
from base.base_predict import BasePredict


class TemplatePredictor(BasePredict):
    def __init__(self, sess, model, data_loader, config, logger):
        super(TemplatePredictor, self).__init__(sess, model, data_loader, config, logger)

    def predict_epoch(self):
        """
        Implement the logic of predict epoch:
        -loop over the number of iterations in the config and call the predict step
        -add any summaries you want using the summary
        """
        pass

    def predict_step(self):
        """
        Implement the logic of the predict step
        - run the tf.Session
        - return any metrics you need to summarize
        """
        pass
