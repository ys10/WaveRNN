#  coding=utf-8
from tqdm import tqdm
from base.base_predict import BasePredict


class ExamplePredictor(BasePredict):
    def __init__(self, sess, model, data_loader, config, logger):
        super(ExamplePredictor, self).__init__(sess, model, data_loader, config, logger)

    def predict_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.iter_per_epoch))
        predictions = []
        for _ in loop:
            pred = self.predict_step()
            predictions.extend(pred[0])
        return predictions

    def predict_step(self):
        predictions = self.sess.run([self.model.predictions])
        return predictions
