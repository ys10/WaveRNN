# coding=utf-8
from tqdm import tqdm
from base.base_train import BaseTrain


class TemplateTrainer(BaseTrain):
    def __init__(self, sess, model, data_loader, config, logger):
        super(TemplateTrainer, self).__init__(sess, model, data_loader, config, logger)

    def train_epoch(self):
        """
       Implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        pass

    def train_step(self):
        """
       Implement the logic of the train step
       - run the tf.Session
       - return any metrics you need to summarize
       """
        pass
