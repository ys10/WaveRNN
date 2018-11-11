# coding=utf-8
from tqdm import tqdm
from base.base_validate import BaseValidate


class TemplateValidater(BaseValidate):
    def __init__(self, sess, model, data_loader, config, logger):
        super(TemplateValidater, self).__init__(sess, model, data_loader, config, logger)

    def validate_epoch(self):
        """
        Implement the logic of validate epoch:
        -loop over the number of iterations in the config and call the validate step
        -add any summaries you want using the summary
        """
        pass

    def validate_step(self):
        """
        Implement the logic of the validate step
        - run the tf.Session
        - return any metrics you need to summarize
        """
        pass
