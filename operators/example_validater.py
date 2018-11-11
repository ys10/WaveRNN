# coding=utf-8
from tqdm import tqdm
from base.base_validate import BaseValidate


class ExampleValidater(BaseValidate):
    def __init__(self, sess, model, data_loader, config, logger):
        super(ExampleValidater, self).__init__(sess, model, data_loader, config, logger)

    def validate_epoch(self):
        """
        Implement the logic of validate epoch:
        -loop over the number of iterations in the config and call the validate step
        -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.iter_per_epoch))
        total_loss, total_acc, total_rec, total_pre, total_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        for _ in loop:
            loss, acc, rec, pre, f1 = self.validate_step()
            total_loss += loss
            total_acc += acc
            total_rec += rec
            total_pre += pre
            total_f1 += f1

        summaries_dict = {
            'loss_validate': total_loss / self.config.iter_per_epoch,
            'acc_validate': total_acc / self.config.iter_per_epoch,
            'rec_validate': total_rec / self.config.iter_per_epoch,
            'pre_validate': total_pre / self.config.iter_per_epoch,
            'f1_validate': total_f1 / self.config.iter_per_epoch,
        }
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def validate_step(self):
        """
        Implement the logic of the validate step
        - run the tf.Session
        - return any metrics you need to summarize
        """
        _, loss, acc, rec, pre, f1 = self.sess.run([
            self.model.train_op, self.model.loss, self.model.acc_op,
            self.model.rec_op, self.model.pre_op, self.model.f1_score])
        return loss, acc, rec, pre, f1
