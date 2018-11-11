# coding=utf-8
import tensorflow as tf
from base.base_model import BaseModel


class TemplateModel(BaseModel):
    """
    Concrete model template.
    """
    def __init__(self, config, data):
        super(TemplateModel, self).__init__(config, data)
        self.build_model()
        self.init_saver()

    def build_model(self):
        """
        Here you build the tf.Graph of any model you want and also define the loss.
        """
        pass
