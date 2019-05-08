from __future__ import unicode_literals
import models
from config import Config



def main(model_name="CNN", is_training=True):
    config = Config()
    if model_name == "CNN":
        model = models.CNN(config)
    elif model_name == "RNN":
        model = models.RNN(config)
    elif model_name == "RCNN":
        model = models.RCNN(config)
    else:
        model = models.FC(config)

    if is_training:
        model.train()
    else:
        model.restore_model()
        model.predict()


# main("CNN")
# main("CNN", False)
# main("RNN")
# main("RNN", False)
main("RCNN")
# main("RCNN", False)
# main("FC")
# main("FC", False)
