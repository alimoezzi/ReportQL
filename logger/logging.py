import os
import logging
from config import configs
import pytorch_lightning as pl


class LoggingCallback(pl.Callback):
    def __init__(self):
        self.logger = logging.getLogger("pytorch_lightning")
        self.logger.setLevel(logging.INFO)

    def on_validation_end(self, trainer, pl_module):
        self.logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    self.logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        self.logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            output_test_results_file = os.path.join(configs['output_dir'], "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        self.logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))
