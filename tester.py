import logging
from utils.logging_utils import LoggerUtils
from utils.checkpoint import read_checkpoint
from utils.evaluation import EvaluationUtils
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import Metrics
from utils.model_utils import get_config_params, verify_checkpoint_params

class Tester(EvaluationUtils):

    def __init__(self, config, data_manager, logger: logging.Logger, writer: SummaryWriter, logger_utils: LoggerUtils, model, criterion, device):
        self.data_manager = data_manager
        self.config = config
        self.logger = logger
        self.writer = writer
        self.logger_utils = logger_utils
        self.model = model
        self.criterion = criterion
        self.device = device
        self.num_classes = len(data_manager.classes)
        # Inizializzazione della classe metrics
        self.metrics = Metrics(model=self.model, device=self.device, num_classes=self.num_classes)
        self.evaluation_utils = EvaluationUtils(model=self.model, criterion=self.criterion, device=self.device, metrics=self.metrics)

        
        # parametri
        self.model_load_path = config.parameters.model_load_path


    def test_model(self, reload_checkpoint, epoch=None):
            self.logger.info("Test finale")
            self.logger.info("Inizio testing...")
            self.current_epoch = epoch

            if reload_checkpoint:
                ckpt = read_checkpoint(self.model_load_path, self.logger)
                if ckpt is None:                
                    self.logger.error("Impossibile caricare modello per testing. Esco.")
                    return None
                self.model.load_state_dict(ckpt['model_state_dict']) 
                self.meta = ckpt.get('meta')
                self.current_epoch = self.meta['epoch']

                config_params = get_config_params(self.config)
                verify_checkpoint_params(self.meta, config_params, self.logger)
                    
            self.logger.info("Inizio valutazione completa")    
            test_metrics = self.evaluation_utils._evaluate_full(self.data_manager.test_loader)
            self.logger.info("Fine valutazione completa")    
            # scrive su writer esistente (dal train o appena creato)
            metrics_dict = {"test_final": test_metrics}

            # Log su terminale e TensorBoard
            self.logger_utils.log_terminal(self.current_epoch, metrics_dict)
            self.logger_utils.log_tensorboard(self.current_epoch, metrics_dict)
            self.logger_utils.log_test_final(self.current_epoch, test_metrics, config_params)

            self.logger.info("Fine testing")
            return test_metrics
