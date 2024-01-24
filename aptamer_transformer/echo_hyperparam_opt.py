from echo.src.base_objective import BaseObjective
import optuna
import traceback
import numpy as np
from aptamer_transformer.main import train_and_evaluate, parse_arguments
from aptamer_transformer.factories_model_loss import model_config

class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):
        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)
    def train(self, trial, conf):
        try:
            return echo_trainer(conf, trial=trial)
        except Exception as e:
            if "CUDA" in str(e):
                raise optuna.TrialPruned()
            elif "not compatible" in str(e):
                raise optuna.TrialPruned()
            else:
                print(traceback.format_exc())
                raise e
    
    
    
def echo_trainer(cfg, trial=None):
    args = parse_arguments()
    # args.distributed = True
    cfg = model_config(cfg)
    working_dir = cfg['working_dir']
    model_type = cfg['model_type']
    cfg = {k: v.replace('{WORKING_DIR}', f'{working_dir}') if isinstance(v, str) else v for k, v in cfg.items()}
    cfg = {k: v.replace('{MODEL_TYPE}', f'{model_type}') if isinstance(v, str) else v for k, v in cfg.items()}
    cfg['running_echo'] = True

    metrics = train_and_evaluate(cfg, trial=trial, args=args)
    
    hyper_param_metric = {"val_loss": min(np.mean(metrics['val_loss'], axis=1))}
    return  hyper_param_metric
    