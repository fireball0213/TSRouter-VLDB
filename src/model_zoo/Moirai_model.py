import os
import sys
from pathlib import Path

_TSFMSRC_DIR = Path(__file__).resolve().parent / "TSFM_src"
if _TSFMSRC_DIR.exists():
    _p = str(_TSFMSRC_DIR)
    if _p not in sys.path:
        sys.path.insert(0, _p)                   
else:
    raise FileNotFoundError(f"[Moirai] TSFM_src directory not found: {_TSFMSRC_DIR}")

from model_zoo.base_model import BaseModel
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

                                                     

class MoiraiModel(BaseModel):

    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
                             
        if self.args.fix_context_len:
            context_length = self.args.context_len
        else:
            context_length = 4000               

        print(
            f"[Moirai] context_len={context_length}, "
            f"batch_size={batch_size}, "
            f"freq_used=Fasle, "
            f"impute_missing=False"
        )

                                                                     
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(self.model_local_path),
            prediction_length=1,                             
            context_length=context_length,
            patch_size=32,
            num_samples=100,
            target_dim=1,                                               
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

                                                                 
        model.hparams.prediction_length = dataset.prediction_length
        model.hparams.target_dim = dataset.target_dim
        model.hparams.past_feat_dynamic_real_dim = dataset.past_feat_dynamic_real_dim

                                                                  
        predictor = model.create_predictor(batch_size=batch_size)
        return predictor
