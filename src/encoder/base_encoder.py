# =============================================
                
                               
                         
#   --repr_input_dim --repr_output_dim --repr_sub_pred_len
# =============================================

import abc
import importlib
import random
from typing import Union

import numpy as np
import torch
import pickle
from encoder.encoder_config import ENCODER_CONFIG

import torch.nn as nn
from utils.path_utils import get_trained_encoder_path
from utils.project_paths import resolve_checkpoint_path

class BaseEncoder(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        pass

class EncoderFactory:
    @staticmethod
    def build_encoder(args, device="cuda") -> BaseEncoder:
        base_name = args.repr_encoder

        simplets_payload = None
        if base_name == "SimpleTS2Vec":
            from encoder.simplets_checkpoint import resolve_simplets_ts2vec_checkpoint

            _, simplets_payload = resolve_simplets_ts2vec_checkpoint(args)

        if base_name not in ENCODER_CONFIG:
            raise ValueError(
                f"Encoder base '{base_name}TSRouter runtime message: "
                f"TSRouter runtime message: {args.repr_encoder}"
            )

        cfg = dict(ENCODER_CONFIG[base_name])
        for key in ("encoder_model_path", "scaler_path"):
            if cfg.get(key):
                cfg[key] = str(resolve_checkpoint_path(cfg[key]))
        if simplets_payload is not None:
            cfg.update(dict(simplets_payload["config"]))
        cfg["encoder_name"] = base_name
        cfg["encoder_type"] = getattr(args, "encoder_type", cfg.get("encoder_type", "Random"))

        req_input = int(getattr(args, "repr_input_dim", cfg.get("default_input_dim", 96)))
        req_output = int(getattr(args, "repr_output_dim", cfg.get("default_embedding_dim", 128)))
        req_pl = int(getattr(args, "repr_sub_pred_len", cfg.get("default_sub_pred_len", 192)))
        if str(base_name).lower() == "none":
            req_output = req_input

                                           
        if "fixed_input_dim" in cfg and req_input != int(cfg["fixed_input_dim"]):
            raise ValueError(
                f"{base_name}TSRouter runtime message: {cfg['fixed_input_dim']}，"
                f"TSRouter runtime message: {req_input}"
            )
        if "fixed_output_dim" in cfg and req_output != int(cfg["fixed_output_dim"]):
            raise ValueError(
                f"{base_name}TSRouter runtime message: {cfg['fixed_output_dim']}，"
                f"TSRouter runtime message: {req_output}"
            )
        if "fixed_sub_pred_len" in cfg and req_pl != int(cfg["fixed_sub_pred_len"]):
            raise ValueError(
                f"{base_name}TSRouter runtime message: {cfg['fixed_sub_pred_len']}，"
                f"TSRouter runtime message: {req_pl}"
            )

        cfg["input_dim"] = int(req_input)
        cfg["embedding_dim"] = int(req_output)
        cfg["sub_pred_len"] = int(req_pl)
        cfg["seq_len"] = int(req_input)
        cfg["context_len"] = int(req_input)
        cfg["pred_len"] = int(req_pl)
        module = importlib.import_module(cfg["module_path"])
        cls = getattr(module, cfg["class_name"])

                                                        
        class Configs:
            def __init__(self, args, cfg):
                self.__dict__.update(cfg)
                self.__dict__.update(vars(args))
                                  
                self.input_dim = int(cfg["input_dim"])
                self.embedding_dim = int(cfg["embedding_dim"])
                self.sub_pred_len = int(cfg["sub_pred_len"])
                self.seq_len = int(cfg["seq_len"])
                self.context_len = int(cfg["context_len"])
                self.pred_len = int(cfg["pred_len"])
                if getattr(self, "random_stats_fusion", None) is None:
                    self.random_stats_fusion = cfg.get("random_stats_fusion", "none")

        configs = Configs(args, cfg)

        encoder_seed = getattr(configs, "repr_encoder_seed", None)
        if encoder_seed is not None:
            np.random.seed(encoder_seed)
            random.seed(encoder_seed)
            torch.manual_seed(encoder_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(encoder_seed)
                torch.cuda.manual_seed_all(encoder_seed)

        encoder = cls(configs=configs, device=device)
        if simplets_payload is not None:
            state = simplets_payload.get("state_dict")
            missing, unexpected = encoder.load_state_dict(state, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    "SimpleTS2Vec strict checkpoint load failed: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            print(
                "[SimpleTS2Vec] loaded checkpoint: "
                f"{getattr(args, 'simplets_ts2vec_checkpoint', '')} "
                f"fingerprint={getattr(args, 'simplets_ts2vec_checkpoint_fingerprint', '')[:12]}"
            )
        if str(getattr(configs, "encoder_type", "")).lower() == "train":
            ckpt_path = get_trained_encoder_path(configs)
            suppress_load = bool(getattr(configs, "_suppress_trained_encoder_load", False))
            if getattr(configs, "_allow_missing_trained_encoder", False) or suppress_load:
                if not torch.cuda.is_available():
                    pass
            elif not torch.jit.is_scripting():
                if not __import__("os").path.exists(ckpt_path):
                    raise FileNotFoundError(
                        f"Trained encoder checkpoint not found: {ckpt_path}. "
                        "Run Step3/save_model_zoo_repr first or disable encoder_type=Train for Step1/2 bootstrap."
                    )
            if (not suppress_load) and __import__("os").path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                state = ckpt.get("state_dict", ckpt)
                missing, unexpected = encoder.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[TrainEncoder] load_state_dict missing={missing}, unexpected={unexpected}")
                print(f"[TrainEncoder] loaded checkpoint: {ckpt_path}")

        scaler = None
        if cfg.get("scaler_path"):
            import pickle
            with open(cfg["scaler_path"], "rb") as f:
                scaler = pickle.load(f)

        if int(encoder.embedding_dim) != int(configs.embedding_dim):
            encoder = _ProjectedEncoder(
                base_encoder=encoder,
                target_dim=int(configs.embedding_dim),
                seed=getattr(configs, "repr_encoder_seed", 0),
                device=device,
            )

        return encoder, scaler, configs


class _ProjectedEncoder(BaseEncoder):
    def __init__(self, base_encoder: BaseEncoder, target_dim: int, seed: int = 0, device="cuda"):
        super().__init__()
        self.base_encoder = base_encoder
        self._embedding_dim = int(target_dim)
        self.device = torch.device(device) if isinstance(device, str) else device

        in_dim = int(base_encoder.embedding_dim)
        self.proj = nn.Linear(in_dim, self._embedding_dim, bias=False)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + 12345)
        with torch.no_grad():
            weight = torch.randn(self._embedding_dim, in_dim, generator=g) / (in_dim ** 0.5)
            self.proj.weight.copy_(weight)
        self.proj.requires_grad_(False)
        self.proj.to(self.device).eval()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @torch.no_grad()
    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        emb = self.base_encoder.encode(series_data)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        emb = emb.float().to(self.device)
        out = self.proj(emb)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out.cpu()
