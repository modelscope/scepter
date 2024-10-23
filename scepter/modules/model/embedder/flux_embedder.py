import torch
from scepter.modules.model.embedder.base_embedder import BaseEmbedder
from scepter.modules.model.registry import EMBEDDERS
from scepter.modules.model.tokenizer.tokenizer_component import whitespace_clean, basic_clean, canonicalize
from scepter.modules.utils.config import dict_to_yaml
import transformers
from scepter.modules.utils.file_system import FS


@EMBEDDERS.register_class()
class HFEmbedder(BaseEmbedder):
    para_dict = {
        "HF_MODEL_CLS": {
            "value": None,
            "description": "huggingface cls in transfomer"
        },
        "MODEL_PATH": {
            "value": None,
            "description": "model folder path"
        },
        "HF_TOKENIZER_CLS": {
            "value": None,
            "description": "huggingface cls in transfomer"
        },

        "TOKENIZER_PATH": {
            "value": None,
            "description": "tokenizer folder path"
        },
        "MAX_LENGTH": {
            "value": 77,
            "description": "max length of input"
        },
        "OUTPUT_KEY": {
            "value": "last_hidden_state",
            "description": "output key"
        },
        "D_TYPE": {
            "value": "float",
            "description": "dtype"
        },
        "BATCH_INFER": {
            "value": False,
            "description": "batch infer"
        }
    }
    para_dict.update(BaseEmbedder.para_dict)
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        hf_model_cls = cfg.get('HF_MODEL_CLS', None)
        model_path = cfg.get("MODEL_PATH", None)
        hf_tokenizer_cls = cfg.get('HF_TOKENIZER_CLS', None)
        tokenizer_path = cfg.get('TOKENIZER_PATH', None)
        self.max_length = cfg.get('MAX_LENGTH', 77)
        self.output_key = cfg.get("OUTPUT_KEY", "last_hidden_state")
        self.d_type = cfg.get("D_TYPE", "float")
        self.clean = cfg.get("CLEAN", "whitespace")
        self.batch_infer = cfg.get("BATCH_INFER", False)
        torch_dtype = getattr(torch, self.d_type)

        assert hf_model_cls is not None and hf_tokenizer_cls is not None
        assert model_path is not None and tokenizer_path is not None

        with FS.get_dir_to_local_dir(tokenizer_path, wait_finish=True) as local_path:
            self.tokenizer = getattr(transformers, hf_tokenizer_cls).from_pretrained(local_path,
                                                                                     max_length = self.max_length,
                                                                                     torch_dtype = torch_dtype)

        with FS.get_dir_to_local_dir(model_path, wait_finish=True) as local_path:
            self.hf_module = getattr(transformers, hf_model_cls).from_pretrained(local_path, torch_dtype = torch_dtype)


        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str], return_mask = False):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        if return_mask:
            return outputs[self.output_key], batch_encoding['attention_mask'].to(self.hf_module.device)
        else:
            return outputs[self.output_key], None

    def encode(self, text, return_mask = False):
        if isinstance(text, str):
            text = [text]
        if self.clean:
            text = [self._clean(u) for u in text]
        if not self.batch_infer:
            cont, mask = [], []
            for tt in text:
                one_cont, one_mask = self([tt], return_mask=return_mask)
                cont.append(one_cont)
                mask.append(one_mask)
            if return_mask:
                return torch.cat(cont, dim=0), torch.cat(mask, dim=0)
            else:
                return torch.cat(cont, dim=0)
        else:
            ret_data = self(text, return_mask = return_mask)
            if return_mask:
                return ret_data
            else:
                return ret_data[0]


    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        return text
    @staticmethod
    def get_config_template():
        return dict_to_yaml('EMBEDDER',
                            __class__.__name__,
                            HFEmbedder.para_dict,
                            set_name=True)

@EMBEDDERS.register_class()
class T5PlusClipFluxEmbedder(BaseEmbedder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    para_dict = {
        'T5_MODEL': {},
        'CLIP_MODEL': {}
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.t5_model = EMBEDDERS.build(cfg.T5_MODEL, logger=logger)
        self.clip_model = EMBEDDERS.build(cfg.CLIP_MODEL, logger=logger)

    def encode(self, text):
        t5_embeds = self.t5_model.encode(text, return_mask = False)
        clip_embeds = self.clip_model.encode(text, return_mask = False)
        # change embedding strategy here
        return {
            'context': t5_embeds,
            'y': clip_embeds,
        }

    @staticmethod
    def get_config_template():
        return dict_to_yaml('EMBEDDER',
                            __class__.__name__,
                            T5PlusClipFluxEmbedder.para_dict,
                            set_name=True)