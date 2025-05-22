from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # type: ignore
from .base import Ensemble
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
from typing import Sequence
import logging

logger = logging.getLogger(__name__)


checkpoint = "microsoft/Phi-4-mini-instruct"
# NOTE - not actually the system prompt, that performs quite poorly.
system_prompt = """\
Summarize the following captions for a scene in an Indian movie:\n{captions}
Only include information supported by 2 or more captions, do not repeat information, and ignore information stating that the scene is in a movie.
"""


class PhiEnsembleDataset(Dataset):
    checkpoint = checkpoint
    prompt = system_prompt

    def __init__(self, in_files: Sequence[Path | str]):
        self.in_files: list[str] = [str(f) for f in in_files]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self.frames: dict[str, pd.DataFrame] = dict()
        for f in self.in_files:
            if Path(f).suffix == ".tsv":
                self.frames[f] = pd.read_csv(
                    f, sep="\t", header=None, names=["filename", "caption"]
                )
            elif Path(f).suffix == ".json":
                self.frames[f] = pd.read_json(f)
            elif Path(f).suffix == ".csv":
                self.frames[f] = pd.read_csv(f, names=["filename", "caption"])
            else:
                raise ValueError(f"Unsupported file format: {Path(f).suffix}")
            self.frames[f] = (
                self.frames[f].sort_values("filename").reset_index(drop=True)
            )

    def __len__(self):
        return len(self.frames[self.in_files[0]])

    def __getitem__(
        self, idx: int | slice | list[int]
    ) -> tuple[list[str], dict[str, Tensor]]:
        logger.info("Starting __getitem__")
        if isinstance(idx, int):
            idx = [idx]
        filenames_batch = (
            list(self.frames.values())[0]["filename"].loc[idx].tolist()
        )
        captions_batch: list[pd.Series[str]] = [
            frame.loc[idx, "caption"] for frame in (self.frames.values())
        ]
        messages_batch: list[dict[str, Tensor]] = []
        captions: tuple[str, ...]
        for captions in zip(*captions_batch):
            messages: list[dict[str, str]]
            captions = tuple(
                (
                    f"{i}. {caption[:1000].replace('\n', ' ')}"
                    for i, caption in enumerate(captions)
                )
            )
            messages = [
                {
                    "role": "user",
                    "content": f"{self.prompt.format(captions='\n'.join(captions))}",
                },
            ]
            # messages_batch.append(messages)
            messages_batch.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    # return_dict=True,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            # print(messages_batch[-1][list(messages_batch[-1].keys())[0]].shape)
        logger.info("Finished messages batch")
        # messages_dict = self.tokenizer.apply_chat_template(
        #     messages_batch,
        #     return_dict=True,
        #     # return_tensors="pt",
        #     tokenize=False,
        #     padding=True,
        #     truncation=True,
        # )
        logger.debug(
            f"{type(messages_batch) = }\n{type(messages_batch[0]) = }"
        )
        logger.debug(f"\n{messages_batch}")
        messages_dict = self.tokenizer(
            messages_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        logger.info("Finished tokenization")
        return filenames_batch, messages_dict


class PhiEnsemble(Ensemble):
    name = "Phi4Ensemble"
    checkpoint = checkpoint
    prompt = system_prompt

    def __init__(
        self, *args, temperature: float = 0.0, in_files: list[Path], **kwargs
    ):
        super().__init__(*args, in_files=in_files, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype="auto",
            trust_remote_code=True,
            # _attn_implementation="flash_attention_2",
        )
        # if you do not use Ampere or later GPUs, change attention to "eager"
        self.generation_config = GenerationConfig.from_pretrained(
            self.checkpoint,
            do_sample=False,
        )  # ! temperature might be a bad choice to change, default is 1.0
        self.max_new_tokens = 1024

    def predict_step(
        self, batch: tuple[list[str], dict[str, Tensor]], batch_idx
    ) -> str | list[str]:
        logger.info("Starting predict_step")
        filenames, inputs = batch
        logger.debug(
            f"1st input prompt (with template): {self.tokenizer.decode(inputs['input_ids'][0])}"
        )
        logger.debug(
            f"Length of batch = {len(filenames)}, {inputs['input_ids'].shape = }"
        )
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            generation_config=self.generation_config,
            # num_logits_to_keep=1,
        )
        logger.info("Finished generation")
        out = out[:, inputs["input_ids"].shape[1] :]
        captions = self.tokenizer.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.debug(f"Output: {captions}")
        logger.info("Finished decoding")
        self._save(filenames, captions)
        return captions
