from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig  # type: ignore
from .base import Model
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import pandas as pd
import json
from typing import TypedDict
from functools import partial

checkpoint = "microsoft/Phi-4-multimodal-instruct"


class Descriptor(TypedDict):
    label: str
    rect_box: list[float]
    polygon: list[list[int]]


def get_pos(pos: tuple[float, float], size: tuple[int, int]) -> str:
    """Split into 9 sections and return the section"""
    x, y = pos
    w, h = size
    if x < w / 3:
        x_pos = "left"
    elif x < 2 * w / 3:
        x_pos = "center"
    else:
        x_pos = "right"
    if y < h / 3:
        y_pos = "top"
    elif y < 2 * h / 3:
        y_pos = "middle"
    else:
        y_pos = "bottom"
    return (
        f"{y_pos} {x_pos}"
        if not (x_pos == "center" and y_pos == "middle")
        else "center"
    )


def get_prompt_description(
    image: Image.Image, descriptors: list[Descriptor]
) -> list[str]:
    descriptions = []
    for descriptor in descriptors:
        label = descriptor["label"]
        rect_box = descriptor["rect_box"]
        centroid = (
            (rect_box[0] + rect_box[2]) / 2,
            (rect_box[1] + rect_box[3]) / 2,
        )
        # width, height = image.size
        pos = get_pos(centroid, image.size)
        descriptions.append(f"{label} ({pos})")
    return descriptions


class PhiSamImageDataset(Dataset):
    checkpoint = checkpoint
    prompt = """<|user|><|image_1|>This image depicts a scene from an Indian movie.
Do not attempt to guess the name of the movie, or comment on the fact that it is a movie.
The following objects have been identified in the image:
{sam_outputs}
Describe the scene in detail, including the characters, their actions and clothing, and the setting.
<|end|><|assistant|>"""

    def __init__(
        self, images: dict[str, Image.Image], sam_outputs: str | Path
    ):
        """Initialize the dataset with images and SAM outputs.

        Args:
            images (dict[str, Image.Image]): Dictionary of image filenames and images.
            sam_outputs (str | Path): Path to the SAM outputs file. (.tsv)
        """
        filenames, _images = zip(*images.items())
        self.filenames: list[str] = list(filenames)
        self.images: list[Image.Image] = list(_images)
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        df = pd.read_csv(sam_outputs, sep="\t", index_col="image")
        df["descriptor"] = df["descriptor"].apply(json.loads)
        self.sam_outputs = [
            get_prompt_description(image, descriptor)
            for image, descriptor in zip(
                self.images, df["descriptor"].loc[self.filenames]
            )
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(
        self, idx: int | list[int]
    ) -> tuple[str | list[str], Tensor | list[Tensor]]:
        filenames = (
            [self.filenames[i] for i in idx]
            if isinstance(idx, list)
            else self.filenames[idx]
        )
        images = (
            [self.images[i] for i in idx]
            if isinstance(idx, list)
            else self.images[idx]
        )
        prompts = (
            [
                self.prompt.format(sam_outputs=", ".join(self.sam_outputs[i]))
                for i in idx
            ]
            if isinstance(idx, list)
            else self.prompt.format(
                sam_outputs=", ".join(self.sam_outputs[idx]),
            )
        )
        images = self.processor(
            images=images,
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return filenames, images


class Phi4Sam(Model):
    name = "Phi4Sam"
    checkpoint = checkpoint
    prompt = PhiSamImageDataset.prompt

    def __init__(self, *args, temperature: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        )
        # if you do not use Ampere or later GPUs, change attention to "eager"
        self.generation_config = GenerationConfig.from_pretrained(
            self.checkpoint, do_sample=False
        )  # ! temperature might be a bad choice to change, default is 1.0
        # (do_sample sets temperature to 0 in effect, and a warning is produced otherwise)
        self.max_new_tokens = 1024

    def predict_step(self, batch, batch_idx) -> str | list[str]:
        filenames, images = batch
        out = self.model.generate(
            **images,
            max_new_tokens=self.max_new_tokens,
            generation_config=self.generation_config,
            num_logits_to_keep=1,
        )
        out = out[:, images["input_ids"].shape[1] :]
        captions = self.processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        self._save(filenames, captions)
        return captions
