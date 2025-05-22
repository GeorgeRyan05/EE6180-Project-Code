import logging
from PIL import Image
from pathlib import Path
from typing import Self, cast, Optional
from collections.abc import Sequence
from torch import Tensor
from transformers.models.blip_2 import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from .base import Model
from torchvision import transforms

logger = logging.getLogger(__name__)

default_checkpoint = "Salesforce/blip2-flan-t5-xxl"


class Blip2PL(Model):
    name = "Blip2"
    checkpoint = default_checkpoint
    prompt = """\
This image depicts a scene from an Indian movie. In the scene,"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.processor = cast(
            Blip2Processor,
            Blip2Processor.from_pretrained(self.checkpoint),
        )
        self.processor.tokenizer.padding_side = "left"  # type: ignore
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.checkpoint
        )

    def predict_step(self, batch, batch_idx) -> str | list[str]:
        filenames, images = batch
        out = self.model.generate(**images)
        captions = self.processor.batch_decode(out, skip_special_tokens=True)
        self._save(filenames, captions)
        return captions

    def on_before_batch_transfer(
        self, batch: tuple[Sequence[str], list[Tensor]], dataloader_idx: int
    ) -> tuple[Sequence[str], dict[str, Tensor]]:
        filenames, images = batch
        if self.prompt:
            images = self.processor(
                images=images,
                text=[self.prompt] * len(images),
                padding=True,
                return_tensors="pt",
                do_rescale=False,
            )
        else:
            images = self.processor(
                images=images,
                padding=True,
                return_tensors="pt",
                do_rescale=False,
            )
        images = {k: v.squeeze() for k, v in images.items()}
        return filenames, images


class ImageDataset(Dataset):
    def __init__(self, images: dict[str, Image.Image]):
        filenames, _images = zip(*images.items())
        self.filenames: list[str] = list(filenames)
        self.images: list[Tensor] = [
            transforms.ToTensor()(img) for img in _images
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(
        self, idx: int | slice
    ) -> tuple[str | list[str], Tensor | list[Tensor]]:
        captions = self.filenames[idx]
        images = self.images[idx]

        return captions, images


class CaptionDataModule(pl.LightningDataModule):
    def __init__(self, images: dict[str, Image.Image]):
        super().__init__()
        self.dataset = ImageDataset(images)

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
        )
