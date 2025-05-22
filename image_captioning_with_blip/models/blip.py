import logging
from PIL import Image
from pathlib import Path
from typing import cast
from collections.abc import Sequence
import torch
from transformers.models.blip import BlipProcessor, BlipForConditionalGeneration
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from .base import Model

logger = logging.getLogger(__name__)

class BlipPL(Model):
    name: str = "Blip"
    checkpoint: str = "Salesforce/blip-image-captioning-base"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = cast(
            BlipProcessor,
            BlipProcessor.from_pretrained(
                self.checkpoint
            ),
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.checkpoint
        )

    def predict_step(
        self,
        batch,
        batch_idx
    ) -> str | list[str]:
        filenames, images = batch
        out = self.model.generate(**images)
        captions = self.processor.batch_decode(out, skip_special_tokens=True)
        if self.out_file:
            with open(self.out_file, "a") as f:
                for fname, c in zip(filenames, captions):
                    print(f'{fname}\t{c}', file=f)
        return captions

class ImageDatasetBlip(Dataset):
    def __init__(self, images: dict[str, Image.Image]):
        filenames, _images = zip(*images.items())
        self.filenames: list[str] = list(filenames)
        self.images: list[Image.Image] = list(_images)
        self.processor: BlipProcessor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )  # type: ignore
        self.processor.tokenizer.padding_side = "left"  # type: ignore

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int|slice) -> tuple[str|list[str], dict[str, torch.Tensor]]:
        captions = self.filenames[idx]
        images = self.images[idx]
        images = self.processor(
            images=images, padding=True, return_tensors="pt"
        )
        images = {
            k: v.squeeze() for k, v in images.items()
        }  # here, squeezing was needed
        return captions, images


class CaptionDataModuleBlip(pl.LightningDataModule):
    def __init__(self, images: dict[str, Image.Image]):
        super().__init__()
        self.dataset = ImageDatasetBlip(images)

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
        )