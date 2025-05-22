from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch import Tensor, nn
from torchvision import transforms
from transformers.models.blip_2 import Blip2Processor
from PIL import Image
import pytorch_lightning as pl


def split_dataset(
    input_data_folder: str | Path,
    split_size: int | float,
    output_data_folder: str | Path,
    overwrite: bool = False,
):
    """
    Split a dataset into training and validation sets.
    """
    input_data_folder = Path(input_data_folder)
    output_data_folder = Path(output_data_folder)
    files = list(input_data_folder.glob("*.jpg"))
    files.sort(key=lambda x: int(x.stem))
    if not (input_data_folder.is_dir() and output_data_folder.is_dir()):
        raise ValueError(
            f"Folder {input_data_folder} or {output_data_folder} does not exist or is not a directory."
        )
    if split_size < 1:
        split_size = round(len(files) * split_size)
    if isinstance(split_size, float):
        raise ValueError("split_size must not be a float if greater than 1.")
    indices = [round(i * len(files) / split_size) for i in range(split_size)]
    for i in indices:
        out_file = output_data_folder / files[i].name
        if out_file.exists() and not overwrite:
            raise ValueError(f"File {out_file} already exists.")
        (out_file).write_bytes(files[i].read_bytes())


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
