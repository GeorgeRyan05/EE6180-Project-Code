from collections.abc import Sequence
from PIL import Image
import logging.config
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
import pytorch_lightning as pl
from image_captioning_with_blip.models.phi4sam import (
    Phi4Sam,
    PhiSamImageDataset,
)

logging.config.dictConfig(yaml.safe_load(Path("logging.yaml").open()))
logger = logging.getLogger(__name__)

################# CAN BE CHANGED TO OTHER MODELS #################
FOLDER = Path("./data_subset")
model = Phi4Sam()

images: dict[str, Image.Image] = {
    f.name: Image.open(f) for f in FOLDER.glob("*.jpg")
}

dataset = PhiSamImageDataset(
    images, sam_outputs=Path("./outputs") / "batch_descriptors.tsv"
)

################# CAN BE CHANGED TO OTHER MODELS #################
model.eval()
model.freeze()

trainer = pl.Trainer(accelerator="gpu", devices=2)
captions = trainer.predict(
    model,
    dataloaders=DataLoader(
        dataset,
        sampler=BatchSampler(SequentialSampler(dataset), 25, drop_last=False),
        num_workers=100,
        batch_size=None,
    ),
    # !!! DO NOT USE A BATCH SIZE OF 2, the first input will get a response full of repeats up to max_new_tokens
    # I don't know about 1, 3 and 6 work fine
)
