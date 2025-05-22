import logging.config
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
import pytorch_lightning as pl
from image_captioning_with_blip.models.phi4ensemble import (
    PhiEnsembleDataset,
    PhiEnsemble,
)

logging.config.dictConfig(yaml.safe_load(Path("logging.yaml").open()))
logger = logging.getLogger(__name__)
assert logger.level == logging.DEBUG, (
    f"Logger level is not set to DEBUG, set to {logger.level}, {logger.name = }"
)

FOLDER = Path("./data_subset")

trainer = pl.Trainer(accelerator="gpu", devices=1)

files = [
    "./outputs/Blip2__Salesforce__blip2-flan-t5-xxl.tsv",
    # "./outputs/Llama3.2__meta-llama__Llama-3.2-11b-Vision-Instruct.tsv",
    # "./outputs/Phi4__microsoft__Phi-4-multimodal-instruct.tsv",
    "./outputs/captions_GSAM.csv",
    # "./outputs/Phi4Sam__microsoft__Phi-4-multimodal-instruct.tsv",
    "./outputs/Qwen2.5__Qwen__Qwen-2.5-Omni-7B.tsv",
    # "./outputs/Phi4__microsoft__Phi-4-multimodal-instruct__1.tsv",
]

dataset = PhiEnsembleDataset(files)
model = PhiEnsemble(in_files=[Path(f) for f in dataset.in_files])
model.eval()
model.freeze()
trainer.predict(
    model,
    dataloaders=DataLoader(
        dataset,
        sampler=BatchSampler(SequentialSampler(dataset), 25, drop_last=False),
        num_workers=10,
        batch_size=None,
    ),
)
