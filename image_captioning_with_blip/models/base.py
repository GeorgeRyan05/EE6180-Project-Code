import pytorch_lightning as pl
from typing import Optional, TypeAlias
from pathlib import Path
import json

type JSON = "dict[str, str | int | float | bool | JSON] | list[str | int | float | bool | JSON]"
JSON_dict = dict[str, str | int | float | bool | JSON]


class Model(pl.LightningModule):
    name: str
    checkpoint: str
    out_file: Path
    prompt: Optional[str] = None

    def __init__(
        self,
        out_file: Optional[Path | str] = None,
        checkpoint: Optional[str] = None,
        *args,
        hyperparameters: Optional[JSON_dict] = None,
        **kwargs,
    ):
        """
        Args:
            out_file (Path | str, optional): Path to the output file (tsv, and a json with the same stem). Defaults to a tsv.
            checkpoint (str, optional): Checkpoint name. Defaults to None.
            *args: Additional arguments for the parent class.
            hyperparameters (JSON_dict, optional): Hyperparameters for the model. Defaults to None.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        if checkpoint is not None:
            self.checkpoint = checkpoint
        self.out_file = (
            Path(
                f"outputs/{self.name}__{self.checkpoint.replace(r'/', '__')}.tsv"
            )
            if out_file is None
            else Path(out_file)
        )
        self.json_file = self.out_file.with_suffix(".json")
        count = 1
        if self.out_file:
            original = self.out_file
            while self.out_file.exists():
                self.out_file = original.with_stem(f"{original.stem}__{count}")
                count += 1
        self.json_file = self.out_file.with_suffix(".json")

        if hyperparameters is None:
            hyperparameters = {}
        hyperparameters = {
            "out_file": str(self.out_file),
            "checkpoint": self.checkpoint,
            "name": self.name,
            "prompt": self.prompt,
        } | hyperparameters

        self.save_hyperparameters(hyperparameters)

    def _save(self, filenames: list[str], captions: list[str]) -> None:
        if not self.out_file:
            return
        with open(self.out_file, "a") as f:
            for fname, c in zip(filenames, captions):
                print(f"{fname}\t{c}", file=f)
        if self.json_file.exists():
            cur_json = json.load(self.json_file.open())
        else:
            cur_json = []
        for filename, caption in zip(filenames, captions):
            cur_json.append(
                {
                    "filename": (filename),
                    "caption": (caption),
                }
            )
        with open(self.json_file, "w") as f:
            json.dump(cur_json, f, indent=2)


class Ensemble(Model):
    name: str = "Ensemble"
    checkpoint: str
    out_file: Path
    prompt: Optional[str] = None
    in_files: list[Path]

    def __init__(
        self,
        out_file: Optional[Path | str] = None,
        checkpoint: Optional[str] = None,
        in_files: Optional[list[Path]] = None,
        **kwargs,
    ):
        if in_files is not None:
            self.in_files = in_files
        else:
            self.in_files = []
        super().__init__(
            out_file,
            checkpoint,
            hyperparameters={"in_files": [str(f) for f in self.in_files]},
            **kwargs,
        )
