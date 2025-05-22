# %%
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import re


def get_file_path(name: str) -> Path:
    """
    Get the file path from the name.
    """
    if not name.endswith(".jpg"):
        name = f"{name}.jpg"
    file = (
        Path("./downloaded_images") / name
        if "downloaded_images" not in name
        else Path(name)
    )
    return file


parser = ArgumentParser()
parser.add_argument(
    "file",
    type=get_file_path,
    help="Image file number or file name (not path), present in downloaded_images",
)
args = parser.parse_args()
img_file: Path = args.file
# %%
OUTPUTS_FOLDER = {
    "Ensemble-Llama": "./outputs/Phi4Ensemble__microsoft__Phi-4-mini-instruct__2.json",
    "Ensemble+Llama": "./outputs/Phi4Ensemble__microsoft__Phi-4-mini-instruct__8.json",
    "Llama3.2": "./outputs/Llama3.2__meta-llama__Llama-3.2-11b-Vision-Instruct.json",
    "Phi4": "./outputs/Phi4__microsoft__Phi-4-multimodal-instruct.json",
}
# %%
frames = {
    name: (pd.read_json(file)).set_index("filename").sort_index()
    for name, file in OUTPUTS_FOLDER.items()
}
captions = {
    name: frame.loc[img_file.name].item() for name, frame in frames.items()
}
# %%
with open("./view.md", "w") as f:
    print(f"![]({img_file})\n", file=f)
    for name, caption in captions.items():
        print(f"**{name}**: \n```\n{caption}\n```\n", file=f)
