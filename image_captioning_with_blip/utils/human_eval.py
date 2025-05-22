from pathlib import Path
from os import PathLike
from typing import cast
import pandas as pd
import textwrap


def generate_comparison(
    image_set: PathLike,
    ground_truth_captions: PathLike,
    generated_captions: PathLike,
    markdown_file: PathLike,
) -> None:
    """
    Convert a TSV file to a Markdown table for comparison.

    Args:
        image_set (PathLike): Path to the TSV file containing the image set.
        ground_truth_captions (PathLike): Path to the TSV file containing the ground truth captions.
        generated_captions (PathLike): Path to the TSV file containing the generated captions.

    Returns:
        str: Markdown formatted string representing the comparison table.
    """
    image_set = Path(image_set)
    ground_truth_captions = Path(ground_truth_captions)
    generated_captions = Path(generated_captions)
    markdown_file = Path(markdown_file)

    # Check formats
    if not (
        image_set.is_dir()
        and ground_truth_captions.is_file()
        and generated_captions.is_file()
    ):
        raise ValueError(
            (
                f"Invalid file or directory paths provided. "
                f"{image_set = }, {ground_truth_captions = }, {generated_captions = }, {markdown_file = }"
            )
        )

    generated_captions_df = (
        pd.read_csv(
            generated_captions,
            sep="\t",
            header=None,
            names=["caption"],
            index_col=0,
        )
        if generated_captions.suffix == ".tsv"
        else pd.read_json(generated_captions).set_index("filename")
    )
    generated_captions_df.sort_index(inplace=True)
    ground_truth_captions_df = pd.read_json(ground_truth_captions)
    ground_truth_captions_df["file"] = ground_truth_captions_df[
        "frame_time"
    ].apply(lambda x: f"{x}.jpg")
    ground_truth_captions_df["caption"] = ground_truth_captions_df[
        "captions"
    ].apply(lambda x: x.popitem()[0])
    ground_truth_captions_df.set_index("file", inplace=True)
    with open(markdown_file, "w") as f:
        f.write("| Image | Ground Truth Caption | Generated Caption |\n")
        f.write("|-------|----------------------|-------------------|\n")
        for idx, row in generated_captions_df.iterrows():
            idx = cast(str, idx)
            image_path = Path("..") / image_set / idx
            # ! Do NOT use absolute paths, preview won't work
            gt_caption = ground_truth_captions_df.loc[idx, "caption"]
            gen_caption = row["caption"].replace("\n", "<br>")
            gen_caption = "<br>".join(textwrap.wrap(gen_caption, width=50))
            f.write(f"| ![]({image_path}) | {gt_caption} | {gen_caption} |\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a Markdown table for image captioning comparison."
    )
    parser.add_argument(
        "image_set", type=Path, help="Path to the image set directory."
    )
    parser.add_argument(
        "ground_truth_captions",
        type=Path,
        help="Path to the ground truth captions TSV file.",
    )
    parser.add_argument(
        "generated_captions",
        type=Path,
        help="Path to the generated captions TSV file.",
    )
    parser.add_argument(
        "markdown_file",
        nargs="?",
        type=Path,
        help="Path to the output Markdown file.",
    )
    args = parser.parse_args()
    if args.markdown_file is None:
        args.markdown_file = args.generated_captions.with_suffix(".md")
    generate_comparison(
        args.image_set,
        args.ground_truth_captions,
        args.generated_captions,
        args.markdown_file,
    )
