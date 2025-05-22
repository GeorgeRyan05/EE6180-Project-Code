from pathlib import Path
import pandas as pd

frame_nos = [761, 998, 2296, 3541, 3773, 7386, 8636]
files = [
    Path("./downloaded_images") / f"{frame_no}.jpg" for frame_no in frame_nos
]
OUTPUTS_FOLDER = {
    "Captions using Ensemble without Llama": "./outputs/Phi4Ensemble__microsoft__Phi-4-mini-instruct__2.json",
    "Captions using Ensemble with Llama": "./outputs/Phi4Ensemble__microsoft__Phi-4-mini-instruct__8.json",
    "Captions using Llama 3.2": "./outputs/Llama3.2__meta-llama__Llama-3.2-11b-Vision-Instruct.json",
    "Captions using Phi 4": "./outputs/Phi4__microsoft__Phi-4-multimodal-instruct.json",
    "Captions using Phi 4 with SAM 2": "./outputs/Phi4Sam__microsoft__Phi-4-multimodal-instruct.json",
    "Captions using Llama 3.2 with SAM 2": "./outputs/captions_GSAM.json",
}
Path("./view.tex").unlink(missing_ok=True)
# %%
for img_file in files:
    frames = {
        name: (pd.read_json(file)).set_index("filename").sort_index()
        for name, file in OUTPUTS_FOLDER.items()
    }
    captions = {
        name: (frame.loc[img_file.name]).item()
        for name, frame in frames.items()
    }

    with open("./view.tex", "a") as f:
        print(
            f"""\
\\clearpage
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.3\\textwidth]{{{Path("figures") / img_file.name}}}
    \\caption{{{img_file.name}}}
    \\label{{fig:{img_file.name}}}
\\end{{figure}}

""",
            file=f,
        )
        for name, caption in captions.items():
            print(
                f"\n\\begin{{lstlisting}}[caption={{{name} for {img_file.name}}}]\n{caption}\n\\end{{lstlisting}}\n",
                file=f,
            )
