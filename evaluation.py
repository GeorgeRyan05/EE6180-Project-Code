from pathlib import Path
from image_captioning_with_blip.metrics import (HuggingFaceEvaluator, load_ground_truth, load_predictions, huggingface_metrics)
import pandas as pd

PREDICTIONS = Path("./captions.txt")
REFERENCE = Path("./mastani/Frames.json")

predictions = load_predictions(PREDICTIONS)
ground_truths = load_ground_truth(REFERENCE)

df = pd.DataFrame({
    "predictions": predictions,
    "ground_truths": ground_truths,
})


evaluator = HuggingFaceEvaluator("bleu", "bertscore")
def evaluate(row):
    pred = row["predictions"]
    gt = row["ground_truths"]
    return evaluator.evaluate(pred, gt)

results: pd.DataFrame = df.apply(
    evaluate,
    axis=1,
    result_type="expand",
)

results.to_csv("./bert_score.tsv", sep="\t", index=True)