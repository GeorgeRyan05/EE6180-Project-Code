from collections.abc import Sequence
from typing import overload, NoReturn, Never, TypedDict, Literal
from evaluate import EvaluationModule, combine, load
import torch
from functools import wraps
from pathlib import Path
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

huggingface_metrics = [
    ("bleu"),
    ("meteor"),
    ("rouge"),
    ("bertscore"),
    ("sacrebleu"),
    ("exact_match"),
]
class JSONFormat(TypedDict):
    file: str
    frame_time: int
    captions: dict[str, Literal[100]]

def load_ground_truth(file: str|Path) -> dict[str, str]:
    """
    Load the ground truth captions from a JSON file.
    Args:
        file (str|Path): Path to the JSON file containing the ground truth captions.
    Returns:
        dict[str, str]: A dictionary mapping image filenames to their corresponding captions.
    """
    with open(file, "r") as f:
        data: list[JSONFormat] = json.load(f)
    ground_truths: dict[str, str] = {}
    for item in data:
        name: str = str(item["frame_time"])
        caption: str = item["captions"].popitem()[0]
        ground_truths[f'{name}.jpg'] = caption
    return ground_truths
    # //  ground_truths_list = [ground_truths[key] for key in sorted(ground_truths.keys())]

def load_predictions(file: str|Path) -> dict[str, str]:
    """
    Load the predicted captions from a .txt file.
    Args:
        file (str|Path): Path to the .txt file containing the predicted captions.
    Returns:
        dict[str, str]: A dictionary mapping image filenames to their corresponding captions.
    """
    predictions: dict[str, str] = {}
    with open(file, "r") as f:
        for line in f:
            name, caption = line.strip().split("\t")
            predictions[name] = caption
    return predictions


def ensure_matching_types(func):
    @wraps(func)
    def wrapper(self, prediction: str|Sequence[str], reference: str|Sequence[str]) -> dict[str, float]|Sequence[float]:
        if isinstance(prediction, str):
            prediction = [prediction]
            if not isinstance(reference, str):
                raise TypeError("If prediction is a single string, reference must also be a single string.")
            reference = [reference]
        if len(prediction) != len(reference):
            raise ValueError("Prediction and reference must have the same length.")
        return func(self, prediction, reference)
    return wrapper

class Evaluator:
    def __init__(self):
        ...

    @ensure_matching_types
    def evaluate(self, prediction: Sequence[str], reference: Sequence[str]) -> dict[str, float]|Sequence[float]:
        """
        Evaluate the prediction against the reference.
        Args:
            prediction (str|Sequence[str]): The predicted caption(s).
            reference (str|Sequence[str]): The reference/ground-truth caption(s).
        Returns:
            dict[str, float]: The evaluation score.
        """
        raise NotImplementedError("Base class")

class HuggingFaceEvaluator:
    def __init__(self, *evaluators: str) -> None:
        evaluators_list = list(evaluators)
        if "bertscore" in evaluators_list:
            evaluators_list.remove("bertscore")
            self.bertscore = load("bertscore")
        else:
            self.bertscore = None

        self.evaluator = combine(evaluators_list)

    @ensure_matching_types
    def evaluate(self, prediction: Sequence[str], reference: Sequence[str]) -> dict[str, float]:
        """
        Evaluate the prediction against the reference using Hugging Face's evaluation module.
        Args:
            prediction (str|Sequence[str]): The predicted caption(s).
            reference (str|Sequence[str]): The reference/ground-truth caption(s).
        Returns:
            dict[str, float]: The evaluation score.
        """
        if self.bertscore is None:
            # It is Falsy (all EvaluationModules are, for some reason), so don't use if not self.bertscore
            return self.evaluator.compute(predictions=prediction, references=reference)
        else:
            return (self.evaluator.compute(predictions=prediction, references=reference)
                    | self.bertscore.compute(predictions=prediction, references=reference, lang="en", device="cuda:3", batch_size=512)) # type: ignore


