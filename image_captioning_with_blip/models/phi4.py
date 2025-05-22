from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig  # type: ignore
from .base import Model
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

checkpoint = "microsoft/Phi-4-multimodal-instruct"


class PhiImageDataset(Dataset):
    checkpoint = checkpoint
    prompt = """<|user|><|image_1|>This image depicts a scene from an Indian movie.
Do not attempt to guess the name of the movie, or comment on the fact that it is a movie.
Describe the scene in detail, including the characters, their actions and clothing, and the setting:
<|end|><|assistant|>"""

    def __init__(self, images: dict[str, Image.Image]):
        filenames, _images = zip(*images.items())
        self.filenames: list[str] = list(filenames)
        self.images: list[Image.Image] = list(_images)
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(
        self, idx: int | slice | list[int]
    ) -> tuple[str | list[str], Tensor | list[Tensor]]:
        filenames = (
            [self.filenames[i] for i in idx]
            if isinstance(idx, list)
            else self.filenames[idx]
        )
        images = (
            [self.images[i] for i in idx]
            if isinstance(idx, list)
            else self.images[idx]
        )
        prompts = (
            [self.prompt] * len(images)
            if isinstance(images, list)
            else self.prompt
        )
        images = self.processor(
            images=images,
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return filenames, images


class Phi4PL(Model):
    name = "Phi4"
    checkpoint = checkpoint
    prompt = PhiImageDataset.prompt
    # Prompt attempt 1

    def __init__(self, *args, temperature: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        )
        # if you do not use Ampere or later GPUs, change attention to "eager"
        self.generation_config = GenerationConfig.from_pretrained(
            self.checkpoint, do_sample=False
        )  # ! temperature might be a bad choice to change, default is 1.0
        # (do_sample sets temperature to 0 in effect, and a warning is produced otherwise)
        self.max_new_tokens = 1024

    def predict_step(self, batch, batch_idx) -> str | list[str]:
        filenames, images = batch
        out = self.model.generate(
            **images,
            max_new_tokens=self.max_new_tokens,
            generation_config=self.generation_config,
            num_logits_to_keep=1,  # ! Won't work if not set, might really need a different number I don't know
        )
        out = out[:, images["input_ids"].shape[1] :]
        captions = self.processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        self._save(filenames, captions)
        return captions
