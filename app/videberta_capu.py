from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


CASES = ("KEEP", "CAPITAL", "UPPER")
PUNCTUATION = ("NONE", ".", ",", ":", "?")
LABELS = tuple(
    f"{case}|{punctuation}"
    for case in CASES
    for punctuation in PUNCTUATION
)
TRAILING_PUNCTUATION = re.compile(r"[,.:?]+$")


class ViDeBERTaCapu:
    """Restore Vietnamese capitalization and punctuation with an ONNX model.

    The model predicts one combined case/punctuation label for the first
    sub-token of every whitespace-delimited input word. Output deliberately
    keeps the same word count as input so callers can safely slice off context.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        model_name: str = "model.int8.onnx",
        threads: int = 4,
        max_length: int = 512,
        punctuation_none_bias: float = 0.0,
        case_keep_bias: float = 0.0,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        model_path = self.model_dir / model_name
        if not model_path.is_file():
            raise FileNotFoundError(f"CAPU ONNX model not found: {model_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=True,
        )
        self.model_name = model_name
        self.max_length = max_length
        self.punctuation_none_bias = punctuation_none_bias
        self.case_keep_bias = case_keep_bias

    @staticmethod
    def _apply_case(word: str, case: str) -> str:
        if case == "UPPER":
            return word.upper()
        if case != "CAPITAL":
            return word
        for index, character in enumerate(word):
            if character.isalpha():
                return word[:index] + character.upper() + word[index + 1:]
        return word

    def __call__(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return [""]

        encoded = self.tokenizer(
            [words],
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        logits = self.session.run(
            ["logits"],
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )[0][0]

        if self.punctuation_none_bias:
            logits[:, [0, 5, 10]] += self.punctuation_none_bias
        if self.case_keep_bias:
            logits[:, 0:5] += self.case_keep_bias

        predictions: dict[int, int] = {}
        for token_index, word_id in enumerate(encoded.word_ids(0)):
            if word_id is not None and word_id not in predictions:
                predictions[word_id] = int(logits[token_index].argmax())

        output: list[str] = []
        for word_id, original_word in enumerate(words):
            label_id = predictions.get(word_id, 0)
            case, punctuation = LABELS[label_id].split("|", 1)
            existing = TRAILING_PUNCTUATION.search(original_word)
            bare_word = original_word[:existing.start()] if existing else original_word
            restored = self._apply_case(bare_word, case)
            # Punctuation labels were trained from GEC `$APPEND_*` actions.
            # Apply them as edits: existing punctuation is input content and
            # must never be deleted/replaced by a new prediction.
            if existing:
                restored += existing.group(0)
            elif punctuation != "NONE":
                restored += punctuation
            output.append(restored)
        return [" ".join(output)]
