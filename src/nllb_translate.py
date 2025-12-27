from __future__ import annotations

import os
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import srt

MODEL_NAME = "facebook/nllb-200-distilled-600M"

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

def warmup(device: str = "cpu") -> None:
    _get_shared(device)

_shared_lock = threading.Lock()
_shared = {}

def _get_shared(device: str):
    with _shared_lock:
        obj = _shared.get(device)
        if obj is not None:
            return obj

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()

        obj = {"tokenizer": tokenizer, "model": model}
        _shared[device] = obj
        return obj

class NllbTranslator:
    def __init__(self, src_lang: str, tgt_lang: str = "eng_Latn", device: str = "cpu"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device

        shared = _get_shared(device)
        self.tokenizer = shared["tokenizer"]
        self.model = shared["model"]

    @torch.inference_mode()
    def _translate_text(self, text: str, max_new_tokens: int = 160) -> str:
        # IMPORTANT: avoid generate() on empty/whitespace-only input
        if not text or not text.strip():
            return ""

        self.tokenizer.src_lang = self.src_lang

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        # Some edge cases can produce 0-length input_ids -> generate() can crash
        if "input_ids" not in inputs or inputs["input_ids"].shape[1] == 0:
            return ""

        # NLLB tokenizers have a lang_code_to_id map; this is safer than convert_tokens_to_ids
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        if forced_bos_token_id is None:
            raise RuntimeError(f"Unknown target language code: {self.tgt_lang}")

        output = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=2,
        )

        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    def translate_srt(self, srt_text: str, max_tokens: int = 400) -> str:
        subs = list(srt.parse(srt_text))
        out = []

        current = []
        current_tokens = 0

        for sub in subs:
            text = sub.content.replace("\r\n", "\n").replace("\r", "\n")
            token_count = len(self.tokenizer.tokenize(text))

            # flush batch if too big
            if current and current_tokens + token_count > max_tokens:
                out.extend(self._translate_batch(current))
                current = []
                current_tokens = 0

            current.append(sub)
            current_tokens += token_count

        if current:
            out.extend(self._translate_batch(current))

        return srt.compose(out)

    def _translate_batch(self, subs):
        lines = [s.content.replace("\r\n", "\n").replace("\r", "\n") for s in subs]

        # If the whole batch is empty, just return it unchanged
        if not any(line.strip() for line in lines):
            return subs

        joined = "\n".join(lines)

        translated_joined = self._translate_text(joined)

        # splitlines() drops trailing empty lines unless we handle it manually
        translated = translated_joined.split("\n")

        # fallback: per-line translation (handles count mismatches)
        if len(translated) != len(lines):
            translated = []
            for line in lines:
                if not line.strip():
                    translated.append("")  # keep empty lines empty
                else:
                    translated.append(self._translate_text(line, max_new_tokens=64))

        for sub, t in zip(subs, translated):
            sub.content = t

        return subs
