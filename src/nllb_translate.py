from __future__ import annotations

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import srt

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Often helps CPU perf on Windows by reducing thread contention.
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

class NllbTranslator:
    def __init__(self, src_lang: str, tgt_lang: str = "eng_Latn", device: str = "cpu"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        self.model.eval()

    @torch.inference_mode()
    def _translate_text(self, text: str, max_new_tokens: int = 160) -> str:
        self.tokenizer.src_lang = self.src_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        output = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=2,
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

    def translate_srt(self, srt_text: str, batch_size: int = 20) -> str:
        subs = list(srt.parse(srt_text))

        for i in range(0, len(subs), batch_size):
            batch = subs[i : i + batch_size]
            lines = [b.content.replace("\r\n", "\n").replace("\r", "\n") for b in batch]

            joined = "\n".join(lines)
            translated = self._translate_text(joined).split("\n")

            if len(translated) != len(lines):
                translated = [self._translate_text(line, max_new_tokens=64) for line in lines]

            for sub, t in zip(batch, translated):
                sub.content = t

        return srt.compose(subs)
