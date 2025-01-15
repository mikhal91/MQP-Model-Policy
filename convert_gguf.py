from unsloth import FastLanguageModel
import os

token = "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"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)




model.save_pretrained_gguf("model", tokenizer, quantization_method="q8_0")