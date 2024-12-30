from unsloth import FastLanguageModel

model = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
print("Model loaded successfully.")
