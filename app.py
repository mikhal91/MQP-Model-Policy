from unsloth import FastLanguageModel
import torch
import os
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True
alpaca_prompt = """Below is a block of text and its extracted causal relationships. Use the information provided to learn to extract causal variables and relationships.

### Input:
{}

### Variables:
{}

### Relationships:
{}
"""

huggingface_model_name = "miikhal/Llama-3.1-8B-extraction"

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for input_text, output in zip(inputs, outputs):
        variable_text = "\n".join(output["variables"])
        relationship_text = "\n".join([f"{rel['cause']} -> {rel['effect']}" for rel in output["relationships"]])
        text = alpaca_prompt.format(input_text, variable_text, relationship_text)
        texts.append(text)
    return {"text": texts}

if __name__ == '__main__':
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        use_xformers=True,  # Ensure xformers is enabled
        token=os.getenv("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")
    )

    # If xformers causes issues, disable it explicitly
    try:
        model.config.use_xformers = True
    except AttributeError:
        print("xformers is not available; disabling xformers optimizations.")
        model.config.use_xformers = False

    FastLanguageModel.for_inference(model)

    # Load dataset
    dataset = load_dataset("miikhal/mqp-immigration-dataset", split="train")

    # Apply formatting function with num_proc=1 for debugging
    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1, remove_columns=dataset.column_names)

    # Training setup
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,  # Debugging without multiprocessing
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # Train
    trainer.train()

    # Save the model
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    model.push_to_hub(huggingface_model_name, token=os.getenv("HF_TOKEN"))
    tokenizer.push_to_hub(huggingface_model_name, token=os.getenv("HF_TOKEN"))

    # Merge to 16bit
    if True:
        model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    if True:
        model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method="merged_16bit", token=os.getenv("HF_TOKEN"))

    # # Merge to 4bit
    # if True:
    #     model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
    # if True:
    #     model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method="merged_4bit", token=os.getenv("HF_TOKEN"))

    # # Just LoRA adapters
    # if True:
    #     model.save_pretrained_merged("model", tokenizer, save_method="lora")
    # if True:
    #     model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method="lora", token=os.getenv("HF_TOKEN"))

    # # Save to 8bit Q8_0
    # if True:
    #     model.save_pretrained_gguf("model", tokenizer)
    # if True:
    #     model.push_to_hub_gguf(huggingface_model_name, tokenizer, token=os.getenv("HF_TOKEN"))

    # # Save to 16bit GGUF
    # if True:
    #     model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
    # if True:
    #     model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method="f16", token=os.getenv("HF_TOKEN"))

    # # Save to q4_k_m GGUF
    # if True:
    #     model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
    # if True:
    #     model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method="q4_k_m", token=os.getenv("HF_TOKEN"))

    # # Save to multiple GGUF options - much faster if you want multiple!
    # if True:
    #     model.push_to_hub_gguf(
    #         huggingface_model_name,  # Change hf to your username!
    #         tokenizer,
    #         quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
    #         token=os.getenv("HF_TOKEN")
    #     )
