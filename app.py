from unsloth import FastLanguageModel
import torch
import os
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from huggingface_hub import HfApi, HfFolder

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True 
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

token = "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"  ## Token from huggingface for the AI Model
huggingface_model_name = "miikhal/Llama-3.1-8B-python-mqp"

# 2. Before Training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = token
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# 3. Load data

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["Instruction"]
    inputs       = examples["Input"]
    responses    = examples["Response"]
    texts = []
    for instruction, input_text, response in zip(instructions, inputs, responses):
        text = alpaca_prompt.format(instruction, input_text, response) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("miikhal/mqp-immigration-dataset", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 4. Training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# 5. After Training
FastLanguageModel.for_inference(model) # Enable native 2x faster  
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Identify the causal relationships and key variables in the text.", # Replace with a valid instruction if testing
        "The greatest upward pressure on prices comes from increased demand for housing.", # Replace with a valid input if testing
        "", # Leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000)

# 6. Saving
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
model.push_to_hub(huggingface_model_name, token = token)
tokenizer.push_to_hub(huggingface_model_name, token = token)



## Note with merging; only run first half 'save...' here and run upload model after this finishes. Upload lines work but dont show progress (takes an ungodly amount of time) Dont use q4_k_m or q5_k_m it doesnt work with my current hardware.


# Merge to 16bit
if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)

# Save to 8bit Q8_0
if True:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q8_0")



## if True: model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method = "merged_16bit", token = token) ## Code breaks and doesnt do anything for like at least 30 mins. File size for first file was 4.9GB and it didnt upload at all stayed at 0%.

# # Merge to 4bit
# if True:
#     model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit",)
# `if True:
#     model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method="merged_4bit", token=token)



# # Just LoRA adapters
# if True:
#     model.save_pretrained_merged("model", tokenizer, save_method="lora",)
# if True:
#     model.push_to_hub_merged(huggingface_model_name, tokenizer, save_method="lora", token=token)

# Save to 8bit Q8_0
#if True:
#    model.save_pretrained_gguf("model", tokenizer,)
#if True:
#    model.push_to_hub_gguf(huggingface_model_name, tokenizer, token=token)

# # Save to 16bit GGUF
# if True:
#     model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
# if True:
#     model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method="f16", token=token)

# # Save to q4_k_m GGUF
# if True:
#     model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
# if True:
#     model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method="q4_k_m", token=token)

# # Save to multiple GGUF options - much faster if you want multiple!
# if True:
#     model.push_to_hub_gguf(
#         huggingface_model_name, # Change hf to your username!
#         tokenizer,
#         quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
#         token = token
#     )