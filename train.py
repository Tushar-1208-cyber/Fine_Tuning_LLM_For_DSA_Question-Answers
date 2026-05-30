print("🚀 START")

# Step 1: Load dataset
from datasets import load_dataset

dataset = load_dataset("json", data_files="dsa_dataset_600.json")
print("✅ Dataset loaded:", dataset)

# Step 2: Format dataset
def format(example):
    return {
        "text": f"### Question:\n{example['input']}\n\n### Answer:\n{example['output']}"
    }

dataset = dataset.map(format)
print("✅ Dataset formatted")

# Step 3: Load model + tokenizer (GPU enabled)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("⏳ Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # 🔥 GPU use karega
    torch_dtype="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# IMPORTANT FIX
tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded on GPU")

# 🔥 STEP 4: TOKENIZE DATASET
def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize)
print("✅ Dataset tokenized")

# Step 5: LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)
print("✅ LoRA applied")

# Step 6: Training (GPU optimized)
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,   # 🔥 GPU pe increase
    logging_steps=10,
    fp16=True,                      # 🔥 speed boost
    save_steps=200,
    save_total_limit=2
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args
)

print("🔥 Training start on GPU...")
trainer.train()

# Step 7: Save
trainer.save_model("./fine_tuned_model")
print("✅ Model saved")

print("🎉 END")