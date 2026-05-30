from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate

# Load model
model_path = "./fine_tuned_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Sample test dataset
test_data = [
    {"input": "What is stack?", "output": "A stack is a linear data structure that follows LIFO."},
    {"input": "What is queue?", "output": "A queue is a linear data structure that follows FIFO."},
    {"input": "What is binary search?", "output": "Binary search is an efficient algorithm used on sorted arrays."}
]

predictions = []
references = []

for item in test_data:
    prompt = f"### Question:\n{item['input']}\n\n### Answer:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(**inputs, max_new_tokens=100)

    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    pred = pred.split("### Answer:")[-1].strip()

    predictions.append(pred)
    references.append([item["output"]])

# Calculate BLEU
bleu_score = bleu.compute(predictions=predictions, references=references)

# Calculate ROUGE
rouge_score = rouge.compute(predictions=predictions, references=[r[0] for r in references])

print("\n===== METRICS =====")
print("BLEU Score:", bleu_score)
print("ROUGE Score:", rouge_score)