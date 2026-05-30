from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./fine_tuned_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("\n🤖 DSA Chatbot Ready! (type 'exit' to stop)\n")

while True:
    user_input = input("👉 Ask Question: ")

    if user_input.lower() == "exit":
        break

    prompt = f"### Question:\n{user_input}\n\n### Answer:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,   # 🔥 IMPORTANT (stable output)
        num_beams=3        # 🔥 better answer quality
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    answer = result.split("### Answer:")[-1].strip()

    print("\n🤖 Answer:\n", answer)
    print("\n" + "-"*50)