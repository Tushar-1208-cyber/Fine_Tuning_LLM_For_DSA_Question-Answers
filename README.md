# Fine-Tuning LLM for DSA Question Answering

## Overview

This project focuses on adapting a Large Language Model (LLM) for Data Structures and Algorithms (DSA) question answering using Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation).

The objective is to improve the model's ability to generate accurate algorithm explanations, Python implementations, and complexity analysis for common DSA problems while requiring minimal computational resources.

The project uses TinyLlama-1.1B as the base model and a custom synthetic dataset of 600 DSA question-answer pairs covering core computer science topics.

---

## Features

* Fine-tuning TinyLlama-1.1B using LoRA
* Custom DSA Question-Answer Dataset
* Parameter-Efficient Training (PEFT)
* BLEU and ROUGE-L Evaluation
* Python-based Algorithm Generation
* Lightweight Training Pipeline
* GPU-Friendly Implementation

---

## Project Structure

```text
├── dsa_dataset_600.json      # DSA training dataset
├── train.py                  # Fine-tuning pipeline
├── test.py                   # Inference script
├── metrics.py                # Evaluation metrics
├── results/                  # Training checkpoints and outputs
├── fine_tuned_model/         # Exported LoRA adapters
└── README.md
```

---

## Dataset

The dataset contains approximately 600 synthetic DSA instruction-response pairs.

### Topics Covered

* Arrays
* Strings
* Linked Lists
* Stacks
* Queues
* Trees
* Binary Search Trees
* Heaps
* Hash Tables
* Graphs
* Dynamic Programming
* Sorting Algorithms
* Time & Space Complexity Analysis

### Dataset Format

```text
Instruction: Explain Binary Search

Response:
Binary Search is an efficient searching algorithm...
```

---

## Base Model

Model Used:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

TinyLlama was selected because it provides a strong balance between performance and computational efficiency, making it suitable for experimentation on consumer-grade GPUs.

---

## LoRA Configuration

| Parameter      | Value          |
| -------------- | -------------- |
| Rank (r)       | 8              |
| Alpha          | 16             |
| Dropout        | 0.05           |
| Target Modules | q_proj, v_proj |
| Task Type      | CAUSAL_LM      |

Only a small fraction of parameters are trained while the original model weights remain frozen.

---

## Training Configuration

| Parameter           | Value |
| ------------------- | ----- |
| Epochs              | 1     |
| Batch Size          | 2     |
| Learning Rate       | 2e-4  |
| Precision           | FP16  |
| Max Sequence Length | 256   |
| Optimizer           | AdamW |

Training was performed using:

* Hugging Face Transformers
* PEFT
* PyTorch

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd Fine_Tuning_LLM_For_DSA_Question-Answers
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install transformers
pip install datasets
pip install peft
pip install accelerate
pip install torch
pip install rouge-score
pip install nltk
```

---

## Training

Run the training pipeline:

```bash
python train.py
```

The model will be fine-tuned using LoRA adapters and checkpoints will be stored inside the results directory.

---

## Inference

Generate answers using the fine-tuned model:

```bash
python test.py
```

Example Prompt:

```text
Write a Python implementation of Binary Search.
```

Example Output:

```python
def binary_search(arr, target):
    left, right = 0, len(arr)-1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

---

## Evaluation

Evaluation is performed using:

### BLEU Score

Measures similarity between generated and reference answers.

### ROUGE-L Score

Measures structural overlap and sequence similarity.

Run evaluation:

```bash
python metrics.py
```

---

## Results

| Metric                | Base Model | Fine-Tuned Model |
| --------------------- | ---------- | ---------------- |
| BLEU                  | ~0.02      | 0.0656           |
| ROUGE-L               | ~0.12      | 0.28             |
| Valid Code Generation | ~35%       | ~80%             |
| Logical Correctness   | ~15%       | ~60%             |

The fine-tuned model demonstrates significant improvements in generating executable Python code and DSA-specific explanations.

---

## Future Improvements

* Expand dataset beyond 2000 samples
* Apply QLoRA for larger models
* Add execution-based evaluation (Pass@K)
* Introduce retrieval-augmented generation (RAG)
* Support multiple programming languages

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* PEFT
* LoRA
* TinyLlama
* NLTK
* ROUGE Score

---

## Author

Tushar Gupta

B.Tech Computer Science Engineering

BML Munjal University

---

## License

This project is developed for educational and research purposes.
