# MentalSafety

MentalSafety is a Python-based project focused on mental health and safety classification using NLP and LLM fine-tuning. The repository combines:

- a transformer-based text classification pipeline for behavioral or safety-related text analysis
- LLM fine-tuning with parameter-efficient adaptation (LoRA) for instruction-based safety responses
- configuration-driven experimentation for training and evaluation

## Overview

This project aims to support the identification and response to potentially harmful, unsafe, or sensitive conversational content related to mental health. The repository is structured around two complementary approaches:

1. NLP classification using a BERT-style model
2. LLM adaptation for instruction-following and safety-oriented generation

## Repository structure

```text
MentalSafety/
├── config/
│   ├── llm_config.yaml
│   └── nlp_config.yaml
├── src/
│   └── llm/
│       ├── dataset.py
│       └── model_utils.py
├── finetune_llm.py
├── train_sentiment.py
├── evaluate_pipeline.py
├── requirements.txt
├── README.md
└── data/
    ├── dataset.jsonl
    └── data.csv
```

## Main components

### `config/nlp_config.yaml`
Defines the configuration for the NLP text classification pipeline, including:

- input data path
- tokenizer model (`bert-base-uncased`)
- maximum sequence length
- training hyperparameters such as learning rate, batch size, epochs, and weight decay

### `config/llm_config.yaml`
Defines the LLM fine-tuning configuration, including:

- data path for JSONL instruction data
- base model (`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`)
- LoRA parameters
- training settings such as batch size, accumulation steps, learning rate, and optimizer

### `src/llm/dataset.py`
Prepares the instruction-tuning dataset by converting prompt-response pairs into chat-style text for the LLM.

### `src/llm/model_utils.py`
Loads the base model with Unsloth and applies a LoRA adaptation layer for efficient fine-tuning.

### `finetune_llm.py`
Contains the training setup for the LoRA-fine-tuned LLM workflow.

### `train_sentiment.py`
Intended for training the sentiment/safety classification model.

### `evaluate_pipeline.py`
Designed to evaluate the model pipeline after training.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Typical dependencies for this project include:

- Python >= 3.10
- PyTorch
- Transformers
- Datasets
- TRL
- Unsloth
- scikit-learn
- pandas
- NumPy

If `requirements.txt` is still empty in the repository, you can install the main packages manually:

```bash
pip install torch transformers datasets trl unsloth pandas scikit-learn numpy
```

## Data format

### NLP classification data
The project expects a CSV file for classification tasks, configured in `config/nlp_config.yaml`:

```text
data/data.csv
```

Typical structure:

```csv
text,label
"I feel extremely anxious and hopeless.",1
"I am doing well and feel calm.",0
```

### LLM instruction data
The LLM pipeline expects a JSONL file configured in `config/llm_config.yaml`:

```text
data/dataset.jsonl
```

Example:

```json
{"prompt": "I feel like I cannot cope anymore.", "response": "You are not alone. Please reach out to a trusted person or a mental health professional."}
```

## Training

### 1) NLP classifier
Use the training script for the sentiment/safety classifier:

```bash
python train_sentiment.py
```

The training configuration is controlled by `config/nlp_config.yaml`.

### 2) LLM fine-tuning
Fine-tune the LLM using the configuration in `config/llm_config.yaml`:

```bash
python finetune_llm.py
```

This pipeline uses LoRA-based adaptation to efficiently fine-tune a base instruction model.

## Evaluation

Evaluation is intended to be carried out through the pipeline script:

```bash
python evaluate_pipeline.py
```

This script is intended to assess the trained classification or generation pipeline depending on the active workflow.

## Notes

- This repository appears to be in an experimental or early development stage.
- Some files such as `train_sentiment.py` and `evaluate_pipeline.py` may be placeholders or incomplete scripts.
- Exact runtime behavior will depend on the dataset files and the installed library versions.
- The LLM configuration uses Unsloth with 4-bit loading and LoRA, which is suitable for resource-efficient fine-tuning.

## Future improvements

- add a standard training CLI
- improve evaluation reporting and metrics
- integrate model checkpoints and artifact tracking
- add safe-response filtering and explainability tools
- document dataset labeling criteria for mental health classification

## License

No explicit license file is present in the repository yet. If you plan to publish or share it publicly, consider adding a license such as MIT or Apache 2.0.

## Contributing

Contributions are welcome. If you want to improve the project, open a pull request with a description of the change and the validation steps used.

## Contact

For questions or collaboration, contact the repository owner or project maintainer.
