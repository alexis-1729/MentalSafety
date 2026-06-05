from transformers import TrainnigArguments
from trl import SFFTrainer
import torch

def init_trainer(config_trainer, model, tokenizer):
    trainer = SFFTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing= False,
        args = TrainnigArguments(
            per_device_train_batch_size = config_trainer['batch_size'],
            gradient_accumulation_steps = config_trainer['acumulation_steps'],
            warmup_steps = 5,
            max_steps = config_trainer['max_steps'],
            learning_rate = config_trainer['learning_rate'],
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "outputs",
            optim = config_trainer['optim'].
            seed = 0
        )
    )

    return trainer.train()