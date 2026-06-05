from unsloth import FastLanguageModel

def prepare_model(config_model):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config_model['base_model_id'],
        max_seq_length= config_model['max_seq_length'],
        load_in_4bit= True
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = config_model['r']
        target_modules= config_model['target_modules'],
        lora_alpha= config_model['lora_alpha'],
        lora_dropout= config_model['lora_dropout'],
        bias= config_model['bias'],
        use_gradient_checkpointing= config_model['gradient_checkpoint'],
    )

    return model, tokenizer