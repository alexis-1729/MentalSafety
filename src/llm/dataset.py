from dataset import load_dataset


def get_dataset_llm(config):
    dataset = load_dataset("json", data_files = config['path_data'], split = "train")

    texts = []

    for i in range(len(dataset['prompt'])):
        text = f"### User: \n {dataset['prompt'][i]}\n ### Asistant: \n {dataset['response'][i]} <|endoftext|>"
        texts.append(text)
    
    dataset = dataset.map(texts, batched = True)

    return dataset