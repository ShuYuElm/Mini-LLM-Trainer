from torch.utils.data import Dataset

class SFTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, answer_only=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_only = answer_only


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        sample = self.dataset[index]

        instruction = sample['instruction'].strip()
        input_text = sample['input'].strip()
        output = sample['output'].strip()

        prompt = f"### Instruction:\n{instruction}\n\n"

        if input_text:
            prompt += f"### Input:\n{input_text}\n\n"

        prompt += "### Response:\n"

        text = prompt + output + "\n\n"

        if self.tokenizer.eos_token is not None:
            text += self.tokenizer.eos_token

        tokenizer_kwargs = {}

        if self.answer_only:
            tokenizer_kwargs["return_offsets_mapping"] = True

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
            **tokenizer_kwargs,
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        if self.answer_only:
            offsets = encoded["offset_mapping"].squeeze(0)
            starts = offsets[:, 0]
            ends = offsets[:, 1]

            answer_mask = (
                (ends > len(prompt))
                & (ends > starts)
            )

            labels[~answer_mask] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

