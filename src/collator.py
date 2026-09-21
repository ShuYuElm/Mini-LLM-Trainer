import torch

class SFTDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)

        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for feature in features:
            current_input_length = len(feature["input_ids"])
            current_attention_length = len(feature["attention_mask"])
            current_label_length = len(feature["labels"])

            padded_input=torch.nn.functional.pad(
                feature["input_ids"],
                (0,max_len - current_input_length),
                value=self.pad_token_id
            )
            padded_attention_mask=torch.nn.functional.pad(
                feature["attention_mask"],
                (0,max_len - current_attention_length),
                value=0
            )
            padded_label=torch.nn.functional.pad(
                feature["labels"],
                (0,max_len - current_label_length),
                value=-100
            )

            padded_input_ids.append(padded_input)
            padded_attention_masks.append(padded_attention_mask)
            padded_labels.append(padded_label)

        batch_input_ids = torch.stack(padded_input_ids, dim=0)
        batch_attention_masks = torch.stack(padded_attention_masks, dim=0)
        batch_labels = torch.stack(padded_labels, dim=0)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_masks,
            "labels": batch_labels
        }


