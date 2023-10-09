from torch.utils.data import DataLoader

class SamplingDataLoader(DataLoader):
    def __init__(self, dataset, tokenizer, text_encoder, device, batch_size=1, **kwargs):
        super().__init__(dataset, collate_fn=self.collate_fn, batch_size=batch_size, **kwargs)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device

    def collate_fn(self, examples):
        prompts = [example["prompt"] for example in examples]
        images = [example["image"] for example in examples]
        masks = [example["mask"] for example in examples]

        prompt_ids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids.to(self.device)

        prompt_embeds = self.text_encoder(prompt_ids)[0]
        
        batch = {
            "prompts": prompts,
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "images": images,
            "masks": masks,
        }

        return batch