from __future__ import annotations

import torch
from src.training.sft_semantic import semantic_completion_mask
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset




class SemanticMultiTurnSFTDataset(MultiTurnSFTDataset):
    """veRL SFT dataset that adds a loss mask for the four JSON semantic values."""

    def _build_messages(self, example):
        images = example.get("images")
        if isinstance(images, list):
            example = dict(example)
            example["images"] = [
                {"image_url": path} if isinstance(path, str) else path
                for path in images
            ]
        return super()._build_messages(example)


    def __getitem__(self, item):
        sample = super().__getitem__(item)
        input_ids = sample["input_ids"]
        loss_mask = sample["loss_mask"]
        assistant_positions = torch.nonzero(loss_mask.to(bool), as_tuple=False).flatten()
        if assistant_positions.numel() == 0:
            raise ValueError(f"SFT sample {item} has no assistant completion tokens")

        assistant_ids = input_ids[assistant_positions].tolist()
        assistant_semantic_mask = torch.tensor(
            semantic_completion_mask(self.tokenizer, assistant_ids),
            dtype=loss_mask.dtype,
            device=loss_mask.device,
        )
        semantic_loss_mask = torch.zeros_like(loss_mask)
        semantic_loss_mask[assistant_positions] = assistant_semantic_mask

        if not torch.any(semantic_loss_mask):
            raise ValueError(f"SFT sample {item} has no recognized semantic JSON value tokens")
        if torch.any(semantic_loss_mask.to(bool) & ~loss_mask.to(bool)):
            raise AssertionError("semantic loss mask must be a subset of the assistant loss mask")

        sample["semantic_loss_mask"] = semantic_loss_mask
        return sample
