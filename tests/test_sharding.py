from __future__ import annotations

import unittest

import torch

from training.runtime import DistributedContext
from training.sharding import build_data_parallel_model, sharding_context


class ShardingTests(unittest.TestCase):
    def test_sharding_context_classifies_fsdp(self) -> None:
        self.assertFalse(sharding_context("ddp", activation_checkpointing=False).is_fsdp)
        self.assertTrue(sharding_context("fsdp_full_shard", activation_checkpointing=False).is_fsdp)
        self.assertTrue(sharding_context("fsdp_shard_grad_op", activation_checkpointing=True).is_fsdp)

    def test_sharding_context_rejects_unknown_strategy(self) -> None:
        with self.assertRaises(SystemExit):
            sharding_context("bogus", activation_checkpointing=False)

    def test_build_data_parallel_model_returns_plain_model_when_not_distributed(self) -> None:
        model = torch.nn.Linear(8, 4)
        wrapped = build_data_parallel_model(
            base_model=model,
            distributed_context=DistributedContext(enabled=False),
            device=torch.device("cpu"),
            precision="fp32",
            sharding=sharding_context("ddp", activation_checkpointing=False),
        )
        self.assertIs(wrapped, model)

    def test_build_data_parallel_model_rejects_fsdp_without_distributed_launch(self) -> None:
        with self.assertRaises(SystemExit):
            build_data_parallel_model(
                base_model=torch.nn.Linear(8, 4),
                distributed_context=DistributedContext(enabled=False),
                device=torch.device("cpu"),
                precision="fp32",
                sharding=sharding_context("fsdp_full_shard", activation_checkpointing=False),
            )


if __name__ == "__main__":
    unittest.main()
