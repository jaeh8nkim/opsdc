"""
OPSD (On-Policy Self-Distillation) Trainer using VERL's HybridEngine.

Orchestrates the OPSD training loop:
  1. Generate: Use sft_prompt (question-only) for student rollout generation
  2. Verify: Check math correctness for metrics (but do NOT filter)
  3. Train: JSD update on ALL rollouts using frozen teacher (ref model)

Key differences from SelfDistillTrainer (sd_trainer.py):
  - Generation uses sft_prompt (question-only), not sd_prompt (with teacher solution)
  - Training uses JSD loss between teacher and student logit distributions
  - ALL rollouts are trained on, not just correct ones
  - Teacher model is the frozen ref_module_fsdp (initial policy weights)
"""

import glob
import json
import logging
import math
import os

logger = logging.getLogger(__name__)
import time
import uuid
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics

from .kl_probe import (
    KLProbeAccumulator,
    KLProbeConfig,
    find_think_close,
    snap_to_boundary,
)
from .opsd_config_validation import (
    compute_use_reverse_kl_sample,
    validate_opsd_config,
)
from .sd_verifier import (
    build_opsd_batch,
    build_opsd_batch_multipass,
    build_sft_batch,
    verify_batch,
)

py_logger = logging.getLogger(__name__)


# Teacher context modes — see opsd_trainer.yaml::opsd.teacher_ctx_mode for semantics.
TEACHER_CTX_MODES = frozenset({"sd_prompt", "reflection_from_gt", "gt_directly", "conciseness_instruction"})
# Subset that requires a Turn 2 generation pass before the OPSD update.
MODES_NEEDING_TURN2 = frozenset({"reflection_from_gt"})


class OPSDTrainer:
    """OPSD trainer: JSD-based on-policy self-distillation.

    The student generates from question-only prompts, and training minimizes
    the JSD divergence between teacher (frozen ref model) and student
    distributions on ALL student rollouts (not just correct ones).

    Args:
        config: OmegaConf config (based on opsd_trainer.yaml).
        tokenizer: HuggingFace tokenizer.
        role_worker_mapping: Mapping from roles to worker classes.
        resource_pool_manager: Manager for Ray resource pools.
        ray_worker_group_cls: Class for Ray worker groups.
        processor: Optional multimodal processor.
        train_dataset: SD prompt dataset.
        collate_fn: Batch collation function.
        device_name: Device name for training.
        val_data_path: Path to SD val parquet for _compute_val_loss().
        val_reward_fn: Reward manager for generation-based validation.
        val_dataset: RL-format dataset for generation-based validation.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, type],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        collate_fn=None,
        device_name=None,
        val_data_path: Optional[str] = None,
        val_reward_fn=None,
        val_dataset: Optional[Dataset] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "OPSDTrainer requires hybrid_engine=True"
        assert Role.ActorRolloutRef in role_worker_mapping, (
            f"OPSDTrainer requires ActorRolloutRef in role_worker_mapping (for ref model), "
            f"got {list(role_worker_mapping.keys())}"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # OPSD-specific config
        self.opsd_config = self.config.get("opsd", {})
        self.beta = self.opsd_config.get("beta", 0.5)
        # Validate loss_type / kl_gating / truncated_handling via shared utility.
        # Raises on invalid; also enforces branched KL's (kl_gating='all' + required
        # truncated_handling) requirement.
        self.loss_type, self.kl_gating, self.truncated_handling = (
            validate_opsd_config(self.opsd_config)
        )
        self.sft_max_length = self.opsd_config.get("sft_max_length", 32768)
        self.check_structure = self.opsd_config.get("check_structure", True)
        self.log_sample_count = self.opsd_config.get("log_sample_count", 5)
        self.test_freq = self.opsd_config.get("test_freq", 10)
        self.log_freq = self.opsd_config.get("log_freq", 5)
        self.teacher_update_freq = self.opsd_config.get("teacher_update_freq", 0) or 0
        self.teacher_ctx_mode = self.opsd_config.get("teacher_ctx_mode", "sd_prompt")
        assert self.teacher_ctx_mode in TEACHER_CTX_MODES, (
            f"Invalid teacher_ctx_mode: {self.teacher_ctx_mode} "
            f"(expected one of {sorted(TEACHER_CTX_MODES)})"
        )

        # ---- KL-by-position probe config ----
        kl_by_pos_cfg = self.opsd_config.get("kl_by_pos", {}) or {}
        self.kl_probe_cfg = KLProbeConfig(
            enabled=bool(kl_by_pos_cfg.get("enabled", True)),
            plot_freq=int(kl_by_pos_cfg.get("plot_freq", 50)),
            n_abs_bins=int(kl_by_pos_cfg.get("n_abs_bins", 64)),
            max_position=int(kl_by_pos_cfg.get("max_position", 8192)),
            min_tokens_per_bin=int(kl_by_pos_cfg.get("min_tokens_per_bin", 30)),
        )
        self.kl_analyze_only = bool(self.opsd_config.get("kl_analyze_only", False))
        self.kl_probe = KLProbeAccumulator(self.kl_probe_cfg)

        # ---- Distance-weighting config ----
        dw_cfg = self.opsd_config.get("distance_weighting", {}) or {}
        self.distance_weight_schedule = str(dw_cfg.get("schedule", "off"))
        self.distance_weight_late_multiplier = float(dw_cfg.get("late_multiplier", 2.0))
        self.distance_weight_alpha = float(dw_cfg.get("alpha", 2.0))
        self.distance_weight_per_sample_norm = bool(dw_cfg.get("per_sample_norm", True))
        assert self.distance_weight_schedule in ("off", "linear", "exp", "step"), (
            f"Invalid distance_weighting.schedule: {self.distance_weight_schedule}"
        )

        # ---- Reinjection config ----
        rj_cfg = self.opsd_config.get("teacher_ctx_reinjection", {}) or {}
        self.reinjection_enabled = bool(rj_cfg.get("enabled", False))
        self.reinjection_interval = int(rj_cfg.get("interval", 2048))
        self.reinjection_content = str(rj_cfg.get("content", "specific_context"))
        self.reinjection_wrapper = str(rj_cfg.get("wrapper", "natural"))
        self.reinjection_mode = str(rj_cfg.get("mode", "multi_pass"))
        assert self.reinjection_content in (
            "specific_context", "conciseness_instruction",
        ), f"Invalid reinjection content: {self.reinjection_content}"
        assert self.reinjection_wrapper == "natural", (
            f"Only wrapper=natural is supported (got {self.reinjection_wrapper})"
        )
        assert self.reinjection_mode in ("cumulative", "multi_pass"), (
            f"Invalid reinjection_mode: {self.reinjection_mode}"
        )

        # Detailed logging
        self.log_dir = self.opsd_config.get(
            "detailed_log_dir",
            os.path.join(
                self.config.trainer.get("default_local_dir", "checkpoints"),
                "detailed_logs",
            ),
        )
        self.rollout_log_dir = os.path.join(self.log_dir, "rollout")
        self.opsd_log_dir = os.path.join(self.log_dir, "opsd")
        self.val_log_dir = os.path.join(self.log_dir, "val_generations")
        self.epiphany_log_dir = os.path.join(self.log_dir, "epiphany")
        # KL probe directories (PDFs + .npz, plus per-rank staging files).
        session_root = os.path.dirname(self.log_dir.rstrip("/")) or self.log_dir
        self.kl_plots_dir = os.path.join(session_root, "kl_plots")
        self.kl_probe_staging_dir = os.path.join(self.kl_plots_dir, ".staging")
        os.makedirs(self.rollout_log_dir, exist_ok=True)
        os.makedirs(self.opsd_log_dir, exist_ok=True)
        os.makedirs(self.val_log_dir, exist_ok=True)
        os.makedirs(self.epiphany_log_dir, exist_ok=True)
        if self.kl_probe_cfg.enabled:
            os.makedirs(self.kl_plots_dir, exist_ok=True)
            os.makedirs(self.kl_probe_staging_dir, exist_ok=True)
        py_logger.info("Rollout logs -> %s", self.rollout_log_dir)
        py_logger.info("OPSD logs    -> %s", self.opsd_log_dir)
        py_logger.info("Val gen logs -> %s", self.val_log_dir)
        py_logger.info("Epiphany logs-> %s", self.epiphany_log_dir)
        py_logger.info("KL plots     -> %s", self.kl_plots_dir)

        # Load expert demonstrations (optional)
        expert_demo_path = self.opsd_config.get("expert_demo_path", None)
        if expert_demo_path:
            import pandas as _pd
            expert_df = _pd.read_parquet(expert_demo_path)
            self.expert_demos = {}
            for _, row in expert_df.iterrows():
                self.expert_demos[row["question"]] = row["expert_demonstration"]
            py_logger.info("Loaded %d expert demonstrations from %s", len(self.expert_demos), expert_demo_path)
        else:
            self.expert_demos = None

        # Create dataloader
        self._create_dataloader(train_dataset, collate_fn)

        # Build pre-tokenized validation batch from held-out teacher data
        self._build_val_data(val_data_path)

        # Generation-based validation
        self.val_reward_fn = val_reward_fn
        self._build_val_dataloader(val_dataset)

    # ------------------------------------------------------------------
    # Initialization helpers (same patterns as SelfDistillTrainer)
    # ------------------------------------------------------------------

    def _create_dataloader(self, train_dataset: Optional[Dataset], collate_fn):
        if train_dataset is None:
            raise ValueError("train_dataset must be provided for OPSDTrainer")

        self.train_dataset = train_dataset

        if collate_fn is None:
            from .sd_dataset import collate_fn as sd_collate_fn
            collate_fn = sd_collate_fn

        batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        num_workers = self.config.data.get("dataloader_num_workers", 0)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            shuffle=self.config.data.get("shuffle", True),
        )
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.get("total_training_steps") is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        py_logger.info("Dataloader batches: %d", len(self.train_dataloader))
        py_logger.info("Total training steps: %d", self.total_training_steps)

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as e:
            py_logger.warning("Could not set total_training_steps in config: %s", e)

    def _build_val_data(self, val_data_path: Optional[str]):
        """Pre-tokenize held-out validation data for periodic val loss computation."""
        import pandas as pd

        self.val_batches = None

        if not val_data_path or not os.path.exists(val_data_path):
            py_logger.info("No val_data_path or file missing, val loss disabled")
            return

        val_df = pd.read_parquet(val_data_path)
        val_max_samples = self.opsd_config.get("val_max_samples", -1)
        if val_max_samples > 0 and len(val_df) > val_max_samples:
            val_df = val_df.head(val_max_samples)
        py_logger.info("Loading %d val samples from %s", len(val_df), val_data_path)

        sft_prompts = val_df["sft_prompt"].tolist()
        teacher_solutions = val_df["teacher_solution"].tolist()

        full_val_batch = build_sft_batch(
            sft_prompts=sft_prompts,
            responses=teacher_solutions,
            tokenizer=self.tokenizer,
            max_length=self.sft_max_length,
        )

        if full_val_batch is None or len(full_val_batch.batch["input_ids"]) == 0:
            py_logger.info("Val batch is empty after tokenization, val loss disabled")
            return

        n_val = full_val_batch.batch["input_ids"].shape[0]
        n_dp = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes

        if n_val % n_dp != 0:
            pad_to = ((n_val // n_dp) + 1) * n_dp
            pad_count = pad_to - n_val
            padded_dict = {}
            for key in ["input_ids", "attention_mask", "position_ids", "loss_mask"]:
                tensor = full_val_batch.batch[key]
                last = tensor[-1:].expand(pad_count, *tensor.shape[1:]).clone()
                padded_dict[key] = torch.cat([tensor, last], dim=0)
            full_val_batch = DataProto.from_single_dict(padded_dict)
            n_val = full_val_batch.batch["input_ids"].shape[0]

        dispatch_batch_size = self.config.data.train_batch_size
        self.val_batches = []
        for start in range(0, n_val, dispatch_batch_size):
            end = min(start + dispatch_batch_size, n_val)
            chunk_dict = {}
            for key in ["input_ids", "attention_mask", "position_ids", "loss_mask"]:
                chunk_dict[key] = full_val_batch.batch[key][start:end]
            chunk_size = chunk_dict["input_ids"].shape[0]
            if chunk_size % n_dp != 0:
                chunk_pad_to = ((chunk_size // n_dp) + 1) * n_dp
                chunk_pad_count = chunk_pad_to - chunk_size
                for key in chunk_dict:
                    last = chunk_dict[key][-1:].expand(chunk_pad_count, *chunk_dict[key].shape[1:]).clone()
                    chunk_dict[key] = torch.cat([chunk_dict[key], last], dim=0)
            self.val_batches.append(DataProto.from_single_dict(chunk_dict))

        py_logger.info(
            "Val data ready: %d samples, %d batch(es), test_freq=%d",
            n_val, len(self.val_batches), self.test_freq,
        )

    def _build_val_dataloader(self, val_dataset: Optional[Dataset]):
        self.val_dataloader = None
        if val_dataset is None or self.val_reward_fn is None:
            return

        from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn

        val_batch_size = self.config.data.get("val_batch_size", None)
        if val_batch_size is None:
            val_batch_size = len(val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            collate_fn=rl_collate_fn,
        )

        py_logger.info(
            "Generation-based val dataloader ready: %d samples, %d batch(es)",
            len(val_dataset), len(self.val_dataloader),
        )

    def _compute_val_loss(self) -> Optional[float]:
        if not self.val_batches:
            return None

        total_loss = 0.0
        total_tokens = 0

        for val_batch in self.val_batches:
            output = self.actor_rollout_wg.compute_val_loss(val_batch)
            raw_metrics = output.meta_info.get("metrics", {})

            losses = raw_metrics.get("val_loss", [])
            tokens = raw_metrics.get("val_tokens", [])
            if not isinstance(losses, list):
                losses = [losses]
            if not isinstance(tokens, list):
                tokens = [tokens]

            for loss_i, tok_i in zip(losses, tokens):
                total_loss += float(loss_i) * float(tok_i)
                total_tokens += float(tok_i)

        if total_tokens == 0:
            return None
        return total_loss / total_tokens

    def _dump_val_generations(
        self, inputs, outputs, scores, reward_extra_infos_dict,
        data_sources=None, token_counts=None,
    ):
        """Dump validation samples as JSONL, one file per global step."""
        filename = os.path.join(
            self.val_log_dir, f"step_{self.global_steps:06d}.jsonl"
        )

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        if data_sources is not None and len(data_sources) == n:
            base_data["data_source"] = list(data_sources)
        if token_counts is not None and len(token_counts) == n:
            base_data["response_tokens"] = token_counts

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        py_logger.info(
            "Dumped %d val generations to %s", n, filename
        )

    def _validate(self) -> dict:
        """Generate responses on val prompts and evaluate with reward function."""
        if self.val_dataloader is None or self.val_reward_fn is None:
            return {}

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_uids = []
        sample_token_counts = []

        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
        val_n = val_kwargs.get("n", 1)
        val_do_sample = val_kwargs.get("do_sample", False)
        val_temperature = val_kwargs.get("temperature", None)
        val_top_p = val_kwargs.get("top_p", None)
        val_top_k = val_kwargs.get("top_k", None)
        val_max_tokens = self.config.opsd.get("val_max_tokens", None)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch.get("reward_model", {}).get("style") == "model"
            ):
                return {}

            reward_model_keys = {"data_source", "reward_model", "extra_info", "uid"} & test_batch.non_tensor_batch.keys()
            non_tensor_keys_to_pop = set(test_batch.non_tensor_batch.keys()) - reward_model_keys
            test_gen_batch = test_batch.pop(
                batch_keys=[],
                non_tensor_batch_keys=list(non_tensor_keys_to_pop),
            )
            test_gen_batch.non_tensor_batch.update(test_batch.non_tensor_batch)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": val_do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            if val_temperature is not None:
                test_gen_batch.meta_info["temperature"] = val_temperature
            if val_top_p is not None:
                test_gen_batch.meta_info["top_p"] = val_top_p
            if val_top_k is not None:
                test_gen_batch.meta_info["top_k"] = val_top_k
            if val_max_tokens is not None:
                test_gen_batch.meta_info["max_new_tokens"] = val_max_tokens

            size_divisor = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
                "num_workers", self.actor_rollout_wg.world_size
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                test_gen_batch_padded
            )
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
            ]
            sample_outputs.extend(output_texts)
            # Count non-pad response tokens
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            for ids in output_ids:
                sample_token_counts.append(int((ids != pad_id).sum().item()))

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            input_ids = test_batch.batch["prompts"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
            ]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = reward_result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(
                        values if isinstance(values, list) else [values]
                    )

            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        if not sample_scores:
            return {}

        # Sanity check: print first prompt/response pair
        py_logger.info(
            "Step %d: val sample[0] score=%.2f\n  PROMPT: %.300s\n  RESPONSE: %.500s",
            self.global_steps,
            sample_scores[0],
            sample_inputs[0],
            sample_outputs[0],
        )

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos_dict
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(
                    int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()
                )
                for metric_name, metric_val in metric2val.items():
                    if (
                        var_name == core_var
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and f"@{n_max}" in metric_name
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Response length metrics (overall + per data_source)
        metric_dict["val/avg_response_tokens"] = sum(sample_token_counts) / max(1, len(sample_token_counts))
        ds_token_counts: dict[str, list] = defaultdict(list)
        for ds, tc in zip(data_sources, sample_token_counts):
            ds_token_counts[ds].append(tc)
        for ds, counts in ds_token_counts.items():
            metric_dict[f"val/{ds}/avg_response_tokens"] = sum(counts) / len(counts)

        # Dump val generations to JSONL for inspection
        self._dump_val_generations(
            sample_inputs, sample_outputs, sample_scores,
            reward_extra_infos_dict,
            data_sources=data_sources,
            token_counts=sample_token_counts,
        )

        py_logger.info(
            "Step %d: _validate() complete -- %d samples, %d metrics",
            self.global_steps, len(sample_scores), len(metric_dict),
        )
        return metric_dict

    # ------------------------------------------------------------------
    # Worker initialization
    # ------------------------------------------------------------------

    def init_workers(self):
        """Initialize distributed workers using Ray backend.

        Creates the actor+rollout+ref worker group and the AgentLoopManager.
        Requires ActorRolloutRef role for the frozen teacher (ref model).
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRolloutRef],
            config=self.config.actor_rollout_ref,
            role=str(Role.ActorRolloutRef),
        )
        self.resource_pool_to_cls[resource_pool][str(Role.ActorRolloutRef)] = actor_rollout_cls

        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg[str(Role.ActorRolloutRef)]
        self.actor_rollout_wg.init_model()

        # Create AgentLoopManager for async generation
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "agent_loop_manager_class"
        )
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=None,
        )

        py_logger.info("OPSD workers initialized successfully")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
            if self.config.trainer.get("default_hdfs_dir")
            else None
        )

        local_mkdir_safe(actor_local_path)

        max_ckpt = self.config.trainer.get("max_actor_ckpt_to_keep", None)

        py_logger.info("Saving checkpoint to %s", actor_local_path)
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_ckpt,
        )
        py_logger.info("Checkpoint saved: step %d", self.global_steps)

    # ------------------------------------------------------------------
    # Teacher weight update
    # ------------------------------------------------------------------

    def _update_teacher_weights(self):
        """Hard-copy student weights to teacher (ref) model via workers.

        Sends a dummy DataProto to trigger update_teacher on each worker.
        """
        n_dp = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        dummy = DataProto.from_single_dict({"_dummy": torch.zeros(n_dp, 1)})
        self.actor_rollout_wg.update_teacher(dummy)
        py_logger.info(
            "Step %d: Teacher weights updated from student (teacher_update_freq=%d)",
            self.global_steps, self.teacher_update_freq,
        )

    # ------------------------------------------------------------------
    # Core Training Loop
    # ------------------------------------------------------------------

    def fit(self):
        """Main training loop: epoch-based OPSD.

        For each epoch, iterates over the SD prompt dataset:
          1. Swap raw_prompt to sft_prompt (question-only) for generation
          2. Generate student responses via AgentLoopManager
          3. Verify correctness for metrics (but do NOT filter)
          4. Train on ALL responses using JSD between teacher and student
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        progress_bar = tqdm(total=self.total_training_steps, desc="OPSD Training")

        self.global_steps += 1

        # Optional: run generation-based validation before training starts
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", False):
            py_logger.info("Running val_before_train...")
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)

        for epoch in range(self.config.trainer.total_epochs):
            epoch_metrics = {
                "epoch/total_generated": 0,
                "epoch/total_correct": 0,
                "epoch/total_trained": 0,
                "epoch/steps": 0,
            }

            for batch_dict in self.train_dataloader:
                step_t0 = time.time()
                metrics = {}

                # ---- Phase 1: Generate (question-only prompt) ----
                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                batch.meta_info["global_steps"] = self.global_steps
                # Cap training generation to sd_max_tokens so we don't generate
                # up to the full response_length (which may be larger for val).
                sd_max_tokens = self.config.opsd.get("sd_max_tokens", None)
                if sd_max_tokens is not None:
                    batch.meta_info["max_new_tokens"] = int(sd_max_tokens)

                # OPSD: save original teacher prompts as JSON strings before swapping
                # raw_prompt contains parsed sd_prompt (list of dicts); convert to JSON
                # so _opsd_update() can pass them to build_opsd_batch().
                sd_prompts_json = np.empty(len(batch.non_tensor_batch["raw_prompt"]), dtype=object)
                for i in range(len(batch.non_tensor_batch["raw_prompt"])):
                    sd_prompts_json[i] = json.dumps(batch.non_tensor_batch["raw_prompt"][i])
                batch.non_tensor_batch["sd_prompt"] = sd_prompts_json

                # Swap raw_prompt to sft_prompt (question-only) for student generation
                for i in range(len(batch.non_tensor_batch["raw_prompt"])):
                    batch.non_tensor_batch["raw_prompt"][i] = json.loads(
                        batch.non_tensor_batch["sft_prompt"][i]
                    )

                gen_t0 = time.time()
                gen_output = self.async_rollout_manager.generate_sequences(batch)
                gen_time = time.time() - gen_t0

                gen_output.meta_info.pop("timing", None)

                # ---- Phase 2: Verify (metrics only, no filtering) ----
                verify_t0 = time.time()
                responses, correct_mask, predictions = self._verify_responses(
                    gen_output, batch
                )
                verify_time = time.time() - verify_t0

                batch_size = len(responses)
                n_correct = sum(correct_mask)

                metrics["sd/batch_size"] = batch_size
                metrics["sd/n_correct"] = n_correct
                metrics["sd/accuracy"] = n_correct / max(1, batch_size)
                metrics["timing/generate_s"] = gen_time
                metrics["timing/verify_s"] = verify_time

                # Response length metrics (tokens)
                _prompt_len = gen_output.batch["prompts"].shape[1]
                _max_resp_tokens = gen_output.batch["responses"].shape[1]
                student_token_counts = []
                for i in range(batch_size):
                    resp_attn = gen_output.batch["attention_mask"][i, _prompt_len:]
                    student_token_counts.append(resp_attn.sum().item())
                metrics["sd/avg_student_token_count"] = sum(student_token_counts) / max(1, len(student_token_counts))
                metrics["sd/max_student_token_count"] = max(student_token_counts) if student_token_counts else 0
                metrics["sd/min_student_token_count"] = min(student_token_counts) if student_token_counts else 0

                # Truncation rate: response hit the generation cap.
                # Use sd_max_tokens (the actual generation limit) rather than
                # the tensor buffer size, which may be much larger.
                _gen_cap = int(
                    self.opsd_config.get("sd_max_tokens", 0)
                ) or _max_resp_tokens
                _n_truncated = sum(
                    1
                    for tc in student_token_counts
                    if tc >= _gen_cap
                )
                metrics["sd/truncation_rate"] = _n_truncated / max(1, batch_size)
                metrics["sd/n_truncated"] = _n_truncated

                # ---- KL gating mask ----
                truncated_mask = [
                    student_token_counts[i] >= _gen_cap
                    for i in range(batch_size)
                ]
                if self.kl_gating == "all":
                    kl_mask = [True] * batch_size
                elif self.kl_gating == "correct_only":
                    kl_mask = list(correct_mask)
                elif self.kl_gating == "incorrect_only":
                    kl_mask = [not c for c in correct_mask]
                else:  # correct_and_truncated
                    kl_mask = [
                        correct_mask[i] or truncated_mask[i]
                        for i in range(batch_size)
                    ]
                n_kl_samples = sum(kl_mask)
                metrics["sd/n_kl_samples"] = n_kl_samples

                if self.loss_type == "correctness_branched_kl":
                    metrics["sd/truncated_handling"] = self.truncated_handling

                teacher_solutions = list(batch.non_tensor_batch.get("teacher_solution", []))
                if teacher_solutions:
                    teacher_lens = [len(t) for t in teacher_solutions]
                    metrics["sd/avg_teacher_response_len"] = sum(teacher_lens) / max(1, len(teacher_lens))

                # ---- Phase 2.5: Generate Turn 2 (if mode requires it) ----
                epiphanies = None
                raw_turn2_outputs = None
                rescue_meta = None
                if self.teacher_ctx_mode in MODES_NEEDING_TURN2:
                    epiphany_t0 = time.time()
                    epiphanies, raw_turn2_outputs, rescue_meta = (
                        self._generate_epiphanies(
                            batch, responses, correct_mask,
                            kl_mask=kl_mask,
                            truncated_mask=truncated_mask,
                        )
                    )
                    epiphany_time = time.time() - epiphany_t0
                    metrics["timing/epiphany_s"] = epiphany_time

                    # Turn 2 token metrics (on active stitched outputs only)
                    epi_token_lens = [
                        len(self.tokenizer.encode(r))
                        for r in raw_turn2_outputs if r
                    ]
                    turn2_max_tokens = int(
                        self.opsd_config.get("turn2_max_tokens", 2048)
                    )
                    rescue_tokens = int(
                        self.opsd_config.get("turn2_rescue_tokens", 1024)
                    )
                    metrics["epiphany/avg_tokens"] = (
                        sum(epi_token_lens) / max(1, len(epi_token_lens))
                    )
                    metrics["epiphany/max_tokens"] = (
                        max(epi_token_lens) if epi_token_lens else 0
                    )

                    # Phase 2 clipping: did the final study notes get truncated?
                    rescued_set = rescue_meta.get("rescued_indices", set())
                    rescue_tok_lens = rescue_meta.get("rescue_token_lens", [])
                    n_clipped = 0
                    rescue_j = 0
                    for i, tl in enumerate(epi_token_lens):
                        if i in rescued_set:
                            # Rescued sample: clipped if rescue hit its budget
                            if rescue_tok_lens[rescue_j] >= rescue_tokens - 1:
                                n_clipped += 1
                            rescue_j += 1
                        else:
                            # Non-rescued: clipped if hit total budget
                            if tl >= turn2_max_tokens - 1:
                                n_clipped += 1
                    metrics["epiphany/clip_pct"] = n_clipped / max(
                        1, len(epi_token_lens)
                    )
                    metrics["epiphany/n_clipped"] = n_clipped
                    metrics["epiphany/n_rescued"] = rescue_meta.get(
                        "n_rescued", 0
                    )

                # ---- Phase 3: Train (OPSD on KL-gated samples) ----
                if n_kl_samples == 0:
                    py_logger.info(
                        "Step %d: No KL-active samples — skipping OPSD update",
                        self.global_steps,
                    )
                    opsd_metrics = {
                        "opsd/loss": 0.0, "opsd/skipped": 1.0, "opsd/n_samples": 0,
                        "opsd/n_kept_active_samples": 0,
                        "opsd/n_dropped_active_samples": 0,
                    }
                    if self.loss_type == "correctness_branched_kl":
                        opsd_metrics["opsd/n_reverse_kl_active_samples"] = 0
                        opsd_metrics["opsd/n_forward_kl_active_samples"] = 0
                    metrics.update(opsd_metrics)
                else:
                    train_t0 = time.time()
                    opsd_metrics = self._opsd_update(
                        batch, responses, epiphanies=epiphanies, kl_mask=kl_mask,
                        correct_mask=list(correct_mask),
                        truncated_mask=truncated_mask,
                    )
                    train_time = time.time() - train_t0
                    metrics.update(opsd_metrics)
                    metrics["timing/train_s"] = train_time

                    # ---- KL probe: ingest per-rank staging files and flush on schedule ----
                    if self.kl_probe_cfg.enabled and self.loss_type in ("reverse_kl", "forward_kl"):
                        # Build active correct/truncated masks in active-sample order,
                        # matching shard_batch_idx assignments made in _opsd_update.
                        if kl_mask is not None:
                            active_idx = [i for i, m in enumerate(kl_mask) if m]
                        else:
                            active_idx = list(range(batch_size))
                        active_correct = [bool(correct_mask[i]) for i in active_idx]
                        active_truncated = [bool(truncated_mask[i]) for i in active_idx]
                        probe_t0 = time.time()
                        self._ingest_kl_probe_staging(
                            self.global_steps, active_correct, active_truncated,
                        )
                        metrics["timing/kl_probe_ingest_s"] = time.time() - probe_t0

                        # Flush at step 1 and every plot_freq.
                        pf = max(1, self.kl_probe_cfg.plot_freq)
                        should_flush = (
                            self.global_steps == 1
                            or self.global_steps % pf == 0
                        )
                        if should_flush and self.kl_probe.has_data():
                            flush_t0 = time.time()
                            self.kl_probe.flush(
                                output_dir=self.kl_plots_dir,
                                step=int(self.global_steps),
                                meta={"teacher_ctx_mode": self.teacher_ctx_mode},
                            )
                            self.kl_probe.reset()
                            metrics["timing/kl_probe_flush_s"] = time.time() - flush_t0
                            py_logger.info(
                                "Step %d: KL probe flushed to %s",
                                self.global_steps, self.kl_plots_dir,
                            )

                    # ---- KL_ANALYZE_ONLY: exit after the first-step diagnostic flush ----
                    if self.kl_analyze_only and self.global_steps >= 1:
                        py_logger.info(
                            "KL_ANALYZE_ONLY: diagnostic flush complete, exiting after step %d.",
                            self.global_steps,
                        )
                        logger.log(data=metrics, step=self.global_steps)
                        return

                # ---- Teacher weight update (if configured) ----
                if self.teacher_update_freq > 0 and self.global_steps % self.teacher_update_freq == 0:
                    teacher_t0 = time.time()
                    self._update_teacher_weights()
                    metrics["timing/teacher_update_s"] = time.time() - teacher_t0
                    metrics["opsd/teacher_updated"] = 1.0

                epoch_metrics["epoch/total_generated"] += batch_size
                epoch_metrics["epoch/total_correct"] += n_correct
                epoch_metrics["epoch/total_trained"] += n_kl_samples
                epoch_metrics["epoch/steps"] += 1

                # ---- Log samples ----
                if self.global_steps % self.log_freq == 0:
                    self._log_rollout_samples(batch, responses, correct_mask, predictions)
                    self._log_epiphany_samples(
                        batch, responses, correct_mask, predictions,
                        epiphanies, raw_turn2_outputs, rescue_meta,
                        kl_mask=kl_mask, truncated_mask=truncated_mask,
                    )

                # ---- Phase 4: Validation ----
                is_last_step = self.global_steps >= self.total_training_steps
                is_val_step = self.test_freq > 0 and self.global_steps % self.test_freq == 0
                if self.val_batches and (is_val_step or is_last_step):
                    val_t0 = time.time()
                    val_loss = self._compute_val_loss()
                    val_time = time.time() - val_t0
                    if val_loss is not None:
                        metrics["val/loss"] = val_loss
                        metrics["timing/val_s"] = val_time
                        py_logger.info("Step %d: val/loss = %.4f (%.1fs)", self.global_steps, val_loss, val_time)

                if self.val_reward_fn is not None and (is_val_step or is_last_step):
                    val_gen_t0 = time.time()
                    val_gen_metrics = self._validate()
                    val_gen_time = time.time() - val_gen_t0
                    metrics.update(val_gen_metrics)
                    metrics["timing/val_gen_s"] = val_gen_time

                # ---- Logging ----
                step_time = time.time() - step_t0
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                    "timing/step_s": step_time,
                })

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)

                # ---- Checkpoint ----
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                self.global_steps += 1

                if is_last_step:
                    py_logger.info(
                        "Epoch %d summary -- Generated: %d, Correct: %d, Trained: %d",
                        epoch, epoch_metrics["epoch/total_generated"],
                        epoch_metrics["epoch/total_correct"],
                        epoch_metrics["epoch/total_trained"],
                    )
                    logger.log(data=epoch_metrics, step=self.global_steps - 1)
                    progress_bar.close()
                    return

            # End-of-epoch summary
            total_gen = epoch_metrics["epoch/total_generated"]
            total_cor = epoch_metrics["epoch/total_correct"]
            py_logger.info(
                "Epoch %d complete -- Generated: %d, Correct: %d (%.1f%%), Trained: %d (ALL)",
                epoch, total_gen, total_cor,
                100 * total_cor / max(1, total_gen),
                epoch_metrics["epoch/total_trained"],
            )
            logger.log(data=epoch_metrics, step=self.global_steps - 1)

            epoch_metrics = {
                "epoch/total_generated": 0,
                "epoch/total_correct": 0,
                "epoch/total_trained": 0,
                "epoch/steps": 0,
            }

        progress_bar.close()
        py_logger.info("OPSD training complete!")

    # ------------------------------------------------------------------
    # Phase 2: Verification (metrics only)
    # ------------------------------------------------------------------

    def _verify_responses(self, gen_output: DataProto, original_batch: DataProto):
        """Decode generated responses and verify correctness (for metrics only)."""
        prompt_length = gen_output.batch["prompts"].shape[1]
        batch_size = gen_output.batch["responses"].shape[0]

        responses = []
        for i in range(batch_size):
            response_ids = gen_output.batch["responses"][i]
            resp_attn_mask = gen_output.batch["attention_mask"][i, prompt_length:]
            valid_ids = response_ids[resp_attn_mask.bool()]
            text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            responses.append(text)

        ground_truths = list(original_batch.non_tensor_batch["ground_truth"])
        correct_mask, predictions = verify_batch(
            responses, ground_truths, check_structure=self.check_structure
        )

        return responses, correct_mask, predictions

    # ------------------------------------------------------------------
    # Phase 2.5: Epiphany generation
    # ------------------------------------------------------------------

    def _get_prompt_config(self):
        """Lazy-load prompt templates from config/prompts.json."""
        if not hasattr(self, "_prompt_config"):
            from pathlib import Path

            config_path = Path(__file__).resolve().parents[2] / "config" / "prompts.json"
            self._prompt_config = json.loads(config_path.read_text())
        return self._prompt_config

    @staticmethod
    def _strip_think_block(text: str) -> str:
        """Remove <think>...</think> block, return only the content after it."""
        idx = text.rfind("</think>")
        if idx >= 0:
            return text[idx + len("</think>"):].strip()
        return text.strip()

    def _generate_epiphanies(
        self,
        original_batch: DataProto,
        responses: list[str],
        correct_mask: list[bool],
        kl_mask: list[bool] | None = None,
        truncated_mask: list[bool] | None = None,
    ) -> tuple[list[str], list[str], dict]:
        """Generate Turn 2 epiphany memos via self-reflection.

        For each sample, builds a multi-turn prompt:
          [user: original question]
          [assistant: student Turn 1 response]
          [user: correctness feedback + memo instruction]

        Then generates the epiphany via sglang, strips <think> blocks,
        and returns cleaned epiphany text for teacher context.

        If the generation is truncated (hits the phase 1 budget), a rescue
        phase continues generation at the token level — either force-closing
        the <think> block first (if truncated mid-reasoning) or continuing
        the study notes (if truncated mid-notes).

        When ``kl_mask`` is provided, only generates epiphanies for active
        (True) samples. Inactive samples get empty strings. This saves
        Turn 2 generation compute when correctness-gated KL is enabled.

        Args:
            kl_mask: Per-sample mask — True = generate epiphany, False = skip.
            truncated_mask: Per-sample flag — True = Turn 1 hit token budget.
                Used to select the truncated Turn 2 template.

        Returns:
            (epiphanies, raw_turn2_outputs, rescue_meta) — stripped memos,
            raw generations, and rescue metadata dict. Inactive samples get
            empty strings in both lists.
        """
        batch_size = len(responses)
        empty_rescue_meta = {
            "n_rescued": 0, "rescued_indices": set(), "rescue_token_lens": [],
        }

        # Determine which samples to generate epiphanies for
        if kl_mask is not None:
            active_indices = [i for i in range(batch_size) if kl_mask[i]]
        else:
            active_indices = list(range(batch_size))
        n_active = len(active_indices)

        if n_active == 0:
            return [""] * batch_size, [""] * batch_size, empty_rescue_meta

        prompts_cfg = self._get_prompt_config()
        turn2_correct = prompts_cfg["reflection_from_gt_turn2_correct"]["template"]
        turn2_incorrect = prompts_cfg["reflection_from_gt_turn2_incorrect"]["template"]
        turn2_truncated = prompts_cfg.get(
            "reflection_from_gt_turn2_truncated", {},
        ).get("template", turn2_incorrect)

        sft_prompts = list(original_batch.non_tensor_batch["sft_prompt"])
        ground_truths = list(original_batch.non_tensor_batch["ground_truth"])
        questions = list(original_batch.non_tensor_batch["question"])

        turn2_max_tokens = int(self.opsd_config.get("turn2_max_tokens", 2048))
        rescue_tokens = int(self.opsd_config.get("turn2_rescue_tokens", 1024))
        phase1_budget = turn2_max_tokens - rescue_tokens

        turn2_raw_prompts = np.empty(n_active, dtype=object)

        for j, orig_i in enumerate(active_indices):
            # Parse original question from sft_prompt
            sft_msgs = json.loads(sft_prompts[orig_i])
            original_user_content = sft_msgs[0]["content"]

            # Look up expert demonstration if available
            format_kwargs = {"ground_truth": ground_truths[orig_i]}
            if self.expert_demos is not None:
                q = questions[orig_i]
                if q not in self.expert_demos:
                    raise KeyError(
                        f"Expert demonstration not found for question: {q[:100]}... "
                        f"All training questions must have expert demos."
                    )
                format_kwargs["expert_demonstration"] = self.expert_demos[q]

            # Build Turn 2 instruction: correct > truncated > incorrect
            if correct_mask[orig_i]:
                turn2_content = turn2_correct.format(**format_kwargs)
            elif truncated_mask is not None and truncated_mask[orig_i]:
                turn2_content = turn2_truncated.format(**format_kwargs)
            else:
                turn2_content = turn2_incorrect.format(**format_kwargs)

            # Multi-turn conversation
            turn2_raw_prompts[j] = [
                {"role": "user", "content": original_user_content},
                {"role": "assistant", "content": responses[orig_i]},
                {"role": "user", "content": turn2_content},
            ]

        # Build DataProto for Turn 2 generation (phase 1)
        turn2_batch = DataProto.from_single_dict({
            "dummy_tensor": torch.zeros(n_active, 1, dtype=torch.uint8),
        })
        turn2_batch.non_tensor_batch["raw_prompt"] = turn2_raw_prompts
        turn2_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(n_active)], dtype=object,
        )
        turn2_batch.meta_info["global_steps"] = self.global_steps
        turn2_batch.meta_info["max_new_tokens"] = phase1_budget

        # Pad batch to divisor for distributed generation
        size_divisor = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "num_workers", self.actor_rollout_wg.world_size
        )
        turn2_batch_padded, pad_size = pad_dataproto_to_divisor(turn2_batch, size_divisor)

        # Generate Turn 2 responses (phase 1)
        turn2_output = self.async_rollout_manager.generate_sequences(turn2_batch_padded)
        turn2_output = unpad_dataproto(turn2_output, pad_size=pad_size)

        # Decode Turn 2 responses (active subset)
        prompt_length = turn2_output.batch["prompts"].shape[1]
        active_raw_outputs = []
        active_epiphanies = []
        for j in range(n_active):
            response_ids = turn2_output.batch["responses"][j]
            resp_attn_mask = turn2_output.batch["attention_mask"][j, prompt_length:]
            valid_ids = response_ids[resp_attn_mask.bool()]
            raw_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            active_raw_outputs.append(raw_text)
            active_epiphanies.append(self._strip_think_block(raw_text))

        # ------------------------------------------------------------------
        # Rescue phase: continue truncated generations at the token level
        # ------------------------------------------------------------------
        think_close_ids = self.tokenizer.encode(
            "\n</think>\n", add_special_tokens=False,
        )

        rescue_indices = []  # list of (active_j, rescue_type)
        rescue_prompt_ids = []
        rescued_active_set = set()

        for j in range(n_active):
            token_len = len(self.tokenizer.encode(active_raw_outputs[j]))
            if token_len < phase1_budget - 1:
                continue  # completed naturally

            # Get original prompt + response token IDs from phase 1.
            # prompts are left-padded — strip padding via attention mask.
            prompt_mask = turn2_output.batch["attention_mask"][j, :prompt_length]
            orig_prompt = turn2_output.batch["prompts"][j][prompt_mask.bool()].tolist()
            resp_ids = turn2_output.batch["responses"][j]
            resp_mask = turn2_output.batch["attention_mask"][j, prompt_length:]
            valid_resp = resp_ids[resp_mask.bool()].tolist()

            if "</think>" not in active_raw_outputs[j]:
                # Truncated inside <think> — force-close, then generate notes
                rescue_prompt_ids.append(orig_prompt + valid_resp + think_close_ids)
                rescue_indices.append((j, "think_truncated"))
            else:
                # Truncated during study notes — continue as-is
                rescue_prompt_ids.append(orig_prompt + valid_resp)
                rescue_indices.append((j, "notes_truncated"))
            rescued_active_set.add(j)

        # Map rescued indices back to original batch indices
        rescued_set = {active_indices[j] for j in rescued_active_set}
        rescue_meta = {
            "n_rescued": len(rescue_indices),
            "rescued_indices": rescued_set,
        }

        if rescue_indices:
            n_rescue = len(rescue_indices)

            # Tell the agent loop worker to pad prompts to this length
            # (instead of the default config prompt_length which is too
            # small for rescue prompts that include the full phase 1 output).
            max_rescue_len = max(len(ids) for ids in rescue_prompt_ids)

            rescue_batch = DataProto.from_single_dict({
                "dummy_tensor": torch.zeros(n_rescue, 1, dtype=torch.uint8),
            })
            rescue_batch.non_tensor_batch["prompt_ids"] = np.array(
                rescue_prompt_ids, dtype=object,
            )
            rescue_batch.non_tensor_batch["_prompt_length"] = np.array(
                [max_rescue_len] * n_rescue, dtype=object,
            )
            rescue_batch.non_tensor_batch["raw_prompt"] = np.array(
                [turn2_raw_prompts[j] for j, _ in rescue_indices],
                dtype=object,
            )
            rescue_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(n_rescue)], dtype=object,
            )
            rescue_batch.meta_info["global_steps"] = self.global_steps
            rescue_batch.meta_info["max_new_tokens"] = rescue_tokens

            rescue_padded, rescue_pad = pad_dataproto_to_divisor(
                rescue_batch, size_divisor,
            )
            rescue_output = self.async_rollout_manager.generate_sequences(
                rescue_padded,
            )
            rescue_output = unpad_dataproto(rescue_output, pad_size=rescue_pad)

            # Decode rescue outputs and stitch into active outputs
            rescue_prompt_len = rescue_output.batch["prompts"].shape[1]
            rescue_token_lens = []
            for k, (j, rescue_type) in enumerate(rescue_indices):
                resp_ids = rescue_output.batch["responses"][k]
                resp_mask = rescue_output.batch["attention_mask"][
                    k, rescue_prompt_len:
                ]
                valid = resp_ids[resp_mask.bool()]
                rescue_text = self.tokenizer.decode(
                    valid, skip_special_tokens=True,
                )
                rescue_token_lens.append(len(valid))

                if rescue_type == "think_truncated":
                    active_raw_outputs[j] = (
                        active_raw_outputs[j] + "\n</think>\n" + rescue_text
                    )
                else:
                    active_raw_outputs[j] = (
                        active_raw_outputs[j] + rescue_text
                    )

                active_epiphanies[j] = self._strip_think_block(active_raw_outputs[j])

            rescue_meta["rescue_token_lens"] = rescue_token_lens
            logger.info(
                "Epiphany rescue: %d/%d active samples continued "
                "(phase1_budget=%d, rescue_budget=%d)",
                n_rescue, n_active, phase1_budget, rescue_tokens,
            )

        # Scatter active results back into full-size lists
        full_epiphanies = [""] * batch_size
        full_raw_outputs = [""] * batch_size
        for j, orig_i in enumerate(active_indices):
            full_epiphanies[orig_i] = active_epiphanies[j]
            full_raw_outputs[orig_i] = active_raw_outputs[j]

        return full_epiphanies, full_raw_outputs, rescue_meta

    # ------------------------------------------------------------------
    # Teacher-prompt builders (one per teacher_ctx_mode)
    # ------------------------------------------------------------------

    def _build_teacher_prompts_reflection_from_gt(
        self,
        student_prompts: list[str],
        epiphanies: list[str],
    ) -> list[str]:
        """[question] + [recall prefix] + [Turn 2 memo] + [recall suffix]."""
        cfg = self._get_prompt_config()["reflection_from_gt_teacher"]
        ctx_token_limit = self.opsd_config.get("turn2_teacher_ctx_tokens", None)
        teacher_prompts = []
        for i in range(len(student_prompts)):
            memo = epiphanies[i]
            if ctx_token_limit is not None:
                token_ids = self.tokenizer.encode(memo)
                if len(token_ids) > ctx_token_limit:
                    memo = self.tokenizer.decode(
                        token_ids[:ctx_token_limit],
                        skip_special_tokens=True,
                    )
            question_content = json.loads(student_prompts[i])[0]["content"]
            content = (
                question_content + "\n\n"
                + cfg["prefix"] + memo + cfg["suffix"]
            )
            teacher_prompts.append(json.dumps([{"role": "user", "content": content}]))
        return teacher_prompts

    def _build_teacher_prompts_gt_directly(
        self,
        student_prompts: list[str],
        ground_truths: list[str],
    ) -> list[str]:
        """[question] + [GT prefix] + [ground truth] + [GT suffix]. No Turn 2."""
        cfg = self._get_prompt_config()["gt_directly_teacher"]
        teacher_prompts = []
        for i in range(len(student_prompts)):
            question_content = json.loads(student_prompts[i])[0]["content"]
            content = (
                question_content + "\n\n"
                + cfg["prefix"] + str(ground_truths[i]) + cfg["suffix"]
            )
            teacher_prompts.append(json.dumps([{"role": "user", "content": content}]))
        return teacher_prompts

    def _build_teacher_prompts_conciseness_instruction(
        self,
        student_prompts: list[str],
    ) -> list[str]:
        """[conciseness prefix] + [question] + [conciseness suffix]. No Turn 2.

        Mirrors train_opsdc.sh: the self-teacher receives the length_prune_teacher
        conciseness instruction wrapping the bare question — no GT, no memo.
        """
        cfg = self._get_prompt_config()["length_prune_teacher"]
        teacher_prompts = []
        for sp in student_prompts:
            question_content = json.loads(sp)[0]["content"]
            content = cfg["prefix"] + question_content + cfg["suffix"]
            teacher_prompts.append(json.dumps([{"role": "user", "content": content}]))
        return teacher_prompts

    # ------------------------------------------------------------------
    # Reinjection helpers
    # ------------------------------------------------------------------

    _CONCISENESS_REINJECT_SNIPPET = (
        "\n\nActually, I should stay concise — avoid unnecessary elaboration, "
        "redundant steps, or restating the problem. Focus only on the key "
        "reasoning steps needed to reach the answer. Continuing from this.\n\n"
    )

    def _build_reinject_snippet_text(self, ctx_text: Optional[str]) -> Optional[str]:
        """Build the natural-wrapper snippet text for one sample.

        Returns None if the content variant can't be constructed for this sample
        (e.g. specific_context with empty ctx_text).
        """
        content = self.reinjection_content
        if content == "conciseness_instruction":
            return self._CONCISENESS_REINJECT_SNIPPET
        if not ctx_text:
            return None
        if content == "specific_context":
            # Full ctx text wrapped in the natural recall frame. Previously we
            # used only the first sentence; removed because split(".") was
            # fragile on abbreviations, decimals, and math equations, and the
            # "compressed" operating point wasn't carrying clear signal.
            return f"\n\nActually, I recall: {ctx_text}. Continuing from this.\n\n"
        raise ValueError(f"Unknown reinjection content: {content}")

    def _build_reinject_positions(
        self,
        response_text: str,
    ) -> tuple[set[int], int]:
        """Return (reinject_positions, think_close_pos) for one response.

        Positions snap back to the nearest sentence boundary; reinjection fires
        only while inside the ``<think>`` block (stops at ``</think>``). For
        truncated rollouts (no ``</think>``), applies for the entire response.
        """
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        if not response_ids:
            return set(), 0
        if response_ids[-1] != self.tokenizer.eos_token_id:
            response_ids = response_ids + [self.tokenizer.eos_token_id]

        think_close_pos = find_think_close(response_ids, self.tokenizer)

        positions: set[int] = set()
        interval = self.reinjection_interval
        next_target = interval
        while next_target < len(response_ids):
            in_thinking = (think_close_pos is None) or (next_target < think_close_pos)
            if not in_thinking:
                break
            snap_pos = snap_to_boundary(response_ids, next_target, self.tokenizer)
            # Only keep positions strictly inside the thinking block after snap.
            if (think_close_pos is None) or (snap_pos < think_close_pos):
                positions.add(snap_pos)
            next_target += interval
        return positions, len(response_ids)

    def _build_reinjection_arrays(
        self,
        responses: list[str],
        ctx_texts: list[Optional[str]],
    ) -> tuple[list[Optional[list[int]]], list[Optional[set[int]]], list[Optional[list[int]]]]:
        """For each sample, produce (snippet_ids, positions_set, positions_sorted)
        or (None, None, None).

        - ``positions_set``: consumed by the cumulative path (``build_opsd_batch``),
          which uses ``s_t in positions`` lookups during interleave.
        - ``positions_sorted``: consumed by the multi_pass path, which needs
          segment boundaries in ascending order.
        """
        if not self.reinjection_enabled:
            n = len(responses)
            return [None] * n, [None] * n, [None] * n

        snippet_list: list[Optional[list[int]]] = []
        positions_set_list: list[Optional[set[int]]] = []
        positions_sorted_list: list[Optional[list[int]]] = []
        for resp, ctx in zip(responses, ctx_texts):
            snippet_text = self._build_reinject_snippet_text(ctx)
            if not snippet_text:
                snippet_list.append(None)
                positions_set_list.append(None)
                positions_sorted_list.append(None)
                continue
            snippet_ids = self.tokenizer.encode(snippet_text, add_special_tokens=False)
            positions, _ = self._build_reinject_positions(resp)
            if not positions:
                snippet_list.append(None)
                positions_set_list.append(None)
                positions_sorted_list.append(None)
                continue
            snippet_list.append(snippet_ids)
            positions_set_list.append(positions)
            positions_sorted_list.append(sorted(positions))
        return snippet_list, positions_set_list, positions_sorted_list

    def _resolve_ctx_texts_for_reinjection(
        self,
        student_prompts: list[str],
        ground_truths: list[str],
        epiphanies: Optional[list[str]],
    ) -> list[Optional[str]]:
        """Return the per-sample ctx text to use when reinjection content =
        specific_context or full. Depends on ``teacher_ctx_mode``.

        - reflection_from_gt: the Turn 2 epiphany memo.
        - gt_directly: the ground-truth string.
        - conciseness_instruction / sd_prompt: no per-sample ctx → None.
          (content=conciseness_instruction doesn't use these anyway; content=
          specific_context/full becomes a no-op for those samples.)
        """
        n = len(student_prompts)
        if self.teacher_ctx_mode == "reflection_from_gt" and epiphanies is not None:
            return [e if e else None for e in epiphanies]
        if self.teacher_ctx_mode == "gt_directly":
            return [str(gt) if gt is not None else None for gt in ground_truths]
        return [None] * n

    # ------------------------------------------------------------------
    # Distance-weighting helpers
    # ------------------------------------------------------------------

    def _distance_weight_raw(self, relpos: torch.Tensor) -> torch.Tensor:
        """Schedule function applied to relpos ∈ [0, 1]. Monotone non-decreasing.

        Raw weights at relpos=0 → 1.0 and at relpos=1 → late_multiplier (default 2).
        """
        M = self.distance_weight_late_multiplier
        schedule = self.distance_weight_schedule
        if schedule == "linear":
            return 1.0 + (M - 1.0) * relpos
        if schedule == "exp":
            a = self.distance_weight_alpha
            denom = math.exp(a) - 1.0
            if denom <= 0:
                return torch.ones_like(relpos)
            return 1.0 + (torch.exp(a * relpos) - 1.0) / denom * (M - 1.0)
        if schedule == "step":
            return torch.where(
                relpos < 0.5,
                torch.ones_like(relpos),
                torch.full_like(relpos, M),
            )
        # "off"
        return torch.ones_like(relpos)

    @staticmethod
    def _build_use_reverse_kl_mask_padded(
        student_loss_mask: torch.Tensor,
        row_use_reverse_kl: list[bool],
    ) -> torch.Tensor:
        """Build the (B, max_L) padded per-token reverse-KL mask.

        True at response-token positions of rows that should use reverse KL
        (correct side per ``compute_use_reverse_kl_sample``). False everywhere
        else — non-response positions AND response positions of forward-KL rows.
        Worker flattens this to (N,) via ``_extract_response_values``.
        """
        B, max_L = student_loss_mask.shape
        if len(row_use_reverse_kl) != B:
            raise ValueError(
                f"row_use_reverse_kl length {len(row_use_reverse_kl)} != "
                f"batch dim {B}"
            )
        out = torch.zeros(B, max_L, dtype=torch.bool)
        bool_mask = student_loss_mask.bool()
        for i, use_rev in enumerate(row_use_reverse_kl):
            if use_rev:
                out[i] = bool_mask[i]
        return out

    def _build_kl_token_weights_padded(
        self,
        student_loss_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Build the (B, max_L) padded weight tensor for distance_weighted_kl.

        Per sample: weights at response-token positions follow the schedule,
        per-sample normalized so sum(weights_i) == L_i (mean-preserving).
        Non-response positions are zero.

        Returns None if distance weighting is off.
        """
        if self.distance_weight_schedule == "off":
            return None

        B, max_L = student_loss_mask.shape
        weights = torch.zeros(B, max_L, dtype=torch.float32)
        for i in range(B):
            mask_i = student_loss_mask[i]
            resp_idx = mask_i.nonzero(as_tuple=True)[0]
            L = int(resp_idx.numel())
            if L <= 0:
                continue
            denom = max(L - 1, 1)
            positions = torch.arange(L, dtype=torch.float32)
            relpos = positions / float(denom)
            raw = self._distance_weight_raw(relpos)
            if self.distance_weight_per_sample_norm and float(raw.sum()) > 0:
                raw = raw * (L / float(raw.sum()))
            weights[i, resp_idx] = raw
        return weights

    # ------------------------------------------------------------------
    # KL probe staging-file ingestion
    # ------------------------------------------------------------------

    def _ingest_kl_probe_staging(
        self,
        step: int,
        active_correct_mask: list[bool],
        active_truncated_mask: list[bool],
    ) -> None:
        """Read per-rank staging files for ``step``, feed to ``self.kl_probe``.

        The trainer emits ``shard_batch_idx`` on the dispatched batch so each
        rank's returned per-sample data can be aligned back to the original
        active-sample ordering. Correctness/truncation flags are then looked up
        by that index.
        """
        if not self.kl_probe_cfg.enabled:
            return

        pattern = os.path.join(self.kl_probe_staging_dir, f"step{step:06d}_rank*.pt")
        files = sorted(glob.glob(pattern))
        if not files:
            return

        # Gather all (shard_batch_idx, segment_idx, kl_array, L) tuples across ranks.
        # In cumulative mode: exactly one row per sample (segment_idx=0).
        # In multi-pass mode: K_i+1 rows per sample; need to re-merge by sample.
        all_rows: list[tuple[int, int, np.ndarray, int]] = []
        for path in files:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                py_logger.warning("KL probe: failed to load %s: %s", path, e)
                continue
            flat = payload.get("per_token_kl_flat")
            lengths = payload.get("per_sample_lengths")
            batch_idx = payload.get("shard_batch_indices")
            segment_idx_t = payload.get("segment_indices")
            if flat is None or lengths is None:
                continue
            flat_np = flat.detach().float().cpu().numpy()
            lens = [int(x) for x in lengths.tolist()]
            b_idx = batch_idx.tolist() if batch_idx is not None else [-1] * len(lens)
            s_idx = segment_idx_t.tolist() if segment_idx_t is not None else [0] * len(lens)
            # Split flat → per-row arrays by length.
            cursor = 0
            for i, L in enumerate(lens):
                if L <= 0:
                    continue
                arr = flat_np[cursor : cursor + L].copy()
                cursor += L
                idx = int(b_idx[i]) if i < len(b_idx) else -1
                seg = int(s_idx[i]) if i < len(s_idx) else 0
                all_rows.append((idx, seg, arr, L))
            try:
                os.remove(path)
            except OSError:
                pass

        if not all_rows:
            return

        # Group rows by sample_idx, sort each group by segment_idx, concat KL arrays.
        by_sample: dict[int, list[tuple[int, np.ndarray, int]]] = {}
        for sample_idx, segment_idx, arr, L in all_rows:
            if sample_idx < 0 or sample_idx >= len(active_correct_mask):
                # Padding-row sentinel (-1) or out-of-range index: skip.
                continue
            by_sample.setdefault(sample_idx, []).append((segment_idx, arr, L))

        if not by_sample:
            return

        # Emit one per-sample entry, sorted by sample_idx for deterministic order.
        kl_arrays: list[np.ndarray] = []
        lengths_list: list[int] = []
        correct: list[bool] = []
        truncated: list[bool] = []
        for sample_idx in sorted(by_sample.keys()):
            segments = sorted(by_sample[sample_idx], key=lambda t: t[0])
            seg_arrays = [arr for _, arr, _ in segments]
            seg_lens = [L for _, _, L in segments]
            per_sample_flat = (
                np.concatenate(seg_arrays, axis=0) if seg_arrays else np.zeros(0, np.float32)
            )
            total_len = int(sum(seg_lens))
            if total_len <= 0:
                continue
            kl_arrays.append(per_sample_flat)
            lengths_list.append(total_len)
            correct.append(bool(active_correct_mask[sample_idx]))
            truncated.append(bool(active_truncated_mask[sample_idx]))

        if not kl_arrays:
            return

        flat_cat = np.concatenate(kl_arrays, axis=0)
        self.kl_probe.add_step(
            torch.from_numpy(flat_cat),
            student_lengths=lengths_list,
            correct_mask=correct,
            truncated_mask=truncated,
        )

    # ------------------------------------------------------------------
    # Phase 3: OPSD Update
    # ------------------------------------------------------------------

    def _opsd_update(
        self,
        original_batch: DataProto,
        responses: list[str],
        epiphanies: list[str] = None,
        kl_mask: list[bool] | None = None,
        correct_mask: list[bool] | None = None,
        truncated_mask: list[bool] | None = None,
    ) -> dict:
        """Build OPSD batch and dispatch JSD training to workers.

        When ``kl_mask`` is provided, only KL-active samples are included
        in the training batch. Inactive samples are filtered out before
        batch construction, saving both teacher forward pass and gradient
        computation.

        Args:
            original_batch: Original batch with sd_prompt, sft_prompt, ground_truth.
            responses: Student-generated response strings.
            epiphanies: Optional list of stripped Turn 2 memos (only populated
                when teacher_ctx_mode == "reflection_from_gt").
            kl_mask: Per-sample mask — True = include in training, False = skip.
            correct_mask: Per-sample correctness flag (required for
                ``correctness_branched_kl``; otherwise may be ``None``).
            truncated_mask: Per-sample truncation flag (required for
                ``correctness_branched_kl``; otherwise may be ``None``).

        Returns:
            Dictionary of training metrics.
        """
        if self.loss_type == "correctness_branched_kl":
            assert kl_mask is not None, (
                "correctness_branched_kl requires kl_mask (even if all-True) "
                "to align with the active-sample filter path"
            )
            assert correct_mask is not None and truncated_mask is not None, (
                "correctness_branched_kl requires correct_mask and truncated_mask"
            )

        student_prompts = list(original_batch.non_tensor_batch["sft_prompt"])
        sd_prompts = list(original_batch.non_tensor_batch["sd_prompt"])
        ground_truths = list(original_batch.non_tensor_batch["ground_truth"])

        # Filter to KL-active samples only
        active: list[int] = list(range(len(responses)))
        if kl_mask is not None:
            active = [i for i, m in enumerate(kl_mask) if m]
            student_prompts = [student_prompts[i] for i in active]
            sd_prompts = [sd_prompts[i] for i in active]
            ground_truths = [ground_truths[i] for i in active]
            responses = [responses[i] for i in active]
            if epiphanies is not None:
                epiphanies = [epiphanies[i] for i in active]

        # Branched KL: compute per-sample direction (reverse vs forward) once,
        # over the active subset. Reused for (i) row broadcast, (ii) metrics,
        # (iii) early-return metric shape.
        use_reverse_kl_sample: list[bool] | None = None
        n_reverse_active = 0
        n_forward_active = 0
        if self.loss_type == "correctness_branched_kl":
            active_correct = [correct_mask[i] for i in active]
            active_truncated = [truncated_mask[i] for i in active]
            use_reverse_kl_sample = compute_use_reverse_kl_sample(
                active_correct, active_truncated, self.truncated_handling,
            )
            n_reverse_active = sum(use_reverse_kl_sample)
            n_forward_active = len(use_reverse_kl_sample) - n_reverse_active

        def _branched_direction_metrics() -> dict:
            """Direction counts + truncated_handling echo for branched mode.

            Called in every return path so invariant
            ``n_reverse_active + n_forward_active == len(active)`` holds
            regardless of early-return / normal completion.
            """
            if self.loss_type != "correctness_branched_kl":
                return {}
            return {
                "opsd/n_reverse_kl_active_samples": n_reverse_active,
                "opsd/n_forward_kl_active_samples": n_forward_active,
            }

        def _kept_metrics(n_kept: int) -> dict:
            """kept/dropped counts in active-sample units. Populated for all
            loss types so dashboards/tests see a stable schema when branched
            KL is on (and harmless when off)."""
            return {
                "opsd/n_kept_active_samples": n_kept,
                "opsd/n_dropped_active_samples": max(0, len(active) - n_kept),
            }

        builders = {
            "sd_prompt": lambda: sd_prompts,
            "reflection_from_gt": lambda: self._build_teacher_prompts_reflection_from_gt(
                student_prompts, epiphanies,
            ),
            "gt_directly": lambda: self._build_teacher_prompts_gt_directly(
                student_prompts, ground_truths,
            ),
            "conciseness_instruction": lambda: self._build_teacher_prompts_conciseness_instruction(
                student_prompts,
            ),
        }
        teacher_prompts = builders[self.teacher_ctx_mode]()

        # ---- Reinjection: compute per-sample snippet_ids + positions ----
        ctx_texts = self._resolve_ctx_texts_for_reinjection(
            student_prompts, ground_truths, epiphanies,
        )
        (
            teacher_reinject_snippets,
            teacher_reinject_positions_set,
            teacher_reinject_positions_sorted,
        ) = self._build_reinjection_arrays(responses, ctx_texts)
        n_reinjected = sum(1 for p in teacher_reinject_positions_sorted if p)

        # ---- Branch: cumulative (one row per sample) vs multi_pass (K+1 rows per sample) ----
        multipass = (
            self.reinjection_enabled
            and self.reinjection_mode == "multi_pass"
        )

        row_use_reverse_kl: list[bool] | None = None

        if multipass:
            built = build_opsd_batch_multipass(
                teacher_prompts=teacher_prompts,
                student_prompts=student_prompts,
                responses=responses,
                tokenizer=self.tokenizer,
                max_length=self.sft_max_length,
                reinject_snippets=teacher_reinject_snippets,
                reinject_positions=teacher_reinject_positions_sorted,
            )
            if built is None:
                return {
                    "opsd/loss": 0.0, "opsd/skipped": 1.0, "opsd/n_samples": 0,
                    **_kept_metrics(0),
                    **_branched_direction_metrics(),
                }
            opsd_batch, row_sample_idx, row_segment_idx = built
            n_rows = opsd_batch.batch["student_input_ids"].shape[0]
            # In multi-pass the "shard_batch_idx" identifies the ORIGINAL sample
            # this row belongs to; a segment_idx within tracks segment ordering
            # for kl_probe reconstruction.
            if self.kl_probe_cfg.enabled:
                opsd_batch.batch["shard_batch_idx"] = torch.tensor(
                    row_sample_idx, dtype=torch.long
                )
                opsd_batch.batch["segment_idx"] = torch.tensor(
                    row_segment_idx, dtype=torch.long
                )
            if use_reverse_kl_sample is not None:
                # Multi-pass expands one active sample into K+1 rows; replicate
                # the per-sample direction decision to each of those rows using
                # the sample index returned by the builder.
                row_use_reverse_kl = [
                    use_reverse_kl_sample[s] for s in row_sample_idx
                ]
            n_kept_active = len(set(row_sample_idx))
        else:
            opsd_batch = build_opsd_batch(
                teacher_prompts=teacher_prompts,
                student_prompts=student_prompts,
                responses=responses,
                tokenizer=self.tokenizer,
                max_length=self.sft_max_length,
                teacher_reinject_snippets=teacher_reinject_snippets,
                teacher_reinject_positions=teacher_reinject_positions_set,
            )
            if opsd_batch is None:
                return {
                    "opsd/loss": 0.0, "opsd/skipped": 1.0, "opsd/n_samples": 0,
                    **_kept_metrics(0),
                    **_branched_direction_metrics(),
                }
            n_rows = opsd_batch.batch["student_input_ids"].shape[0]
            if self.kl_probe_cfg.enabled:
                opsd_batch.batch["shard_batch_idx"] = torch.arange(
                    n_rows, dtype=torch.long
                )
                opsd_batch.batch["segment_idx"] = torch.zeros(n_rows, dtype=torch.long)
            if use_reverse_kl_sample is not None:
                # Cumulative path is 1:1 between active samples and rows unless
                # ``build_opsd_batch`` silently drops tokenization failures
                # (rare in practice; pre-existing issue affecting kl_probe too).
                # Fail loudly if that drop ever happens under branched KL so a
                # silent misalignment does not sneak into the loss.
                assert n_rows == len(use_reverse_kl_sample), (
                    f"build_opsd_batch dropped {len(use_reverse_kl_sample) - n_rows} "
                    "sample(s); correctness_branched_kl relies on 1:1 row-to-active "
                    "mapping in the cumulative path."
                )
                row_use_reverse_kl = list(use_reverse_kl_sample)
            n_kept_active = n_rows

        n_samples = n_rows  # legacy name; now may be row count in multi-pass

        # ---- Distance-weighting: build (B, max_L) padded weights tensor ----
        # In multi-pass, weights are computed per ROW using the row's loss_mask
        # (which is already gated to the current segment). Since each sample's
        # segments together cover the full response, per-sample normalization
        # is still valid at row level.
        kl_weights_padded = self._build_kl_token_weights_padded(
            opsd_batch.batch["student_loss_mask"]
        )
        if kl_weights_padded is not None:
            opsd_batch.batch["kl_token_weights_padded"] = kl_weights_padded

        # ---- Branched KL: per-token reverse-KL mask ----
        # Mirrors ``kl_token_weights_padded``: (B, max_L) bool with True at
        # response positions of rows that should use reverse KL (correct side)
        # and False elsewhere (non-response OR forward-KL rows). Worker flattens
        # this to (N,) via ``_extract_response_values`` and hands it to the
        # branched loss function.
        if row_use_reverse_kl is not None:
            use_reverse_kl_padded = self._build_use_reverse_kl_mask_padded(
                opsd_batch.batch["student_loss_mask"], row_use_reverse_kl,
            )
            opsd_batch.batch["use_reverse_kl_mask_padded"] = use_reverse_kl_padded

        # DP-pad: ensure batch is divisible by number of DP workers
        n_dp = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        if n_samples % n_dp != 0:
            pad_to = ((n_samples // n_dp) + 1) * n_dp
            pad_count = pad_to - n_samples
            padded_dict = {}
            for key in opsd_batch.batch.keys():
                tensor = opsd_batch.batch[key]
                last = tensor[-1:].expand(pad_count, *tensor.shape[1:]).clone()
                padded_dict[key] = torch.cat([tensor, last], dim=0)
            # Sentinels for padding rows: -1 for shard_batch_idx (won't match any
            # active sample) and segment_idx (marks pad row).
            if "shard_batch_idx" in padded_dict:
                sentinel = torch.full((pad_count,), -1, dtype=torch.long)
                padded_dict["shard_batch_idx"] = torch.cat(
                    [opsd_batch.batch["shard_batch_idx"], sentinel], dim=0,
                )
            if "segment_idx" in padded_dict:
                sentinel = torch.full((pad_count,), -1, dtype=torch.long)
                padded_dict["segment_idx"] = torch.cat(
                    [opsd_batch.batch["segment_idx"], sentinel], dim=0,
                )
            opsd_batch = DataProto.from_single_dict(padded_dict)
            py_logger.debug(
                "Step %d: Padded OPSD batch from %d to %d for %d DP workers",
                self.global_steps, n_samples, pad_to, n_dp,
            )

        # Pass config via meta_info so the worker can read it
        opsd_batch.meta_info["opsd_beta"] = self.beta
        opsd_batch.meta_info["opsd_loss_type"] = self.loss_type
        # KL probe staging — worker writes its shard's per-token KL to this dir.
        if self.kl_probe_cfg.enabled and self.loss_type in ("reverse_kl", "forward_kl"):
            opsd_batch.meta_info["collect_per_token_kl"] = True
            opsd_batch.meta_info["kl_probe_staging_dir"] = self.kl_probe_staging_dir
            opsd_batch.meta_info["global_steps"] = int(self.global_steps)

        py_logger.info(
            "Step %d: OPSD update with %d rows (beta=%.2f, loss=%s, reinject=%d, mode=%s, dw=%s)",
            self.global_steps, n_samples, self.beta, self.loss_type,
            n_reinjected,
            self.reinjection_mode if self.reinjection_enabled else "off",
            self.distance_weight_schedule,
        )

        # Dispatch to workers
        opsd_output = self.actor_rollout_wg.update_opsd(opsd_batch)

        # Reduce metrics across DP workers
        opsd_metrics = reduce_metrics(opsd_output.meta_info["metrics"])
        opsd_metrics["opsd/n_samples"] = n_samples
        opsd_metrics["opsd/n_reinjected"] = n_reinjected
        opsd_metrics["opsd/reinjection_mode"] = (
            self.reinjection_mode if self.reinjection_enabled else "off"
        )
        opsd_metrics["opsd/distance_weighting"] = self.distance_weight_schedule
        opsd_metrics.update(_kept_metrics(n_kept_active))
        opsd_metrics.update(_branched_direction_metrics())
        return opsd_metrics

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_rollout_samples(
        self,
        batch: DataProto,
        responses: list[str],
        correct_mask: list[bool],
        predictions: list[str],
    ):
        """Log student generation samples."""
        n_log = min(self.log_sample_count, len(responses))
        if n_log == 0:
            return

        questions = list(batch.non_tensor_batch["question"])
        ground_truths = list(batch.non_tensor_batch["ground_truth"])
        teacher_solutions = list(batch.non_tensor_batch.get("teacher_solution", []))
        sft_prompts = list(batch.non_tensor_batch.get("sft_prompt", []))
        sd_prompts = list(batch.non_tensor_batch.get("sd_prompt", []))

        n_correct = sum(correct_mask)
        avg_resp_len = sum(len(r) for r in responses) / max(1, len(responses))
        py_logger.info(
            "Step %d rollout: %d/%d correct, avg response len: %.0f chars",
            self.global_steps, n_correct, len(correct_mask), avg_resp_len,
        )
        for i in range(n_log):
            status = "correct" if correct_mask[i] else "wrong"
            q_preview = str(questions[i])[:80] if i < len(questions) else "?"
            resp_preview = responses[i][:120].replace("\n", "\\n") if responses[i] else "(empty)"
            py_logger.info("  [%s] Q: %s...", status, q_preview)
            py_logger.info("       Pred: %s | GT: %s", predictions[i], ground_truths[i])
            py_logger.info("       Resp: %s...", resp_preview)

        samples = []
        for i in range(n_log):
            # sft_prompt and sd_prompt are stored as JSON strings; parse for readability
            sft_prompt_parsed = None
            if i < len(sft_prompts):
                try:
                    sft_prompt_parsed = json.loads(sft_prompts[i])
                except (json.JSONDecodeError, TypeError):
                    sft_prompt_parsed = str(sft_prompts[i])

            sd_prompt_parsed = None
            if i < len(sd_prompts):
                try:
                    sd_prompt_parsed = json.loads(sd_prompts[i])
                except (json.JSONDecodeError, TypeError):
                    sd_prompt_parsed = str(sd_prompts[i])

            samples.append({
                "sample_idx": i,
                "question": str(questions[i]) if i < len(questions) else "",
                "student_generation_prompt": sft_prompt_parsed,
                "teacher_logit_prompt": sd_prompt_parsed,
                "student_response": responses[i],
                "teacher_response": str(teacher_solutions[i]) if i < len(teacher_solutions) else "",
                "ground_truth": str(ground_truths[i]) if i < len(ground_truths) else "",
                "prediction": str(predictions[i]),
                "is_correct": bool(correct_mask[i]),
            })

        step_file = os.path.join(self.rollout_log_dir, f"step_{self.global_steps:06d}.json")
        with open(step_file, "w", encoding="utf-8") as f:
            json.dump(
                {"step": self.global_steps, "n_samples": n_log, "samples": samples},
                f, indent=2, ensure_ascii=False,
            )

    def _log_epiphany_samples(
        self,
        batch: DataProto,
        responses: list[str],
        correct_mask: list[bool],
        predictions: list[str],
        epiphanies: list[str] | None,
        raw_turn2_outputs: list[str] | None,
        rescue_meta: dict | None = None,
        kl_mask: list[bool] | None = None,
        truncated_mask: list[bool] | None = None,
    ):
        """Log Turn 1 in/out, teacher input, and (when applicable) Turn 2 in/out.

        Runs for every teacher_ctx_mode so the probe can show the exact teacher
        context. Turn 2 fields are present only when the mode produced one.
        """
        n_log = min(self.log_sample_count, len(responses))
        if n_log == 0:
            return

        sft_prompts = list(batch.non_tensor_batch.get("sft_prompt", []))
        sd_prompts = list(batch.non_tensor_batch.get("sd_prompt", []))
        ground_truths = list(batch.non_tensor_batch["ground_truth"])
        questions = list(batch.non_tensor_batch["question"])
        prompts_cfg = self._get_prompt_config()

        has_turn2 = epiphanies is not None
        if has_turn2:
            reflection_cfg = prompts_cfg["reflection_from_gt_teacher"]
            turn2_correct = prompts_cfg["reflection_from_gt_turn2_correct"]["template"]
            turn2_incorrect = prompts_cfg["reflection_from_gt_turn2_incorrect"]["template"]
            turn2_truncated = prompts_cfg.get(
                "reflection_from_gt_turn2_truncated", {},
            ).get("template", turn2_incorrect)
        gt_directly_cfg = prompts_cfg.get("gt_directly_teacher", {})
        conciseness_cfg = prompts_cfg.get("length_prune_teacher", {})

        rescued_set = (
            rescue_meta.get("rescued_indices", set()) if rescue_meta else set()
        )

        samples = []
        for i in range(n_log):
            # Turn 1 input
            try:
                turn1_input = json.loads(sft_prompts[i])
            except (json.JSONDecodeError, TypeError, IndexError):
                turn1_input = str(sft_prompts[i]) if i < len(sft_prompts) else ""

            sft_msgs = json.loads(sft_prompts[i]) if i < len(sft_prompts) else []
            original_content = sft_msgs[0]["content"] if sft_msgs else ""
            gt = ground_truths[i] if i < len(ground_truths) else ""

            expert_demo = None
            if self.expert_demos is not None and i < len(questions):
                expert_demo = self.expert_demos.get(questions[i], "")

            # Turn 2 (only when the mode generates one)
            turn2_input = None
            if has_turn2:
                format_kwargs = {"ground_truth": gt}
                if expert_demo is not None:
                    format_kwargs["expert_demonstration"] = expert_demo
                if correct_mask[i]:
                    turn2_user_content = turn2_correct.format(**format_kwargs)
                elif truncated_mask is not None and truncated_mask[i]:
                    turn2_user_content = turn2_truncated.format(**format_kwargs)
                else:
                    turn2_user_content = turn2_incorrect.format(**format_kwargs)
                turn2_input = [
                    {"role": "user", "content": original_content},
                    {"role": "assistant", "content": responses[i]},
                    {"role": "user", "content": turn2_user_content},
                ]

            # Teacher input — exactly what _opsd_update will hand the teacher
            if self.teacher_ctx_mode == "reflection_from_gt":
                teacher_content = (
                    original_content + "\n\n"
                    + reflection_cfg["prefix"] + epiphanies[i] + reflection_cfg["suffix"]
                )
                teacher_input = [{"role": "user", "content": teacher_content}]
            elif self.teacher_ctx_mode == "gt_directly":
                teacher_content = (
                    original_content + "\n\n"
                    + gt_directly_cfg.get("prefix", "")
                    + str(gt)
                    + gt_directly_cfg.get("suffix", "")
                )
                teacher_input = [{"role": "user", "content": teacher_content}]
            elif self.teacher_ctx_mode == "conciseness_instruction":
                teacher_content = (
                    conciseness_cfg.get("prefix", "")
                    + original_content
                    + conciseness_cfg.get("suffix", "")
                )
                teacher_input = [{"role": "user", "content": teacher_content}]
            else:  # sd_prompt — passthrough from dataset
                try:
                    teacher_input = json.loads(sd_prompts[i]) if i < len(sd_prompts) else []
                except (json.JSONDecodeError, TypeError):
                    teacher_input = str(sd_prompts[i]) if i < len(sd_prompts) else ""

            sample = {
                "sample_idx": i,
                "teacher_ctx_mode": self.teacher_ctx_mode,
                "is_correct": bool(correct_mask[i]),
                "kl_active": bool(kl_mask[i]) if kl_mask is not None else True,
                "ground_truth": gt,
                "prediction": str(predictions[i]),
                "rescued": i in rescued_set,
                "turn1_input": turn1_input,
                "turn1_output": responses[i],
                "turn1_tokens": len(self.tokenizer.encode(responses[i])),
                "teacher_input": teacher_input,
            }
            if has_turn2:
                sample["turn2_input"] = turn2_input
                sample["turn2_output_raw"] = raw_turn2_outputs[i]
                sample["turn2_epiphany"] = epiphanies[i]
                sample["turn2_tokens"] = len(self.tokenizer.encode(raw_turn2_outputs[i]))
            if expert_demo is not None:
                sample["expert_demonstration"] = expert_demo
            samples.append(sample)

        step_file = os.path.join(self.epiphany_log_dir, f"step_{self.global_steps:06d}.json")
        with open(step_file, "w", encoding="utf-8") as f:
            json.dump(
                {"step": self.global_steps, "n_samples": n_log, "samples": samples},
                f, indent=2, ensure_ascii=False,
            )
