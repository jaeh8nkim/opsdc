"""
OPSD (On-Policy Self-Distillation) worker.

Extends SelfDistillWorker with a JSD-based training step (``update_opsd``).
The teacher model is the frozen ``ref_module_fsdp`` (initial policy weights)
and the student model is the trainable ``actor_module_fsdp``.

Key difference from ``update_sft``:
  - Two forward passes per micro-batch (teacher no-grad + student with-grad)
  - Loss is JSD divergence between teacher and student logit distributions
  - Trains on ALL rollouts, not just correct ones
"""

import logging
import math
import os
from typing import Optional

import psutil
import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.protocol import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)

from .sd_worker import SelfDistillWorker

logger = logging.getLogger(__name__)


class OPSDWorker(SelfDistillWorker):
    """Worker that adds OPSD JSD training to the HybridEngine actor+rollout+ref worker.

    Inherits all capabilities from SelfDistillWorker:
      - init_model, wake_up, sleep, chat_completion, generate
      - rollout_mode / trainer_mode for weight sync
      - update_sft, compute_val_loss
      - FSDP model + optimizer + scheduler management

    Adds:
      - update_opsd: JSD-based training using frozen ref model as teacher
    """

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_teacher(self, data: DataProto) -> DataProto:
        """Hard-copy student (actor) weights to teacher (ref) model.

        Copies all parameters from actor_module_fsdp to ref_module_fsdp
        using FSDP.summon_full_params for correctness.  Each rank
        materialises full params temporarily, copies, then scatters back.

        The input DataProto is a dummy (one row per DP worker) used only
        to satisfy the dispatch decorator.

        Returns:
            DataProto with meta_info confirming the update.
        """
        assert self._is_actor, "update_teacher requires actor role"
        assert hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None, (
            "update_teacher requires ref_module_fsdp (ref model)"
        )

        # Load both models to GPU if offloaded
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            load_fsdp_model_to_gpu(self.ref_module_fsdp)

        # Copy student -> teacher using summon_full_params
        with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False, recurse=True):
            with FSDP.summon_full_params(self.ref_module_fsdp, writeback=True, recurse=True):
                for p_student, p_teacher in zip(
                    self.actor_module_fsdp.parameters(),
                    self.ref_module_fsdp.parameters(),
                ):
                    p_teacher.data.copy_(p_student.data)

        logger.info("Teacher weights updated from student (hard copy)")

        # Offload back to CPU if needed
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        return DataProto(meta_info={"teacher_updated": True})

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_opsd(self, data: DataProto) -> DataProto:
        """Perform one OPSD training step using JSD between teacher and student.

        The input DataProto should contain:
          - teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_loss_mask
          - student_input_ids, student_attention_mask, student_position_ids, student_loss_mask

        The method:
          1. Loads actor FSDP model + optimizer + ref model to GPU (if offloaded)
          2. Splits data into micro-batches for gradient accumulation
          3. For each micro-batch:
             a. Forward teacher (ref_module_fsdp) with no_grad -> teacher logits
             b. Forward student (actor_module_fsdp) with grad -> student logits
             c. Compute JSD loss over response positions
          4. Clips gradients and steps optimizer + scheduler
          5. Offloads everything back to CPU (if offloaded)

        Returns:
            DataProto with meta_info containing training metrics.
        """
        assert self._is_actor, "update_opsd requires actor role"

        # --- Load actor model & optimizer to GPU if offloaded ---
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        # --- Load ref model to GPU if offloaded ---
        ref_offloaded = False
        if hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None:
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.ref_module_fsdp)
                ref_offloaded = True

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # will be moved per micro-batch
            beta = data.meta_info.get("opsd_beta", 0.5)
            loss_type = data.meta_info.get("opsd_loss_type", "jsd")
            metrics = self._opsd_training_step(data, beta=beta, loss_type=loss_type)

            # LR scheduler step
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["opsd/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() / (1024**3)
            )
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() / (1024**3)
            )
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        # --- Offload ref model back to CPU ---
        if ref_offloaded:
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
            log_gpu_memory_usage("After offload ref model during update_opsd", logger=logger)

        # --- Offload actor model & optimizer back to CPU ---
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_opsd", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_opsd", logger=logger)

        return output

    # ------------------------------------------------------------------
    # Core OPSD training logic
    # ------------------------------------------------------------------

    def _opsd_training_step(
        self, data: DataProto, beta: float = 0.5, loss_type: str = "jsd"
    ) -> dict:
        """Divergence-based training with gradient accumulation.

        For each micro-batch:
          1. Forward teacher (frozen ref model) on teacher_input_ids -> teacher logits
          2. Forward student (trainable actor) on student_input_ids -> student logits
          3. Extract response logits using respective loss_masks
          4. Compute divergence loss (JSD or reverse KL)
          5. Backward with gradient accumulation scaling

        Args:
            data: DataProto with teacher_* and student_* tensors.
            beta: JSD interpolation parameter (0.5 = symmetric JSD).
                  Unused for reverse_kl but passed through for API consistency.
            loss_type: "jsd" or "reverse_kl".

        Returns:
            Dictionary of training metrics.
        """
        self.actor_module_fsdp.train()
        self.ref_module_fsdp.eval()

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_liger = self.config.model.get("use_liger", False)

        micro_batch_size = self.config.actor.get(
            "ppo_micro_batch_size_per_gpu",
            self.config.actor.get("micro_batch_size_per_gpu", 2),
        )

        batch_size = data.batch["student_input_ids"].shape[0]
        if batch_size == 0:
            return {"opsd/loss": 0.0, "opsd/num_tokens": 0, "opsd/batch_size": 0}

        micro_batches = data.split(micro_batch_size)
        n_micro_batches = len(micro_batches)
        grad_accum = max(1, n_micro_batches)

        device = get_device_id()

        self.actor_optimizer.zero_grad()
        total_loss = 0.0
        total_tokens = 0
        total_student_entropy = 0.0
        total_teacher_entropy = 0.0
        total_entropy_tokens = 0

        # kl_probe collection (per-token KL + per-sample response lengths across
        # the shard). Written to a staging file at end of step when enabled.
        collect_per_token_kl = bool(data.meta_info.get("collect_per_token_kl", False)) \
            and loss_type == "reverse_kl"
        per_token_kl_chunks: list[torch.Tensor] = []
        per_sample_lengths_shard: list[int] = []

        # Per-sample original batch indices (for matching trainer-side correctness
        # / truncation masks after DP sharding). Each sample in the shard has an
        # integer batch index; we track them per micro-batch.
        per_sample_batch_indices_shard: list[int] = []

        # Distance-weighting token weights (padded) — optional input batch field.
        has_token_weights = "kl_token_weights_padded" in data.batch.keys()

        for mb_idx, micro_batch in enumerate(micro_batches):
            micro_batch = micro_batch.to(device)

            # Extract teacher tensors
            t_input_ids = micro_batch.batch["teacher_input_ids"]
            t_attention_mask = micro_batch.batch["teacher_attention_mask"]
            t_position_ids = micro_batch.batch["teacher_position_ids"]
            t_loss_mask = micro_batch.batch["teacher_loss_mask"]

            # Extract student tensors
            s_input_ids = micro_batch.batch["student_input_ids"]
            s_attention_mask = micro_batch.batch["student_attention_mask"]
            s_position_ids = micro_batch.batch["student_position_ids"]
            s_loss_mask = micro_batch.batch["student_loss_mask"]

            # Per-sample response lengths (number of loss_mask=1 positions per row).
            # Used for both kl_probe split-per-sample and optional token-weight extraction.
            mb_sample_lengths = s_loss_mask.sum(dim=-1).to(torch.long).cpu().tolist()
            if collect_per_token_kl:
                per_sample_lengths_shard.extend(int(L) for L in mb_sample_lengths)
                # Original batch indices if provided by trainer
                if "shard_batch_idx" in micro_batch.batch.keys():
                    per_sample_batch_indices_shard.extend(
                        micro_batch.batch["shard_batch_idx"].cpu().tolist()
                    )

            # Teacher forward (frozen, no grad)
            with torch.no_grad():
                if use_remove_padding:
                    teacher_logits = self._forward_logits_unpadded(
                        self.ref_module_fsdp, t_input_ids, t_attention_mask, t_position_ids,
                        t_loss_mask,
                    )
                else:
                    teacher_logits = self._forward_logits_padded(
                        self.ref_module_fsdp, t_input_ids, t_attention_mask, t_position_ids,
                        t_loss_mask,
                    )

            # Student forward (trainable, with grad)
            if use_remove_padding:
                student_logits = self._forward_logits_unpadded(
                    self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids,
                    s_loss_mask,
                )
            else:
                student_logits = self._forward_logits_padded(
                    self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids,
                    s_loss_mask,
                )

            # Align teacher and student logits by length
            # Both are (N_response_tokens, vocab_size) where N may differ.
            min_len = min(teacher_logits.shape[0], student_logits.shape[0])
            if min_len == 0:
                continue

            t_logits_aligned = teacher_logits[:min_len]
            s_logits_aligned = student_logits[:min_len]
            del teacher_logits  # free memory

            # Compute divergence loss
            loss_fn_map = {
                "jsd": (self._compute_jsd_loss, self._compute_jsd_loss_liger),
                "reverse_kl": (self._compute_reverse_kl_loss, self._compute_reverse_kl_loss_liger),
            }
            if loss_type not in loss_fn_map:
                raise ValueError(f"Unknown loss_type: {loss_type!r}. Expected one of {list(loss_fn_map)}")
            fn_standard, fn_liger = loss_fn_map[loss_type]

            # Optional per-token KL weights (for distance_weighted_kl). Provided by
            # the trainer as a (mb_B, max_L) padded tensor aligned with student
            # input_ids. Extracted to a (min_len,) flat tensor using the same
            # shift+mask logic as _forward_logits_*.
            mb_token_weights = None
            if has_token_weights:
                w_padded = micro_batch.batch["kl_token_weights_padded"]  # (mb_B, max_L)
                mb_token_weights = self._extract_response_values(
                    w_padded, s_loss_mask
                ).to(s_logits_aligned.device)
                # Safety: trim to min_len in case of any length divergence.
                mb_token_weights = mb_token_weights[:min_len]

            # Return the unreduced per-token KL when kl_probe collection is on.
            want_per_token = collect_per_token_kl

            if loss_type == "reverse_kl":
                fn = (
                    self._compute_reverse_kl_loss_liger if use_liger
                    else self._compute_reverse_kl_loss
                )
                result = fn(
                    t_logits_aligned,
                    s_logits_aligned,
                    beta=beta,
                    token_weights=mb_token_weights,
                    return_per_token=want_per_token,
                )
            else:
                fn = fn_liger if use_liger else fn_standard
                result = fn(t_logits_aligned, s_logits_aligned, beta=beta)

            if isinstance(result, tuple) and len(result) == 3:
                loss, n_tokens, per_token_kl = result
            else:
                loss, n_tokens = result
                per_token_kl = None

            if per_token_kl is not None:
                # Detach to CPU float32 and accumulate. Each micro-batch contributes
                # (min_len,) to the flat response-token sequence.
                per_token_kl_chunks.append(per_token_kl.detach().float().cpu())

            # Compute entropy for both teacher and student (no grad needed)
            with torch.no_grad():
                if use_liger:
                    s_ent, t_ent = self._compute_entropy_liger(
                        s_logits_aligned, t_logits_aligned
                    )
                else:
                    s_ent, t_ent = self._compute_entropy(
                        s_logits_aligned, t_logits_aligned
                    )
                total_student_entropy += s_ent * min_len
                total_teacher_entropy += t_ent * min_len
                total_entropy_tokens += min_len

            del t_logits_aligned

            # Scale for gradient accumulation
            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            total_loss += loss.detach().item()
            total_tokens += n_tokens

        # --- Gradient clipping and optimizer step ---
        grad_clip = self.config.actor.get("grad_clip", 1.0)
        if isinstance(self.actor_module_fsdp, FSDP):
            grad_norm = self.actor_module_fsdp.clip_grad_norm_(max_norm=grad_clip)
        else:
            from torch.distributed._composable.fsdp import FSDPModule
            if isinstance(self.actor_module_fsdp, FSDPModule):
                from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
                grad_norm = fsdp2_clip_grad_norm_(
                    self.actor_module_fsdp.parameters(), max_norm=grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_module_fsdp.parameters(), max_norm=grad_clip
                )

        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        if torch.isfinite(grad_norm):
            self.actor_optimizer.step()
        else:
            logger.warning("Non-finite grad_norm (%.4f), skipping optimizer step", grad_norm.item())
            self.actor_optimizer.zero_grad()

        avg_loss = total_loss / max(1, n_micro_batches)

        avg_student_entropy = total_student_entropy / max(1, total_entropy_tokens)
        avg_teacher_entropy = total_teacher_entropy / max(1, total_entropy_tokens)

        metrics = {
            "opsd/loss": avg_loss,
            "opsd/grad_norm": grad_norm.detach().item(),
            "opsd/num_tokens": int(total_tokens),
            "opsd/batch_size": batch_size,
            "opsd/beta": beta,
            "opsd/use_liger": int(use_liger),
            "opsd/student_entropy": avg_student_entropy,
            "opsd/teacher_entropy": avg_teacher_entropy,
            "opsd/entropy_diff": avg_student_entropy - avg_teacher_entropy,
        }

        # kl_probe staging: write per-rank file for trainer to aggregate.
        if collect_per_token_kl and per_token_kl_chunks:
            staging_dir = data.meta_info.get("kl_probe_staging_dir")
            step = int(data.meta_info.get("global_steps", 0))
            if staging_dir:
                try:
                    os.makedirs(staging_dir, exist_ok=True)
                    rank = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                        else 0
                    )
                    payload = {
                        "per_token_kl_flat": torch.cat(per_token_kl_chunks, dim=0),
                        "per_sample_lengths": torch.tensor(
                            per_sample_lengths_shard, dtype=torch.long
                        ),
                        "shard_batch_indices": torch.tensor(
                            per_sample_batch_indices_shard, dtype=torch.long
                        ) if per_sample_batch_indices_shard else None,
                        "step": step,
                        "rank": rank,
                    }
                    path = os.path.join(staging_dir, f"step{step:06d}_rank{rank:02d}.pt")
                    torch.save(payload, path)
                    metrics["opsd/kl_probe_staged"] = 1.0
                except Exception as e:
                    logger.exception("KL probe staging write failed: %s", e)
                    metrics["opsd/kl_probe_staged"] = 0.0
        return metrics

    # ------------------------------------------------------------------
    # Helper: extract response-aligned values from a (B, max_L) padded tensor
    # using the same shift+mask logic as _forward_logits_padded. Used to
    # derive a flat (N_response,) weight tensor from a (B, max_L) padded
    # weights tensor without re-running the forward.
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response_values(
        tensor_padded: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Flatten-and-select response-position values mirroring forward_logits.

        Args:
            tensor_padded: (B, max_L) values aligned with the student's input_ids
                positions. tensor_padded[b, i] is the value associated with
                student input_ids[b, i].
            loss_mask: (B, max_L) 1-at-response-token positions (same semantics
                as ``student_loss_mask``).

        Returns:
            (N,) flat tensor where N = number of response tokens in the
            micro-batch, ordered identically to ``_forward_logits_*`` outputs.
        """
        shift_v = tensor_padded[:, 1:]
        shift_mask = loss_mask[:, 1:]
        B, S = shift_v.shape
        flat_v = shift_v.reshape(B * S)
        flat_mask = shift_mask.reshape(B * S)
        response_indices = flat_mask.nonzero(as_tuple=True)[0]
        return flat_v[response_indices]

    # ------------------------------------------------------------------
    # Forward logits: Padded path
    # ------------------------------------------------------------------

    def _forward_logits_padded(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning response-position logits (padded path).

        Args:
            model: FSDP model (ref_module_fsdp or actor_module_fsdp).
            input_ids: (B, max_len)
            attention_mask: (B, max_len)
            position_ids: (B, max_len)
            loss_mask: (B, max_len) — 1 for response tokens

        Returns:
            Flattened response logits: (N_response_tokens, vocab_size)
            where N_response_tokens = sum of loss_mask shifted positions.
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits  # (B, max_len, V)
            del outputs

        # Shift for next-token prediction: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :]  # (B, max_len-1, V)
        shift_loss_mask = loss_mask[:, 1:]  # (B, max_len-1)
        del logits

        # Flatten and select response positions
        B, S, V = shift_logits.shape
        flat_logits = shift_logits.reshape(B * S, V)
        flat_mask = shift_loss_mask.reshape(B * S)
        del shift_logits

        response_indices = flat_mask.nonzero(as_tuple=True)[0]
        response_logits = flat_logits[response_indices]  # (N, V)

        return response_logits

    # ------------------------------------------------------------------
    # Forward logits: Unpadded path
    # ------------------------------------------------------------------

    def _forward_logits_unpadded(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning response-position logits (unpadded path).

        Same as _forward_logits_padded but uses flash_attn_varlen for memory
        efficiency on long sequences.

        Returns:
            Flattened response logits: (N_response_tokens, vocab_size)
        """
        # Unpad inputs
        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)  # (1, total_nnz)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,  # triggers flash_attn_varlen
                position_ids=position_ids_rmpad,
                use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)  # (total_nnz, V)
            del outputs

        # Unpad loss_mask using the same indices, then shift
        loss_mask_flat = loss_mask.reshape(-1)
        loss_mask_rmpad = loss_mask_flat[indices]  # (total_nnz,)

        # Shifted loss mask: position i predicts token i+1
        shifted_loss_mask = torch.zeros_like(loss_mask_rmpad)
        shifted_loss_mask[:-1] = loss_mask_rmpad[1:]

        # Select response-position logits
        response_indices = shifted_loss_mask.nonzero(as_tuple=True)[0]
        response_logits = logits_rmpad[response_indices]  # (N, V)

        return response_logits

    # ------------------------------------------------------------------
    # JSD loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_jsd_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 512,
    ) -> tuple[torch.Tensor, int]:
        """Compute Jensen-Shannon divergence between teacher and student logits.

        JSD_beta(p_T || p_S) = beta * KL(p_T || m) + (1-beta) * KL(p_S || m)
        where m = beta * p_T + (1-beta) * p_S

        Processes tokens in chunks to avoid OOM from materializing full
        (N, V) float32 probability tensors.  Each chunk holds at most
        ~6 * (chunk_size, V) float32 intermediates; with chunk_size=512
        and V=152K this is ~1.8 GB instead of tens of GB for the full N.

        Args:
            teacher_logits: (N, V) — logits from frozen teacher (no grad).
            student_logits: (N, V) — logits from trainable student (with grad).
            beta: Interpolation parameter (0.5 = symmetric JSD).
            chunk_size: Number of tokens to process at a time.

        Returns:
            (loss, n_tokens) — scalar loss and number of tokens.
        """
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

        # Accumulate JSD sum over chunks of tokens to bound peak memory.
        # Slicing into student_logits preserves autograd (view op).
        jsd_sum = torch.tensor(0.0, device=student_logits.device)

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)

            # Convert chunks to float32 for numerical stability
            t_chunk = teacher_logits[start:end].float()
            s_chunk = student_logits[start:end].float()

            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            del t_chunk, s_chunk

            t_probs = t_log_probs.exp()
            s_probs = s_log_probs.exp()

            # Mixture: m = beta * p_T + (1-beta) * p_S
            m_log_probs = (beta * t_probs + (1.0 - beta) * s_probs).clamp(min=1e-8).log()

            # KL(p_T || m) per token
            kl_t = (t_probs * (t_log_probs - m_log_probs)).sum(dim=-1)
            del t_probs, t_log_probs

            # KL(p_S || m) per token
            kl_s = (s_probs * (s_log_probs - m_log_probs)).sum(dim=-1)
            del s_probs, s_log_probs, m_log_probs

            jsd_chunk = beta * kl_t + (1.0 - beta) * kl_s
            jsd_sum = jsd_sum + jsd_chunk.sum()
            del kl_t, kl_s, jsd_chunk

        loss = jsd_sum / n_tokens
        return loss, n_tokens

    @staticmethod
    def _compute_entropy(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        chunk_size: int = 512,
    ) -> tuple[float, float]:
        """Compute average per-token entropy for student and teacher distributions.

        H(p) = -sum(p * log(p))

        Uses chunked processing to avoid OOM, same as _compute_jsd_loss.

        Args:
            student_logits: (N, V) — student logits.
            teacher_logits: (N, V) — teacher logits.
            chunk_size: Number of tokens to process at a time.

        Returns:
            (student_entropy, teacher_entropy) — average per-token entropy (nats).
        """
        n_tokens = student_logits.shape[0]
        if n_tokens == 0:
            return 0.0, 0.0

        s_entropy_sum = 0.0
        t_entropy_sum = 0.0

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)

            s_log_probs = F.log_softmax(student_logits[start:end].float(), dim=-1)
            s_probs = s_log_probs.exp()
            s_entropy_sum += -(s_probs * s_log_probs).sum(dim=-1).sum().item()
            del s_probs, s_log_probs

            t_log_probs = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            t_probs = t_log_probs.exp()
            t_entropy_sum += -(t_probs * t_log_probs).sum(dim=-1).sum().item()
            del t_probs, t_log_probs

        return s_entropy_sum / n_tokens, t_entropy_sum / n_tokens

    # ------------------------------------------------------------------
    # Liger-style JSD loss: logsumexp mixture + progressive teacher freeing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_jsd_loss_liger(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 256,
    ) -> tuple[torch.Tensor, int]:
        """Memory-efficient JSD using logsumexp for the mixture distribution.

        Improvements over ``_compute_jsd_loss``:
          1. **logsumexp mixture**: Computes log(beta*p_T + (1-beta)*p_S) via
             logsumexp in log-space, avoiding explicit probability tensors.
             This reduces peak float32 intermediates from ~6 (C,V) to ~4 (C,V).
          2. **Progressive teacher freeing**: Clones teacher logits into chunks
             and frees the original, so only one teacher chunk is alive at a time.
          3. **Smaller default chunk_size** (256 vs 512) for lower per-chunk peak.

        Args:
            teacher_logits: (N, V) — logits from frozen teacher (no grad).
            student_logits: (N, V) — logits from trainable student (with grad).
            beta: JSD interpolation parameter (0.5 = symmetric JSD).
            chunk_size: Number of tokens per chunk.

        Returns:
            (loss, n_tokens) — scalar loss and number of response tokens.
        """
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

        # Pre-compute log(beta) and log(1-beta) for logsumexp mixture
        log_beta = math.log(beta) if beta > 0 else float("-inf")
        log_1m_beta = math.log(1.0 - beta) if beta < 1 else float("-inf")

        # Clone teacher chunks and free the original contiguous tensor.
        # Teacher has no grad, so cloning is cheap and frees ~N*V*2 bytes.
        teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
        del teacher_logits

        jsd_sum = torch.tensor(0.0, device=student_logits.device)

        for i, t_chunk in enumerate(teacher_chunks):
            start = i * chunk_size
            end = start + t_chunk.shape[0]

            # Convert to float32 for numerical stability
            t_lp = F.log_softmax(t_chunk.float(), dim=-1)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
            del t_chunk
            teacher_chunks[i] = None  # allow GC

            # logsumexp mixture: log_m = log(beta * exp(t_lp) + (1-beta) * exp(s_lp))
            #                         = logsumexp([t_lp + log_beta, s_lp + log_1m_beta])
            log_m = torch.logsumexp(
                torch.stack([t_lp + log_beta, s_lp + log_1m_beta], dim=0),
                dim=0,
            )

            # KL(p_T || m) and KL(p_S || m) via F.kl_div with log_target=True
            kl_t = F.kl_div(log_m, t_lp, reduction="none", log_target=True).sum(dim=-1)
            del t_lp
            kl_s = F.kl_div(log_m, s_lp, reduction="none", log_target=True).sum(dim=-1)
            del s_lp, log_m

            jsd_chunk = beta * kl_t + (1.0 - beta) * kl_s
            jsd_sum = jsd_sum + jsd_chunk.sum()
            del kl_t, kl_s, jsd_chunk

        loss = jsd_sum / n_tokens
        return loss, n_tokens

    # ------------------------------------------------------------------
    # Reverse KL loss: KL(student || teacher)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reverse_kl_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 512,
        token_weights: Optional[torch.Tensor] = None,
        return_per_token: bool = False,
    ):
        """Compute reverse KL divergence: KL(student || teacher).

        KL(p_S || p_T) = sum_x p_S(x) * [log p_S(x) - log p_T(x)]

        This is "mode-seeking": the student concentrates probability mass on
        the teacher's high-probability tokens, which encourages the student
        to adopt the teacher's concise reasoning style without spreading
        probability over tokens the teacher assigns low weight to.

        Args:
            teacher_logits: (N, V) — logits from frozen teacher (no grad).
            student_logits: (N, V) — logits from trainable student (with grad).
            beta: Unused, kept for API compatibility with JSD.
            chunk_size: Number of tokens to process at a time.
            token_weights: Optional (N,) float tensor. Per-token KL is
                multiplied by these weights before reduction (used by
                distance_weighted_kl). The denominator remains ``n_tokens``,
                so the loss is mean-preserving when the trainer pre-normalizes
                weights per-sample to sum to L_i.
            return_per_token: If True, additionally return the unreduced
                (N,) per-token KL tensor (detached, for kl_probe logging).

        Returns:
            (loss, n_tokens) or (loss, n_tokens, per_token_kl) if
            ``return_per_token``.
        """
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            zero = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
            if return_per_token:
                return zero, 0, torch.zeros(0, device=student_logits.device)
            return zero, 0

        kl_sum = torch.tensor(0.0, device=student_logits.device)
        per_token_parts = [] if return_per_token else None

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)

            t_log_probs = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            s_log_probs = F.log_softmax(student_logits[start:end].float(), dim=-1)

            s_probs = s_log_probs.exp()
            kl_chunk = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
            del t_log_probs, s_log_probs, s_probs

            if return_per_token:
                per_token_parts.append(kl_chunk.detach())

            if token_weights is not None:
                kl_chunk = kl_chunk * token_weights[start:end]

            kl_sum = kl_sum + kl_chunk.sum()
            del kl_chunk

        loss = kl_sum / n_tokens
        if return_per_token:
            per_token_kl = torch.cat(per_token_parts, dim=0)
            return loss, n_tokens, per_token_kl
        return loss, n_tokens

    @staticmethod
    def _compute_reverse_kl_loss_liger(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 256,
        token_weights: Optional[torch.Tensor] = None,
        return_per_token: bool = False,
    ):
        """Memory-efficient reverse KL: KL(student || teacher).

        Same as ``_compute_reverse_kl_loss`` but with progressive teacher
        freeing (clone teacher chunks, delete original) for lower peak memory.

        See :meth:`_compute_reverse_kl_loss` for full arg/return docstring —
        the ``token_weights`` and ``return_per_token`` arguments behave
        identically here.
        """
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            zero = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
            if return_per_token:
                return zero, 0, torch.zeros(0, device=student_logits.device)
            return zero, 0

        teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
        del teacher_logits

        kl_sum = torch.tensor(0.0, device=student_logits.device)
        per_token_parts = [] if return_per_token else None

        for i, t_chunk in enumerate(teacher_chunks):
            start = i * chunk_size
            end = start + t_chunk.shape[0]

            t_lp = F.log_softmax(t_chunk.float(), dim=-1)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
            del t_chunk
            teacher_chunks[i] = None

            # KL(p_S || p_T) = sum p_S * (log p_S - log p_T)
            kl_chunk = F.kl_div(t_lp, s_lp, reduction="none", log_target=True).sum(dim=-1)
            del t_lp, s_lp

            if return_per_token:
                per_token_parts.append(kl_chunk.detach())

            if token_weights is not None:
                kl_chunk = kl_chunk * token_weights[start:end]

            kl_sum = kl_sum + kl_chunk.sum()
            del kl_chunk

        loss = kl_sum / n_tokens
        if return_per_token:
            per_token_kl = torch.cat(per_token_parts, dim=0)
            return loss, n_tokens, per_token_kl
        return loss, n_tokens

    # ------------------------------------------------------------------
    # Liger-style entropy: logsumexp-based, smaller chunks
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_entropy_liger(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        chunk_size: int = 256,
    ) -> tuple[float, float]:
        """Memory-efficient per-token entropy using smaller chunks.

        H(p) = -sum(p * log(p)) = log(Z) - (1/Z) * sum(x_i * exp(x_i)) / Z
        Simplified: H(p) = log_softmax-based computation.

        Args:
            student_logits: (N, V) — student logits.
            teacher_logits: (N, V) — teacher logits.
            chunk_size: Number of tokens per chunk.

        Returns:
            (student_entropy, teacher_entropy) — average per-token entropy (nats).
        """
        n_tokens = student_logits.shape[0]
        if n_tokens == 0:
            return 0.0, 0.0

        s_entropy_sum = 0.0
        t_entropy_sum = 0.0

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)

            # Student entropy: H = -sum(p * log_p)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
            s_entropy_sum += -(s_lp.exp() * s_lp).sum(dim=-1).sum().item()
            del s_lp

            # Teacher entropy
            t_lp = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            t_entropy_sum += -(t_lp.exp() * t_lp).sum(dim=-1).sum().item()
            del t_lp

        return s_entropy_sum / n_tokens, t_entropy_sum / n_tokens
