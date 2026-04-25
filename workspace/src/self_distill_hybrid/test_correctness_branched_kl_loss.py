"""Unit tests for the correctness-branched KL loss (standard + liger variants).

Run with:
    python -m pytest test_correctness_branched_kl_loss.py -v
"""

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Namespace-extracted copies of the branched KL loss functions. Mirrors the
# pattern in test_opsd_jsd.py so tests don't need VERL / torch-FSDP imports.
# ---------------------------------------------------------------------------

class _CB:
    """Static copies of the branched KL loss functions from OPSDWorker."""

    @staticmethod
    def compute(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        use_reverse_kl_mask: torch.Tensor,
        chunk_size: int = 512,
        token_weights=None,
        return_per_token: bool = False,
    ):
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            zero = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
            if return_per_token:
                return zero, 0, torch.zeros(0, device=student_logits.device)
            return zero, 0

        assert use_reverse_kl_mask.shape == (n_tokens,)

        kl_sum = torch.tensor(0.0, device=student_logits.device)
        per_token_parts = [] if return_per_token else None

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            t_lp = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)

            s_p = s_lp.exp()
            rev_kl = (s_p * (s_lp - t_lp)).sum(dim=-1)
            del s_p

            t_p = t_lp.exp()
            fwd_kl = (t_p * (t_lp - s_lp)).sum(dim=-1)
            del t_p

            kl_chunk = torch.where(use_reverse_kl_mask[start:end], rev_kl, fwd_kl)
            del t_lp, s_lp, rev_kl, fwd_kl

            if return_per_token:
                per_token_parts.append(kl_chunk.detach())
            if token_weights is not None:
                kl_chunk = kl_chunk * token_weights[start:end]
            kl_sum = kl_sum + kl_chunk.sum()
            del kl_chunk

        loss = kl_sum / n_tokens
        if return_per_token:
            return loss, n_tokens, torch.cat(per_token_parts, dim=0)
        return loss, n_tokens

    @staticmethod
    def compute_liger(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        use_reverse_kl_mask: torch.Tensor,
        chunk_size: int = 256,
        token_weights=None,
        return_per_token: bool = False,
    ):
        n_tokens = teacher_logits.shape[0]
        if n_tokens == 0:
            zero = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
            if return_per_token:
                return zero, 0, torch.zeros(0, device=student_logits.device)
            return zero, 0

        assert use_reverse_kl_mask.shape == (n_tokens,)

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

            s_p = s_lp.exp()
            rev_kl = (s_p * (s_lp - t_lp)).sum(dim=-1)
            del s_p

            t_p = t_lp.exp()
            fwd_kl = (t_p * (t_lp - s_lp)).sum(dim=-1)
            del t_p

            kl_chunk = torch.where(use_reverse_kl_mask[start:end], rev_kl, fwd_kl)
            del t_lp, s_lp, rev_kl, fwd_kl

            if return_per_token:
                per_token_parts.append(kl_chunk.detach())
            if token_weights is not None:
                kl_chunk = kl_chunk * token_weights[start:end]
            kl_sum = kl_sum + kl_chunk.sum()
            del kl_chunk

        loss = kl_sum / n_tokens
        if return_per_token:
            return loss, n_tokens, torch.cat(per_token_parts, dim=0)
        return loss, n_tokens


# Reference implementations used as ground truth (no chunking).
def _ref_reverse_kl(teacher_logits, student_logits):
    t_lp = F.log_softmax(teacher_logits.float(), dim=-1)
    s_lp = F.log_softmax(student_logits.float(), dim=-1)
    s_p = s_lp.exp()
    return (s_p * (s_lp - t_lp)).sum(dim=-1).mean()


def _ref_forward_kl(teacher_logits, student_logits):
    t_lp = F.log_softmax(teacher_logits.float(), dim=-1)
    s_lp = F.log_softmax(student_logits.float(), dim=-1)
    t_p = t_lp.exp()
    return (t_p * (t_lp - s_lp)).sum(dim=-1).mean()


def _make_logits(n_tokens, vocab_size, seed=42, requires_grad=False):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(n_tokens, vocab_size, generator=gen, dtype=torch.bfloat16)
    if requires_grad:
        logits = logits.float().requires_grad_(True)
    return logits


# ---------------------------------------------------------------------------
# Parity with reference KL implementations
# ---------------------------------------------------------------------------


class TestBranchedParity:
    """At the extremes, the branched loss should reduce to plain reverse / forward KL."""

    @pytest.mark.parametrize("n_tokens,vocab_size", [
        (1, 10),
        (16, 128),
        (300, 2000),
    ])
    def test_all_true_matches_reverse_kl(self, n_tokens, vocab_size):
        t = _make_logits(n_tokens, vocab_size, seed=1)
        s = _make_logits(n_tokens, vocab_size, seed=2)
        mask = torch.ones(n_tokens, dtype=torch.bool)

        loss_branched, _ = _CB.compute(t, s, mask)
        loss_ref = _ref_reverse_kl(t, s)

        torch.testing.assert_close(loss_branched, loss_ref, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("n_tokens,vocab_size", [
        (1, 10),
        (16, 128),
        (300, 2000),
    ])
    def test_all_false_matches_forward_kl(self, n_tokens, vocab_size):
        t = _make_logits(n_tokens, vocab_size, seed=3)
        s = _make_logits(n_tokens, vocab_size, seed=4)
        mask = torch.zeros(n_tokens, dtype=torch.bool)

        loss_branched, _ = _CB.compute(t, s, mask)
        loss_ref = _ref_forward_kl(t, s)

        torch.testing.assert_close(loss_branched, loss_ref, atol=1e-5, rtol=1e-4)

    def test_mixed_mask_matches_manual_mix(self):
        n, v = 32, 128
        t = _make_logits(n, v, seed=5)
        s = _make_logits(n, v, seed=6)
        mask = torch.tensor([i % 2 == 0 for i in range(n)], dtype=torch.bool)

        # Hand-rolled expected value: per-token reverse KL where mask=True,
        # forward KL where mask=False, averaged over n.
        t_lp = F.log_softmax(t.float(), dim=-1)
        s_lp = F.log_softmax(s.float(), dim=-1)
        rev = (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1)
        fwd = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1)
        per_tok = torch.where(mask, rev, fwd)
        expected = per_tok.mean()

        loss, _ = _CB.compute(t, s, mask)
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Liger vs standard numerical equivalence
# ---------------------------------------------------------------------------


class TestLigerEquivalence:
    @pytest.mark.parametrize("n_tokens,vocab_size", [
        (1, 10),
        (64, 256),
        (300, 5000),   # larger than liger chunk size = 256
        (1024, 2000),  # several chunks
    ])
    def test_liger_vs_standard_value(self, n_tokens, vocab_size):
        t = _make_logits(n_tokens, vocab_size, seed=7)
        s = _make_logits(n_tokens, vocab_size, seed=8)
        # Mixed mask, including boundary-crossing patterns.
        mask = torch.tensor(
            [((i // 17) % 3) != 0 for i in range(n_tokens)], dtype=torch.bool,
        )

        loss_std, nt_std = _CB.compute(t.clone(), s.clone(), mask)
        loss_lig, nt_lig = _CB.compute_liger(t.clone(), s.clone(), mask)

        assert nt_std == nt_lig == n_tokens
        torch.testing.assert_close(loss_std, loss_lig, atol=1e-5, rtol=1e-4)

    def test_liger_vs_standard_gradient(self):
        n, v = 128, 512
        t = _make_logits(n, v, seed=9)
        s_std = _make_logits(n, v, seed=10).float().requires_grad_(True)
        s_lig = s_std.data.clone().requires_grad_(True)
        mask = torch.tensor(
            [((i // 7) % 2) == 0 for i in range(n)], dtype=torch.bool,
        )

        loss_std, _ = _CB.compute(t.clone(), s_std, mask)
        loss_std.backward()

        loss_lig, _ = _CB.compute_liger(t.clone(), s_lig, mask)
        loss_lig.backward()

        assert s_std.grad is not None
        assert s_lig.grad is not None
        torch.testing.assert_close(s_std.grad, s_lig.grad, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# Gradient flow + edge cases
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_grad_nonzero(self):
        n, v = 32, 128
        t = _make_logits(n, v, seed=11)
        s = _make_logits(n, v, seed=12, requires_grad=True)
        mask = torch.tensor([i % 3 == 0 for i in range(n)], dtype=torch.bool)

        loss, _ = _CB.compute(t, s, mask)
        loss.backward()

        assert s.grad is not None
        assert s.grad.abs().sum() > 0

    def test_grad_flows_to_forward_tokens(self):
        """Forward KL tokens should contribute nonzero gradient (mass-covering)."""
        n, v = 16, 64
        t = _make_logits(n, v, seed=13)
        s = _make_logits(n, v, seed=14, requires_grad=True)
        mask = torch.zeros(n, dtype=torch.bool)  # all forward

        loss, _ = _CB.compute(t, s, mask)
        loss.backward()
        assert s.grad.abs().sum() > 0


class TestEdgeCases:
    def test_empty_input(self):
        t = torch.empty(0, 16, dtype=torch.bfloat16)
        s = torch.empty(0, 16, dtype=torch.bfloat16)
        mask = torch.empty(0, dtype=torch.bool)

        loss, nt = _CB.compute(t, s, mask)
        assert nt == 0
        assert loss.item() == 0.0

        loss_l, nt_l = _CB.compute_liger(t, s, mask)
        assert nt_l == 0
        assert loss_l.item() == 0.0

    def test_return_per_token(self):
        n, v = 32, 64
        t = _make_logits(n, v, seed=15)
        s = _make_logits(n, v, seed=16)
        mask = torch.tensor([i % 2 == 0 for i in range(n)], dtype=torch.bool)

        loss, nt, ptk = _CB.compute(t, s, mask, return_per_token=True)
        assert ptk.shape == (n,)
        # mean of per-token matches scalar loss
        torch.testing.assert_close(ptk.mean(), loss.detach(), atol=1e-5, rtol=1e-4)

    def test_shape_mismatch_raises(self):
        t = _make_logits(16, 32, seed=17)
        s = _make_logits(16, 32, seed=18)
        bad_mask = torch.ones(10, dtype=torch.bool)
        with pytest.raises(AssertionError):
            _CB.compute(t, s, bad_mask)


class TestTokenWeights:
    """token_weights should scale per-token KL before averaging."""

    def test_uniform_weights_match_unweighted(self):
        n, v = 16, 64
        t = _make_logits(n, v, seed=19)
        s = _make_logits(n, v, seed=20)
        mask = torch.tensor([i % 2 == 0 for i in range(n)], dtype=torch.bool)

        loss_unweighted, _ = _CB.compute(t.clone(), s.clone(), mask)
        weights = torch.ones(n)
        loss_weighted, _ = _CB.compute(t.clone(), s.clone(), mask, token_weights=weights)
        torch.testing.assert_close(loss_weighted, loss_unweighted, atol=1e-5, rtol=1e-4)

    def test_doubled_weights_doubles_loss(self):
        n, v = 16, 64
        t = _make_logits(n, v, seed=21)
        s = _make_logits(n, v, seed=22)
        mask = torch.tensor([i % 2 == 0 for i in range(n)], dtype=torch.bool)

        loss_unweighted, _ = _CB.compute(t.clone(), s.clone(), mask)
        weights = torch.full((n,), 2.0)
        loss_weighted, _ = _CB.compute(t.clone(), s.clone(), mask, token_weights=weights)
        torch.testing.assert_close(
            loss_weighted, loss_unweighted * 2.0, atol=1e-5, rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Standalone forward-KL loss (mirrors _compute_forward_kl_loss{,_liger}
# in opsd_worker.py). Namespace-extracted to avoid VERL imports.
# ---------------------------------------------------------------------------


class _FK:
    @staticmethod
    def compute(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        chunk_size: int = 512,
        token_weights=None,
        return_per_token: bool = False,
    ):
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
            t_lp = F.log_softmax(teacher_logits[start:end].float(), dim=-1)
            s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
            t_p = t_lp.exp()
            kl_chunk = (t_p * (t_lp - s_lp)).sum(dim=-1)
            del t_lp, s_lp, t_p

            if return_per_token:
                per_token_parts.append(kl_chunk.detach())
            if token_weights is not None:
                kl_chunk = kl_chunk * token_weights[start:end]
            kl_sum = kl_sum + kl_chunk.sum()
            del kl_chunk

        loss = kl_sum / n_tokens
        if return_per_token:
            return loss, n_tokens, torch.cat(per_token_parts, dim=0)
        return loss, n_tokens

    @staticmethod
    def compute_liger(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        chunk_size: int = 256,
        token_weights=None,
        return_per_token: bool = False,
    ):
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

            kl_chunk = F.kl_div(s_lp, t_lp, reduction="none", log_target=True).sum(dim=-1)
            del t_lp, s_lp

            if return_per_token:
                per_token_parts.append(kl_chunk.detach())
            if token_weights is not None:
                kl_chunk = kl_chunk * token_weights[start:end]
            kl_sum = kl_sum + kl_chunk.sum()
            del kl_chunk

        loss = kl_sum / n_tokens
        if return_per_token:
            return loss, n_tokens, torch.cat(per_token_parts, dim=0)
        return loss, n_tokens


class TestForwardKLStandalone:
    """Parity for the standalone forward-KL loss used by OPSD_LOSS_TYPE=forward_kl."""

    @pytest.mark.parametrize("n_tokens,vocab_size", [
        (1, 10),
        (16, 128),
        (300, 2000),
    ])
    def test_forward_kl_matches_reference(self, n_tokens, vocab_size):
        t = _make_logits(n_tokens, vocab_size, seed=11)
        s = _make_logits(n_tokens, vocab_size, seed=12)
        loss_fk, _ = _FK.compute(t, s)
        loss_ref = _ref_forward_kl(t, s)
        torch.testing.assert_close(loss_fk, loss_ref, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("n_tokens,vocab_size,chunk", [
        (1, 10, 512),
        (16, 128, 4),
        (300, 2000, 64),
    ])
    def test_forward_kl_liger_matches_standard(self, n_tokens, vocab_size, chunk):
        t = _make_logits(n_tokens, vocab_size, seed=13)
        s = _make_logits(n_tokens, vocab_size, seed=14)
        loss_std, _ = _FK.compute(t.clone(), s, chunk_size=chunk)
        loss_lig, _ = _FK.compute_liger(t.clone(), s, chunk_size=chunk)
        torch.testing.assert_close(loss_lig, loss_std, atol=1e-5, rtol=1e-4)

    def test_forward_kl_token_weights_mean_preserving(self):
        n, v = 32, 128
        t = _make_logits(n, v, seed=15)
        s = _make_logits(n, v, seed=16)

        # Uniform unit weights reproduce unweighted loss.
        loss_unweighted, _ = _FK.compute(t.clone(), s.clone())
        loss_unit, _ = _FK.compute(t.clone(), s.clone(), token_weights=torch.ones(n))
        torch.testing.assert_close(loss_unit, loss_unweighted, atol=1e-5, rtol=1e-4)

        # Constant scaling factors out cleanly.
        loss_scaled, _ = _FK.compute(t.clone(), s.clone(), token_weights=torch.full((n,), 2.0))
        torch.testing.assert_close(
            loss_scaled, loss_unweighted * 2.0, atol=1e-5, rtol=1e-4,
        )

        # Non-uniform weights match a hand-computed weighted mean over n tokens
        # (denominator stays n_tokens, matching the loss's mean-preserving contract).
        weights = torch.linspace(0.1, 3.0, n)
        t_lp = F.log_softmax(t.float(), dim=-1)
        s_lp = F.log_softmax(s.float(), dim=-1)
        per_tok = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1)
        expected = (per_tok * weights).sum() / n
        loss_weighted, _ = _FK.compute(t.clone(), s.clone(), token_weights=weights)
        torch.testing.assert_close(loss_weighted, expected, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
