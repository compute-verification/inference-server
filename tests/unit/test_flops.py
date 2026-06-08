"""Unit tests for the exact FLOPs module (modules/proof_server/flops.py)."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server import flops as F

# A tiny hand-friendly shape for isolation tests.
TINY = F.ModelShape(L=1, d=8, h=2, d_h=4, f=16, V=10, h_kv=2)
QWEN17 = F.model_shape_from_config(F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"])


class TestFlops(unittest.TestCase):
    def test_decode_token_matches_hand_count(self):
        # Qwen3-1.7B, one decode token at s=500, fwd, logits=1 (see plan §2.6):
        #   W_layer = 4*2048*2048 + 4*2048*1024 + 6*2048*6144 = 100,663,296; *28 = 2,818,572,288
        #   head    = 2*2048*151936 = 622,329,856
        #   attn    = 28*4*16*128*500 = 114,688,000
        #   total   = 3,555,590,144
        self.assertEqual(F.flops(QWEN17, tokens=1, attended=500, mode="fwd", logits=1),
                         3_555_590_144)

    def test_mlp_term_uses_six_d_f_not_four(self):
        # With L=1, attended=0, logits=0, tokens=1 the only surviving term is
        # W_LAYER = 4*d*h*d_h + 4*d*h_kv*d_h + 6*d*f. The MLP sub-term is 6*d*f
        # (gate+up+down), NOT 4*d*f. Hand: 4*8*2*4 + 4*8*2*4 + 6*8*16
        #   = 256 + 256 + 768 = 1280.
        self.assertEqual(F.W_LAYER(TINY), 1280)
        self.assertEqual(F.flops(TINY, tokens=1, attended=0, mode="fwd", logits=0), 1280)

    def test_gqa_reduces_kv_projection_only(self):
        full = F.ModelShape(L=1, d=8, h=2, d_h=4, f=16, V=10, h_kv=2)
        gqa = F.ModelShape(L=1, d=8, h=2, d_h=4, f=16, V=10, h_kv=1)
        # (a) with attention off, smaller h_kv is cheaper (KV projections shrink)
        self.assertLess(F.flops(gqa, tokens=1, attended=0),
                        F.flops(full, tokens=1, attended=0))
        # (b) attention cost depends on h, not h_kv: tokens=0 isolates attention.
        self.assertEqual(F.flops(gqa, tokens=0, attended=100),
                         F.flops(full, tokens=0, attended=100))

    def test_lora_weight_term_is_double_forward(self):
        # attended=0 -> only the weight term, which is x2 under lora_bwd.
        fwd = F.flops(QWEN17, tokens=4, attended=0, mode="fwd", logits=1)
        lora = F.flops(QWEN17, tokens=4, attended=0, mode="lora_bwd", logits=1)
        self.assertEqual(lora, 2 * fwd)

    def test_lora_full_step_is_just_over_2x(self):
        # Pinned SMALL shape (batch=4, seq=64). The ratio is 2 + attention_fraction
        # and GROWS with seq (it exceeds 2.05 past ~1600 tokens) -- keep seq small.
        batch, seq = 4, 64
        tokens, attended, logits = batch * seq, batch * seq * (seq + 1) // 2, batch * seq
        fwd = F.flops(QWEN17, tokens, attended, mode="fwd", logits=logits)
        lora = F.flops(QWEN17, tokens, attended, mode="lora_bwd", logits=logits)
        self.assertTrue(2.0 < lora / fwd < 2.05, f"ratio={lora / fwd}")

    def test_attention_scales_linearly_with_attended(self):
        a = F.flops(QWEN17, tokens=0, attended=100)
        b = F.flops(QWEN17, tokens=0, attended=300)
        self.assertEqual(b, 3 * a)

    def test_bad_mode_raises(self):
        with self.assertRaises(ValueError):
            F.flops(TINY, tokens=1, attended=0, mode="full_bwd")

    def test_zero_work_is_zero(self):
        self.assertEqual(F.flops(TINY, tokens=0, attended=0, logits=0), 0)


class TestShapeLookup(unittest.TestCase):
    def test_bare_and_hf_keys_resolve_the_same(self):
        self.assertEqual(F.shape_for("Qwen/Qwen3-1.7B"),
                         F.shape_for("hf://Qwen/Qwen3-1.7B"))

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            F.shape_for("hf://nope/unknown")

    def test_config_fallbacks(self):
        # head_dim defaults to d//h; h_kv defaults to h.
        s = F.model_shape_from_config({"num_hidden_layers": 2, "hidden_size": 16,
                                       "num_attention_heads": 4, "intermediate_size": 32,
                                       "vocab_size": 100})
        self.assertEqual(s.d_h, 4)
        self.assertEqual(s.h_kv, 4)


if __name__ == "__main__":
    unittest.main()
