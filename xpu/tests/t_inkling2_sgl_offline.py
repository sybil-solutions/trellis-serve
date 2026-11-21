"""Drive the SGLang Inkling2 model class without a server: fake ForwardBatch / attention backend / KV pool.
Checks: prefill (chunked) + decode logits vs the reference forward; two interleaved requests.

  python3 t_inkling2_sgl_offline.py /models/Inkling-2-Mini
"""
import sys, faulthandler, json, types, torch
faulthandler.enable()
sys.path.insert(0, "/opt/trellis-serve/xpu")
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
import exl3xpu.inkling2.sgl_model as SM
from exl3xpu.inkling2.config import Inkling2LoopSglConfig

md = sys.argv[1]
dev = torch.device("xpu")
cfg = Inkling2LoopSglConfig(**json.load(open(f"{md}/config.json")))


class Pool:
    def __init__(self, layers, size, dtype):
        self.k = [torch.zeros(size, 8, 128, device=dev, dtype=dtype) for _ in range(layers)]
        self.v = [torch.zeros(size, 8, 128, device=dev, dtype=dtype) for _ in range(layers)]
    def set_kv_buffer(self, layer, loc, k, v, layer_id_override=None, **kw):
        self.k[layer_id_override][loc] = k.to(self.k[0].dtype); self.v[layer_id_override][loc] = v.to(self.v[0].dtype)
    def get_key_buffer(self, i): return self.k[i]
    def get_value_buffer(self, i): return self.v[i]


SIZE = 8192
pool = Pool(cfg.num_hidden_layers, SIZE, torch.float8_e4m3fn if "--fp8" in sys.argv else torch.bfloat16)
r2t = torch.zeros(8, 4096, dtype=torch.int32, device=dev)
r2t[0] = torch.arange(1, 4097, dtype=torch.int32); r2t[1] = torch.arange(4097, 8193, dtype=torch.int32) % SIZE
backend = types.SimpleNamespace(token_to_kv_pool=pool, req_to_token=r2t)
import sglang.srt.model_executor.forward_context as FC
FC.get_attn_backend = lambda: backend


class Mode:
    def __init__(self, kind): self.kind = kind
    def is_idle(self): return False
    def is_extend(self): return self.kind == "extend"
    def is_target_verify(self): return False
    def is_decode(self): return self.kind == "decode"


def fb_extend(slots, pres, ns):
    loc = torch.cat([r2t[s, p:p + n] for s, p, n in zip(slots, pres, ns)]).long()
    pos = torch.cat([torch.arange(p, p + n) for p, n in zip(pres, ns)]).to(dev)
    return types.SimpleNamespace(forward_mode=Mode("extend"), req_pool_indices=torch.tensor(slots, device=dev),
                                 extend_seq_lens_cpu=ns, extend_prefix_lens_cpu=pres, seq_lens_cpu=None, out_cache_loc=loc), pos


def fb_decode(slots, pres):
    loc = torch.stack([r2t[s, p] for s, p in zip(slots, pres)]).long()
    return types.SimpleNamespace(forward_mode=Mode("decode"), req_pool_indices=torch.tensor(slots, device=dev),
                                 seq_lens_cpu=torch.tensor([p + 1 for p in pres]), seq_lens=None, out_cache_loc=loc), torch.tensor(pres, device=dev)


import sglang.srt.layers.logits_processor as LP
LP.LogitsProcessor = lambda c: None
m = SM.Inkling2LoopForConditionalGeneration(cfg)
m.logits_processor = lambda ids, h, head, fb: torch.nn.functional.linear(h, head.weight).float()
with safe_open(f"{md}/model-00001-of-00001.safetensors", "pt", device="cpu") as f:
    m.load_weights((k, f.get_tensor(k)) for k in f.keys())
print("loaded", flush=True)

tok = PreTrainedTokenizerFast(tokenizer_file=f"{md}/tokenizer.json")
text = open(f"{md}/README.md").read()[2000:6000]
ids = torch.tensor(tok.encode(text)[:300], device=dev)
cap = 30.0
sc = lambda x: cap * torch.tanh(x / cap)

# request A: prefill 0..199 in two chunks (120 + 80), then decode 200..219 token by token
outs = []
fb, pos = fb_extend([0], [0], [120]); outs.append(m.forward(ids[:120], pos, fb))
fb, pos = fb_extend([0], [120], [80]); outs.append(m.forward(ids[120:200], pos, fb))
for i in range(200, 220):
    fb, pos = fb_decode([0], [i]); outs.append(m.forward(ids[i:i + 1], pos, fb))
logits = sc(torch.cat(outs))[:, :200058]
print("forward ok", logits.shape, flush=True)

from exl3xpu.inkling2.ref import Inkling2Ref
del m; torch.xpu.empty_cache()
ref = Inkling2Ref(md)
rl = ref.forward(ids[:220])
def kl(a, b):
    pa, pb = a.log_softmax(-1), b.log_softmax(-1)
    return (pa.exp() * (pa - pb)).sum(-1)
k = kl(rl.float(), logits.float())
print(f"KL vs ref: prefill max {k[:200].max().item():.2e} mean {k[:200].mean().item():.2e} | decode max {k[200:].max().item():.2e} mean {k[200:].mean().item():.2e}")
print(f"argmax agree {(rl.argmax(-1) == logits.argmax(-1)).float().mean().item():.3f}")
nll = torch.nn.functional.cross_entropy(logits[:-1].float(), ids[1:220]); print(f"ppl sgl {nll.exp().item():.3f}")
