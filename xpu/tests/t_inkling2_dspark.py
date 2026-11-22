"""DSpark drafter acceptance, offline: greedy target decode with feature capture; at every step the drafter proposes
7 tokens from the same state a speculative server would have, compared with the target's greedy continuation.

  python3 t_inkling2_dspark.py /models/Inkling-2-Mini [n_new]
"""
import sys, os, time, types, json
sys.argv.append("--no-ref")
here = os.path.dirname(os.path.abspath(__file__))
# reuse the offline driver's fake pool / batches / model load (stops before its own checks via --dspark-import)
src = open(os.path.join(here, "t_inkling2_sgl_offline.py")).read()
cut = src.index('if "--prof-prefill" in sys.argv:')
exec(compile(src[:cut], "t_inkling2_sgl_offline.py", "exec"))
from exl3xpu.inkling2.dspark import Inkling2DSpark

N_NEW = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 160
mcfg = json.load(open(f"{md}/config.json"))["mtp_config"]
taps = mcfg["target_layer_ids"][:-1]
shifted = [v + 1 for v in taps]
m.feat_visits = {v: i for i, v in enumerate(taps + shifted)}
m.feat_visits = {v: i for i, v in enumerate(taps)}
SEL = {"full": [0, 1, 2, 3]}
VOC = {"full": 0}
dr = Inkling2DSpark(f"{md}/mtp.safetensors", mcfg, m, n_slots=4, dev=dev)
heads = {}
for v, k in VOC.items():
    dr.set_draft_vocab(k or dr.V); heads[v] = (dr.Vd, dr.head, dr.fold, dr.fold_b)
drs = {"full": (dr, 0)}
def use(v):
    dr.Vd, dr.head, dr.fold, dr.fold_b = heads[v]
# drafter graph replay time per vocab (bs 1, nd 1 and 3)
for v in VOC:
    use(v)
    for nd in (1, 3):
        bufs = (torch.tensor([0], device=dev), torch.tensor([5], device=dev), torch.tensor([10], device=dev))
        for _ in range(2): dr.propose(*bufs, nd)
        torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            dr.propose(*bufs, nd)
        for _ in range(3): g.replay()
        torch.xpu.synchronize(); t0 = time.perf_counter()
        for _ in range(20): g.replay()
        torch.xpu.synchronize(); print(f"drafter graph {v} nd={nd}: {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms", flush=True)
        del g
in_k = {k: 0 for k in VOC}; n_tok = 0
print("drafter loaded", flush=True)

try:
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer
    htok = get_tokenizer(md)
    chat = lambda q: htok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True,
                                              reasoning_effort="none")
except Exception as e:  # noqa
    print("chat template unavailable:", e); chat = None
prompts = ["Explain how a hash map works and when its operations degrade. Use a short example in Python.",
           "Write a Python function that parses an ISO-8601 date string without using datetime, with tests.",
           "List ten practical tips for reducing memory use in a long-running web server, one sentence each.",
           "Summarise the causes of the French Revolution in about 200 words."]
enc = [chat(q) if chat else tok.encode(q) for q in prompts]
if enc and not isinstance(enc[0], list):
    enc = [list(e["input_ids"]) if hasattr(e, "keys") else list(e) for e in enc]
enc.append(tok.encode(open(f"{md}/README.md").read()[2000:5000])[:400])

VAR = list(drs)
stats = {v: [] for v in VAR}
tprop = []
for pi, pr in enumerate(enc):
    ids = torch.tensor(pr, device=dev)
    P = ids.shape[0]
    fb, pos = fb_extend([0], [0], [P])
    lg = m.forward(ids, pos, fb)
    for v in VAR:
        d_, sl = drs[v]
        d_.ingest(torch.full((P,), sl, device=dev), pos.long(), m.last_feats[:, SEL[v]])
    gen = [int(lg[-1].argmax())]
    drafts = {v: [] for v in VAR}
    p = P
    for step in range(N_NEW):
        for v in VAR:
            d_, sl = drs[v]
            use(v)
            torch.xpu.synchronize(); t0 = time.perf_counter()
            d = d_.propose(torch.tensor([sl], device=dev), torch.tensor([gen[-1]], device=dev), torch.tensor([p], device=dev), 7)
            torch.xpu.synchronize(); tprop.append(time.perf_counter() - t0)
            drafts[v].append(d[0].tolist())
        fb, pos = fb_decode([0], [p])
        lg = m.forward(torch.tensor([gen[-1]], device=dev), pos, fb)
        for v in VAR:
            d_, sl = drs[v]
            d_.ingest(torch.full((1,), sl, device=dev), pos.long(), m.last_feats[:, SEL[v]])
        gen.append(int(lg[-1].argmax()))
        p += 1
    for v, k in VOC.items():
        in_k[v] += sum(1 for x in gen if x < (k or 10**9)); 
    n_tok += len(gen)
    line = f"prompt {pi}: P={P}"
    for v in VAR:
        ml = []
        for j, d in enumerate(drafts[v][: len(gen) - 8]):
            k = 0
            while k < 7 and d[k] == gen[j + 1 + k]:
                k += 1
            ml.append(k)
        stats[v] += ml
        line += f" | {v}: @4 {sum(1 + min(k, 3) for k in ml) / len(ml):.2f} @8 {sum(1 + k for k in ml) / len(ml):.2f}"
    print(line, flush=True)
    print("   text:", repr(tok.decode(gen[-120:]))[:300], flush=True)
for v, ml in stats.items():
    print(f"ALL {v}: accept @nq2 {sum(1 + min(k, 1) for k in ml) / len(ml):.3f} @nq4 {sum(1 + min(k, 3) for k in ml) / len(ml):.3f}"
          f" @nq8 {sum(1 + k for k in ml) / len(ml):.3f} ({len(ml)} steps)")
print("target tokens inside draft vocab:", {v: round(c / n_tok, 4) for v, c in in_k.items()})
tprop.sort()
print(f"propose time median {tprop[len(tprop) // 2] * 1e3:.1f} ms (eager)")
