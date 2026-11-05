"""Iteration 12: full-attention verify 3D kernel (iteration-11 tree) sweep of block rows, warps, segments,
tile, stages and waves_per_eu at 90K and 32K context, batch sizes 3/10/16. Graph-replay device time per call.
usage: sweep.py OUT.json CONTEXTS BATCHES"""
import sys, json, itertools, torch
sys.path.insert(0, "/home/yanyuqin/demo/muse455/campaign/c16/probes/attn")
import verify_attn_bench as vb
import aiter.ops.triton.attention.unified_attention as ua
orig_get = ua.get_unified_attention_config
OVR = {}
def patched(op, params, backend="triton", arch=None):
    c = dict(orig_get(op, params, backend, arch))
    if op == "attn_3d":
        for k in ("waves_per_eu", "num_stages"):
            if k in OVR: c[k] = OVR[k]
    return c
ua.get_unified_attention_config = patched
def gt(inp, out, iters=30):
    for _ in range(3): vb.run(inp, (-1,-1), out)
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        vb.run(inp, (-1,-1), out); torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(10): vb.run(inp, (-1,-1), out)
    torch.cuda.synchronize(); g.replay()
    a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters): g.replay()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)*1000/iters/10
VARIANTS=[{}]
for bm,nw in [(64,2),(64,4),(64,8),(128,4),(128,8),(128,16)]:
  for seg in [32,64,128]:
    for tile in [32,64]:
      for st in [2,3]:
        VARIANTS.append({"BM":bm,"NW":nw,"SEG":seg,"TILE":tile,"num_stages":st})
for w in [1,2,4]:
    VARIANTS.append({"BM":128,"NW":8,"SEG":64,"TILE":64,"num_stages":2,"waves_per_eu":w})
res=[]
for ctx in [int(c) for c in sys.argv[2].split(",")]:
  for bs in [int(b) for b in sys.argv[3].split(",")]:
    inp = vb.case_inputs(bs, ctx, seed=bs*7+ctx); ref=None
    for v in VARIANTS:
        OVR.clear(); OVR.update(v)
        ua._GFX12_VERIFY_3D_BLOCK_M=v.get("BM",128); ua._GFX12_VERIFY_3D_WARPS=v.get("NW",8)
        ua._GFX12_VERIFY_3D_SEGMENTS=v.get("SEG",64); ua._GFX12_VERIFY_3D_TILE=v.get("TILE",64)
        ua._GFX12_VERIFY_3D_STAGES=v.get("num_stages",2)
        out=torch.zeros_like(inp["q"])
        try: us=gt(inp,out)
        except Exception as ex:
            print(json.dumps({"batch":bs,"context":ctx,"cfg":v,"error":str(ex)[:200]}),flush=True); continue
        o=out.float()
        if ref is None: ref=o
        row={"batch":bs,"context":ctx,"cfg":v,"us":round(us,1),"nan":bool(o.isnan().any()),"max_abs_diff":float((o-ref).abs().max())}
        print(json.dumps(row),flush=True); res.append(row)
    del inp; torch.cuda.empty_cache()
json.dump(res,open(sys.argv[1],"w"),indent=1)
