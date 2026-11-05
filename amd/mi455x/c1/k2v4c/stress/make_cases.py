"""CPU only: 8-row verify cases with a chosen number of distinct routed experts (nu), checked on the reference route.

Token t's router input x_t is optimized so that its top-8 is a chosen set S_t (64 picks, 64 - nu repeats); the case
is then built backwards through the pre-router chain (x = n rstd(n) (w_fn + 1), n = alpha stream + beta o_proj(core)):
core = small random (o_proj term ~1e-3 of the stream), stream = x / ((w_fn + 1) alpha). Every case is re-routed with
validation.seam_and_route (the K2 reference) and kept only if its nu is the target.
    make_cases.py <layers> <out.pt> [nus]
"""
import sys, time
import torch
sys.path.insert(0, "/data/muse455/k2/cand/k2v4_land1")
from port import consts as K
from port import validation as V
torch.set_num_threads(32)


def optimize_x(lay, sets, g, steps=400):
    """Router inputs [len(sets), H] (already normed as the kernel's x) whose top-8 is each set; ok flags."""
    w = lay.w
    wr, bias = w["w_router"].float(), w["bias"].float()
    m = len(sets)
    mask = torch.zeros(m, K.E, dtype=torch.bool)
    for t, st in enumerate(sets):
        mask[t, st] = True
    x = (torch.randn(m, K.H, generator=g) * 0.3).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    scale = (w["w_fn"].float() + 1.0)
    for i in range(steps):
        xn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * scale  # the norm the kernel applies
        sel = torch.sigmoid((xn @ wr.T) * K.ROUTER_MULT) + bias
        lo = sel.masked_fill(~mask, 1e9).min(-1).values
        hi = sel.masked_fill(mask, -1e9).max(-1).values
        if bool(((lo - hi) > 0.03).all()):
            break
        inn = sel.masked_fill(~mask, 0.0).sum(-1) / 8
        out = sel.masked_fill(mask, -1e9).topk(8, dim=-1).values.mean(-1)
        loss = (out - inn).sum() + 10 * torch.relu(0.04 - (lo - hi)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    ok = ((lo - hi) > 0.0).tolist()
    return (x.detach() * torch.rsqrt((x.detach() ** 2).mean(-1, keepdim=True) + 1e-6)) * scale, ok


def token_x(lay, used, nrep, g, steps=600):
    """One router input whose top-8 has exactly ``nrep`` experts of ``used`` (the easiest ones) and 8 - nrep fresh
    experts (whichever unused experts are easiest at the current x). Returns (x, set) or None."""
    w = lay.w
    wr, bias = w["w_router"].float(), w["bias"].float()
    scale = w["w_fn"].float() + 1.0
    usedm = torch.zeros(K.E, dtype=torch.bool)
    usedm[used] = True
    x = (torch.randn(1, K.H, generator=g) * 0.3).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    for i in range(steps):
        xn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * scale
        sel = (torch.sigmoid((xn @ wr.T) * K.ROUTER_MULT) + bias)[0]
        with torch.no_grad():
            rep = sel.masked_fill(~usedm, -1e9).topk(nrep).indices if nrep else torch.zeros(0, dtype=torch.long)
            fresh = sel.masked_fill(usedm, -1e9).topk(8 - nrep).indices
            m = torch.zeros(K.E, dtype=torch.bool)
            m[rep] = True
            m[fresh] = True
        lo = sel[m].min()
        hi = sel[~m].max()
        if float((lo - hi).detach()) > 0.03:
            break
        loss = torch.relu(0.04 - (sel[m].unsqueeze(1) - sel[~m].unsqueeze(0))).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if float(lo - hi) <= 0.0:
        return None
    xd = x.detach()
    return (xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + 1e-6)) * scale, m.nonzero()[:, 0].tolist()


def case_x(lay, nu, g):
    """8 router inputs with 64 - nu repeated picks (spread over tokens 1 .. 7)."""
    reps = [0] * 8
    for i in range(64 - nu):
        reps[1 + i % 7] += 1
    used, xs = [], []
    for t in range(8):
        for _ in range(6):
            r = token_x(lay, used, reps[t], g)
            if r is not None:
                break
        else:
            return None
        x, st = r
        used += [e for e in st if e not in used]
        xs.append(x[0])
    return torch.cat([v.unsqueeze(0) for v in xs]) if len(used) == nu else None


def build(lay, x, g):
    w = lay.w
    alpha, beta = w["alpha_a"].float(), w["beta_a"].float()
    sc = w["w_fn"].float() + 1.0
    n = torch.where(sc != 0, x / torch.where(sc != 0, sc, torch.ones_like(sc)), torch.zeros_like(x))  # gain 0: free
    n = n / n.pow(2).mean(-1, keepdim=True).sqrt()
    core = (torch.randn(8, K.CORE * K.TP, generator=g) * 0.05).to(torch.bfloat16)
    acc = torch.zeros(8, K.H)
    for r in range(K.TP):  # the reference's per-rank o_proj (seam_and_route, oproj_ranked)
        acc = acc + V.bf(core[:, r * K.CORE:(r + 1) * K.CORE].float() @ w["w_o"][:, r * K.CORE:(r + 1) * K.CORE].float().T)
    xo = V.bf(acc)
    b = xo * V.rstd(xo, K.EPS_POST_ATTN)
    s = 4.0  # n's scale is free (rstd removes it): the stream carries s n minus the branch
    stream = ((s * n - beta * b) / alpha).float()
    return V.Case(core, stream)


def main():
    layers = [int(v) for v in sys.argv[1].split(",")]
    nus = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "58,60,61,62,63,64").split(",")]
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    out = {}
    for layer in layers:
        t0 = time.time()
        lay = V.LayerData(layer, "cpu")
        g = torch.Generator().manual_seed(1000 + layer)
        cases = []
        gs = torch.Generator().manual_seed(7 + layer)
        for i in range(24):  # synthetic verify rows (validation.make_case): a shared direction of strength corr
            corr = 0.9 * i / 23
            case = V.make_case(lay, corr, int(torch.randint(1 << 30, (1,), generator=gs)), dev="cpu")
            r = V.seam_and_route(lay, case)
            cases.append({"core": case.core, "stream_a": case.stream_a, "nu": int(torch.unique(r["ids"]).numel()),
                          "gap": float(r["gap"].min()), "kind": "synthetic"})
        for nu in nus:
            got, tries = 0, 0
            while got < per and tries < 6 * per:
                tries += 1
                x = case_x(lay, nu, g)
                if x is None:
                    continue
                case = build(lay, x, g)
                r = V.seam_and_route(lay, case)
                n = int(torch.unique(r["ids"]).numel())
                gap = float(r["gap"].min())
                if n == nu:
                    cases.append({"core": case.core, "stream_a": case.stream_a, "nu": n, "gap": gap, "kind": "adv"})
                    got += 1
            print(f"L{layer} nu {nu}: {got}/{tries}", flush=True)
        out[layer] = cases
        print(f"L{layer}: {len(cases)} cases {time.time() - t0:.0f} s", flush=True)
    torch.save(out, sys.argv[2])


main()
