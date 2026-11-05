"""Per-trace (root agent session) split of the window into time with >=1 request in flight
(latency on the critical path) and time with none (trace-fixed gaps). The progress rate of a
trace scales as 1/(gap + k*latency) when request latency scales by k, so its tokens scale the
same way. Token-weighted over traces this gives throughput per chip against k. CPU only.
Usage: python3 -I eval_trace_elasticity.py EXPORT OUT.json [--split ttft|decode]"""
import json,sys,collections
p,outp=sys.argv[1],sys.argv[2]; dur=600.0
R=[json.loads(l) for l in open(p) if l.strip()]
prof=[r for r in R if r["metadata"]["benchmark_phase"]!="warmup"]
t0=min(r["metadata"]["credit_issued_ns"] for r in prof)/1e9; t1=t0+dur
tr=collections.defaultdict(list)
for r in prof: tr[r["metadata"]["root_correlation_id"]].append(r)
rows=[]
for k,L in tr.items():
    iv=[];ivd=[]
    tok=0
    for r in L:
        m=r["metadata"];s=m["request_start_ns"]/1e9;e=m["request_end_ns"]/1e9
        ft=s+r["metrics"].get("time_to_first_token",{}).get("value",(e-s)*1e3)/1e3
        iv.append((max(s,t0),min(e,t1)));ivd.append((max(ft,t0),min(e,t1)))
        if e<=t1 and not m.get("was_cancelled"):
            tok+=r["metrics"]["input_sequence_length"]["value"]+r["metrics"].get("output_sequence_length",{}).get("value",0)
    def cover(I):
        I=sorted(x for x in I if x[1]>x[0]);c=0;cs=ce=None
        for a,b in I:
            if cs is None or a>ce:
                if cs is not None:c+=ce-cs
                cs,ce=a,b
            else: ce=max(ce,b)
        if cs is not None:c+=ce-cs
        return c
    first=max(t0,min(r["metadata"]["request_start_ns"]/1e9 for r in L))
    active=t1-first
    busy=cover(iv); dec=cover(ivd)
    rows.append({"trace":k[:8],"requests":len(L),"tokens":tok,"active_s":round(active,1),"in_flight_s":round(busy,1),"decoding_s":round(dec,1)})
T=sum(r["tokens"] for r in rows)
def tput(k_all=1.0,k_dec=None):
    s=0
    for r in rows:
        a=r["active_s"] or 1; l=r["in_flight_s"]/a; d=r["decoding_s"]/a
        if k_dec is None: new=(1-l)+k_all*l
        else: new=(1-d)+k_dec*d
        s+=r["tokens"]/new
    return s/dur/4
base=tput()
out={"traces":len(rows),"tokens":T,"base_tput_per_chip":round(base,1),
 "token_weighted_in_flight_share":round(sum(r["tokens"]*r["in_flight_s"]/max(r["active_s"],1) for r in rows)/T,3),
 "token_weighted_decoding_share":round(sum(r["tokens"]*r["decoding_s"]/max(r["active_s"],1) for r in rows)/T,3)}
for k in [0.95,0.9,0.8,0.5,0.0]:
    out[f"latency_x{k}"]=round(100*(tput(k)/base-1),2)
    out[f"decode_only_x{k}"]=round(100*(tput(k_dec=k)/base-1),2)
lo,hi=0.0,1.0
for _ in range(50):
    m=(lo+hi)/2
    if tput(m)>=18554*base/14585.9: lo=m
    else: hi=m
out["latency_scale_needed_for_target"]=round(lo,3)
out["rows"]=sorted(rows,key=lambda r:-r["tokens"])
json.dump(out,open(outp,"w"),indent=1)
print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=1))
for r in out["rows"][:18]: print(r)
