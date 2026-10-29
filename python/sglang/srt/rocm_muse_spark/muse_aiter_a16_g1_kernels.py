"""Integer-only local-block compaction; small-align and AOT use the same copy.

The metadata producer contract supplies valid expert IDs, slot IDs, and num_post.
All output metadata is overwritten each call, including tails/zero-local routes.
No global Tensor caches; buffers are scoped to one forwarded G1.
"""
import triton
import triton.language as tl

@triton.jit
def _compact(SID,EID,POST,VSID,VEID,VPOST,SLOT, N:tl.constexpr, NB:tl.constexpr, B:tl.constexpr, M:tl.constexpr, SLOTS:tl.constexpr):
    b=tl.arange(0,B)
    post=tl.load(POST)
    e=tl.load(EID+b,b<NB,other=-1)
    live=(b<NB)&(b*16<post)&(e>=0)&(e<64)
    rank=tl.cumsum(live.to(tl.int32))-1
    count=tl.sum(live.to(tl.int32),0)
    tl.store(VPOST,count*16)
    tl.store(VEID+rank,e,live)
    tl.store(VEID+b,0,(b<NB)&(b>=count))
    j=tl.arange(0,16)
    r=b[:,None]*16+j[None,:]
    slot=tl.load(SID+r,(r<N)&(r<post),other=SLOTS)
    good=(slot>=0)&(slot<SLOTS)
    packed=tl.where(good,((slot%8)<<24)|(slot//8),M)
    dst=rank[:,None]*16+j[None,:]
    tl.store(VSID+dst,packed,live[:,None])
    tl.store(SLOT+dst,tl.where(good,slot,SLOTS),live[:,None])
    tail=(r<NB*16)&(r>=count*16)
    tl.store(VSID+r,M,tail)
    tl.store(SLOT+r,SLOTS,tail)

@triton.jit
def _scatter(SORTED,SLOT,POST,OUT, C:tl.constexpr, SLOTS:tl.constexpr):
    r=tl.program_id(0)
    col=tl.program_id(1)*C+tl.arange(0,C)
    post=tl.load(POST)
    slot=tl.load(SLOT+r)
    live=(r<post)&(slot>=0)&(slot<SLOTS)&(col<4096)
    v=tl.load(SORTED+r*4096+col,live,other=0.)
    tl.store(OUT+slot*4096+col,v,live)

def allocate(torch,meta,m):
    sid,eid,post=meta
    nb=eid.numel()
    return dict(vsid=torch.empty((nb*16,),device=sid.device,dtype=torch.int32),
                veid=torch.empty_like(eid),vpost=torch.empty_like(post),
                slots=torch.empty((nb*16,),device=sid.device,dtype=torch.int32),
                sorted_inter=torch.empty((nb*16,4096),device=sid.device,dtype=torch.bfloat16))

def compact(meta,b,m):
    sid,eid,post=meta
    # Return the actual compiled object for external diagnostics; retain no tensors.
    return _compact[(1,)](sid,eid,post,b['vsid'],b['veid'],b['vpost'],b['slots'],sid.numel(),eid.numel(),triton.next_power_of_2(eid.numel()),m,m*8,num_warps=4)

def scatter(b,inter,m):
    return _scatter[(b['slots'].numel(),4)](b['sorted_inter'],b['slots'],b['vpost'],inter,1024,m*8,num_warps=4)
