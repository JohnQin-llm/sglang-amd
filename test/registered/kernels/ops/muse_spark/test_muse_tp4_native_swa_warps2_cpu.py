"""CPU-only actual-source guard tests; no Torch/Triton imports."""
import ast,importlib.util,json,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
BACKEND=ROOT/'python/sglang/srt/layers/attention/aiter_backend.py'
SUPPORT=ROOT/'test/registered/kernels/ops/muse_spark/test_muse_tp4_native_full_warps2_cpu.py'

def main():
    spec=importlib.util.spec_from_file_location('native_cpu_support',SUPPORT)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch=types.SimpleNamespace(bfloat16='bf16',float8_e4m3fn='fp8',int32='i32',int64='i64',float32='f32')
    par=types.SimpleNamespace(tp_size=4,attn_tp_size=4,moe_ep_size=4,attn_dcp_size=1)
    props=types.SimpleNamespace(gcnArchName='gfx950');torch.cuda=types.SimpleNamespace(current_device=lambda:0,get_device_properties=lambda _:props)
    ns={'torch':torch,'get_parallel':lambda:par}
    m.extract(BACKEND.read_text(),['_muse_tp4_native_low_warps_context','_muse_tp4_native_swa_low_warps_context','_muse_tp4_native_swa_low_warps_abi'],ns)
    runner=types.SimpleNamespace(is_draft_worker=False,model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(architectures=['MuseSparkForConditionalGeneration'])))
    ctx=ns['_muse_tp4_native_swa_low_warps_context'];assert ctx(runner,{'swa_gqa'}) and not ctx(runner,{'full_gqa'})
    def gate(b,heads=32):
        f=m.fixture(b,heads)
        return ns['_muse_tp4_native_swa_low_warps_abi'](f['q'],f['k'],f['v'],f['out'],f['cu'],f['seq'],f['table'],f['scale'],f['kd'],f['vd'],16,64,4)
    for b in range(1,21):assert gate(b)==(b in (10,12,14,16)) and not gate(b,16)
    for field in ('tp_size','attn_tp_size','moe_ep_size','attn_dcp_size'):
        old=getattr(par,field);setattr(par,field,8);assert not ctx(runner,{'swa_gqa'});setattr(par,field,old)
    for prop,obj,value in [('is_draft_worker',runner,True),('gcnArchName',props,'gfx942'),('architectures',runner.model_config.hf_config,['Other'])]:
        old=getattr(obj,prop);setattr(obj,prop,value);assert not ctx(runner,{'swa_gqa'});setattr(obj,prop,old)
    # Constructor must install this independent token through this exact guard.
    tree=ast.parse(BACKEND.read_text());found=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.If) and isinstance(n.test,ast.Call) and isinstance(n.test.func,ast.Name) and n.test.func.id=='_muse_tp4_native_swa_low_warps_context':found.append(n)
    assert len(found)==1
    assert ast.unparse(found[0])=='if _muse_tp4_native_swa_low_warps_context(model_runner, self.muse_verify_kernels):\n    self.muse_verify_kernels |= {\'swa_gqa_tp4_low_warps\'}'
    print(json.dumps({'status':'passed_CPU_only','batch_and_EP8_checks':40,'context_checks':9,'constructor_token':True,'GPU_executed':False}))
if __name__=='__main__':main()
