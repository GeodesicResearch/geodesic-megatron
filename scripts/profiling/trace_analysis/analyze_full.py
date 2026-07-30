import gzip, json, sys, collections, re

path = sys.argv[1]; label = sys.argv[2]

def merge(intervals):
    """merge list of (s,e) -> (merged_list, total_len)"""
    if not intervals: return [], 0.0
    intervals.sort()
    out=[list(intervals[0])]
    for s,e in intervals[1:]:
        if s<=out[-1][1]:
            if e>out[-1][1]: out[-1][1]=e
        else:
            out.append([s,e])
    tot=sum(e-s for s,e in out)
    return out, tot

def subtract(a, b):
    """length of a minus b, both merged lists of [s,e]. returns length of (a \ b)."""
    # walk
    res=0.0; j=0; nb=len(b)
    for s,e in a:
        cur=s
        while j<nb and b[j][1]<=cur: j+=1
        k=j
        while k<nb and b[k][0]<e:
            bs,be=b[k]
            if bs>cur:
                res+= min(bs,e)-cur
            cur=max(cur,be)
            if cur>=e: break
            k+=1
        if cur<e: res+= e-cur
    return res

def categorize(nm):
    # NCCL first (split later by annotation, here just kernel-level)
    if nm.startswith('ncclDevKernel_SendRecv'): return 'nccl_sendrecv'
    if nm.startswith('ncclDevKernel_AllReduce'): return 'nccl_allreduce'
    if nm.startswith('ncclDevKernel_ReduceScatter'): return 'nccl_reducescatter'
    if nm.startswith('ncclDevKernel_AllGather'): return 'nccl_allgather'
    if nm.startswith('ncclDevKernel_AllToAll') or nm.startswith('ncclDevKernel_Broadcast'): return 'nccl_other'
    if nm.startswith('ncclDevKernel') or nm.startswith('ncclKernel'): return 'nccl_other'
    # Mamba / SSM
    if any(k in nm for k in ('_chunk_scan','_chunk_state','_state_passing','causal_conv1d','_make_chunk_sort','_sort_chunks_by_map','selective_scan','_bmm_chunk')):
        return 'mamba_ssm'
    # Attention
    if ('sdpa' in nm or 'flash' in nm or 'fmha' in nm or 'attention' in nm.lower()):
        return 'attention'
    # MoE routing/permute/topk/act
    if any(k in nm for k in ('_permute_kernel','_unpermute_kernel','_row_id_map','mbtopk','moe_act','fused__to_copy_mul_pow_relu','sort_pairs','RadixSort','cub::','DeviceRadix','DeviceSelect','DeviceScan','indexFuncLargeIndex','index_elementwise','gather_kernel','_moe')):
        return 'moe_route'
    # GEMM
    if nm.startswith('nvjet') or 'cutlass' in nm or 'gemm' in nm.lower() or 'cublas' in nm.lower() or 'wgmma' in nm.lower() and 'sdpa' not in nm:
        return 'gemm'
    # memcpy/memset
    if nm.startswith('Memcpy'):
        if 'DtoH' in nm: return 'memcpy_d2h'
        if 'HtoD' in nm: return 'memcpy_h2d'
        return 'memcpy_d2d'
    if nm.startswith('Memset'): return 'memset'
    # norm / layernorm / rmsnorm
    if 'layer_norm' in nm or 'LayerNorm' in nm or 'rms' in nm.lower() or 'norm' in nm.lower():
        return 'norm'
    # elementwise / copy / cat / fill
    if any(k in nm for k in ('elementwise','CatArrayBatched','FillFunctor','direct_copy','_to_copy','vectorized','reduce_kernel','Reduce','bfloat16_copy','triton_poi','triton_red','triton_')):
        return 'elementwise'
    return 'other'

with gzip.open(path,'rt') as f:
    data=json.load(f)
ev=data['traceEvents']
steps=[(e['ts'],e['dur']) for e in ev if e.get('cat')=='user_annotation' and str(e.get('name','')).startswith('ProfilerStep')]
w0,wd=steps[-1]; w1=w0+wd
window=wd/1e6

cat_dur=collections.Counter(); cat_cnt=collections.Counter()
cat_intervals=collections.defaultdict(list)
name_dur=collections.Counter(); name_cnt=collections.Counter()
allgpu=[]
for e in ev:
    c=e.get('cat')
    if c not in ('kernel','gpu_memcpy','gpu_memset'): continue
    ts=e['ts']
    if ts<w0 or ts>w1: continue
    d=e['dur']; nm=e['name']
    cat=categorize(nm)
    cat_dur[cat]+=d; cat_cnt[cat]+=1
    s=ts; en=ts+d
    cat_intervals[cat].append((s,en))
    allgpu.append((s,en))
    name_dur[nm]+=d; name_cnt[nm]+=1

# unions
_,U_all=merge([list(x) for x in allgpu])
cat_union={}
for cat,iv in cat_intervals.items():
    _,u=merge([list(x) for x in iv]); cat_union[cat]=u

# compute vs nccl unions for exposed calc
nccl_cats={'nccl_sendrecv','nccl_allreduce','nccl_reducescatter','nccl_allgather','nccl_other'}
compute_iv=[]; nccl_iv=[]
for cat,iv in cat_intervals.items():
    if cat in nccl_cats: nccl_iv+= [list(x) for x in iv]
    else: compute_iv+=[list(x) for x in iv]
comp_m,comp_u=merge(compute_iv)
nccl_m,nccl_u=merge(nccl_iv)
exposed_nccl=subtract(nccl_m, comp_m)
# also sendrecv-only exposed
sr_m,sr_u=merge([list(x) for x in cat_intervals.get('nccl_sendrecv',[])])
exposed_sr=subtract(sr_m, comp_m)

# NCCL split by annotation (gpu_user_annotation)
na=collections.Counter()
for e in ev:
    if e.get('cat')=='gpu_user_annotation':
        ts=e['ts']
        if ts<w0 or ts>w1: continue
        n=e['name']
        if n.startswith('nccl:'): na[n]+=e['dur']

print("=====",label,"=====")
print("window_s=%.4f"%window)
print("U_all(busy)=%.4f  idle=%.4f (%.1f%%)"%(U_all/1e6,(wd-U_all)/1e6, 100*(wd-U_all)/wd))
print("compute_union=%.4f  nccl_union=%.4f  exposed_nccl=%.4f  overlapped_nccl=%.4f"%(comp_u/1e6,nccl_u/1e6,exposed_nccl/1e6,(nccl_u-exposed_nccl)/1e6))
print("sendrecv_union=%.4f exposed_sendrecv=%.4f"%(sr_u/1e6, exposed_sr/1e6))
print("\nCATEGORY | sum_s | union_s | %win | count")
order=['gemm','mamba_ssm','attention','moe_route','norm','elementwise','other',
       'nccl_sendrecv','nccl_allreduce','nccl_reducescatter','nccl_allgather','nccl_other',
       'memcpy_d2h','memcpy_h2d','memcpy_d2d','memset']
for cat in order:
    if cat in cat_dur:
        print("%-20s %8.3f %8.3f %6.1f %8d"%(cat,cat_dur[cat]/1e6,cat_union[cat]/1e6,100*cat_dur[cat]/wd,cat_cnt[cat]))
print("SUM_all_cats sum_s=%.3f"%(sum(cat_dur.values())/1e6))
print("\nNCCL annotation split (gpu_user_annotation, sum_s):")
for n,d in na.most_common(12):
    print("  %-40s %8.3f"%(n,d/1e6))
print("\nTOP 25 kernels by sum:")
for nm,d in name_dur.most_common(25):
    print("  %8.3f %7d  %s"%(d/1e6,name_cnt[nm],nm[:110]))
print()
