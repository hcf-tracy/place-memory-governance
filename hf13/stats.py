from __future__ import annotations
import math,random,re
from collections import defaultdict

def _mean(xs):return sum(xs)/len(xs) if xs else None

def bootstrap_ci(values, *, reps=4000, seed=20260907, alpha=.05):
    """IID bootstrap used only where the experimental unit is genuinely replicated (causal pairs)."""
    xs=[float(x) for x in values if x is not None]
    if not xs:return [None,None]
    if len(xs)==1:return [xs[0],xs[0]]
    rng=random.Random(seed);n=len(xs);bs=[]
    for _ in range(int(reps)):bs.append(sum(xs[rng.randrange(n)] for __ in range(n))/n)
    bs.sort();lo=max(0,min(len(bs)-1,int((alpha/2)*len(bs))));hi=max(0,min(len(bs)-1,int((1-alpha/2)*len(bs))-1));return [bs[lo],bs[hi]]

def conversation_block_interval(values,unit_ids,*,reps=4000,seed=20260907,alpha=.05):
    """Within-trajectory block sensitivity interval, NOT a population CI.

    Checkpoints are serially dependent. We first aggregate within conversation and
    then resample the four conversation blocks. This interval is explicitly labeled
    sensitivity-only because the benchmark contains one prespecified trajectory.
    """
    g=defaultdict(list)
    for v,u in zip(values,unit_ids):
        if v is None:continue
        m=re.match(r'c(\d+)_',str(u));key=f'conversation_{m.group(1)}' if m else 'unknown';g[key].append(float(v))
    blocks=[_mean(x) for _,x in sorted(g.items()) if x]
    if not blocks:return [None,None],0
    if len(blocks)==1:return [blocks[0],blocks[0]],1
    rng=random.Random(seed);n=len(blocks);bs=[]
    for _ in range(int(reps)):bs.append(_mean([blocks[rng.randrange(n)] for __ in range(n)]))
    bs.sort();lo=max(0,min(len(bs)-1,int((alpha/2)*len(bs))));hi=max(0,min(len(bs)-1,int((1-alpha/2)*len(bs))-1));return [bs[lo],bs[hi]],len(blocks)

def sign_test_p(deltas):
    ds=[float(x) for x in deltas if x is not None and abs(float(x))>1e-15];n=len(ds)
    if not n:return 1.0
    k=min(sum(x>0 for x in ds),sum(x<0 for x in ds));den=2**n;p=2*sum(math.comb(n,i) for i in range(k+1))/den
    return min(1.0,p)

def holm_adjust(ps):
    n=len(ps);order=sorted(range(n),key=lambda i:ps[i]);out=[1.0]*n;running=0.0
    for rank,i in enumerate(order):
        adj=min(1.0,(n-rank)*ps[i]);running=max(running,adj);out[i]=running
    return out

def format_p(p):return '<0.001' if p is not None and p<.001 else (f'{p:.3f}' if p is not None else '--')

def _paired_state(eval_a,eval_b):
    A={x['turn_id']:x['state_f1'] for x in eval_a['state_checkpoints']};B={x['turn_id']:x['state_f1'] for x in eval_b['state_checkpoints']};ids=sorted(set(A)&set(B));return [B[i]-A[i] for i in ids],ids

def _paired_query(eval_a,eval_b,key):
    A={x['turn_id']:x[key] for x in eval_a['queries']};B={x['turn_id']:x[key] for x in eval_b['queries']};ids=sorted(set(A)&set(B));return [B[i]-A[i] for i in ids],ids

def _paired_olr(pair_a,pair_b):
    A={x['pair_id']:x for x in pair_a};B={x['pair_id']:x for x in pair_b};ids=[];ds=[]
    for i in sorted(set(A)&set(B)):
        if A[i].get('eligible') and B[i].get('eligible'):ids.append(i);ds.append(float(B[i].get('olr_event',0))-float(A[i].get('olr_event',0)))
    return ds,ids

def _paired_olr_reduction(css_rows,place_rows):
    C={x['pair_id']:x for x in css_rows};P={x['pair_id']:x for x in place_rows};ids=[];ds=[]
    for i in sorted(set(C)&set(P)):
        if C[i].get('eligible') and P[i].get('eligible'):ids.append(i);ds.append(float(C[i].get('olr_event',0))-float(P[i].get('olr_event',0)))
    return ds,ids

def _paired_interaction(primary_evals,key):
    maps={c:{x['turn_id']:float(x[key]) for x in primary_evals[c]['queries']} for c in ('native','css','qasc','place')};ids=sorted(set.intersection(*(set(maps[c]) for c in maps)))
    return [((maps['place'][i]-maps['qasc'][i])-(maps['css'][i]-maps['native'][i])) for i in ids],ids

def _descriptive_effect(name,comparison,metric,ds,ids,reps,seed):
    ci,nblocks=conversation_block_interval(ds,ids,reps=reps,seed=seed)
    return {'effect_id':name,'comparison':comparison,'metric':metric,'estimate':_mean(ds),'interval95':ci,'interval_type':'within-trajectory conversation-block sensitivity interval','p':None,'p_H':None,'p_H_display':'--','n_checkpoints':len(ds),'conversation_blocks':nblocks,'statistical_unit':'prespecified longitudinal checkpoint nested in one trajectory','formal_population_inference':False,'unit_ids':ids}

def _pair_effect(name,comparison,metric,ds,ids,reps,seed):
    p=sign_test_p(ds)
    return {'effect_id':name,'comparison':comparison,'metric':metric,'estimate':_mean(ds),'interval95':bootstrap_ci(ds,reps=reps,seed=seed),'interval_type':'pair-level bootstrap 95% CI','p':p,'p_H':p,'p_H_display':format_p(p),'n_pairs':len(ds),'statistical_unit':'causal paired world','formal_population_inference':True,'unit_ids':ids}

def _reliability(pe):
    n=max(1,int(pe.get('row_count') or 0));model=int(pe.get('model_operation_error_turn_count') or 0);rec=int(pe.get('provider_recovered_turn_count') or 0);failed=int(pe.get('failed_turn_count') or 0)
    return {'model_operation_error_turns':model,'model_operation_error_rate':model/n,'recovered_turns':rec,'recovered_rate':rec/n,'failed_turns':failed,'failed_rate':failed/n,'terminal_status':pe.get('status')}

def aggregate(primary_evals,pair_rows,utility_rows,fault_rows,closed_evals, *, bootstrap_reps=4000):
    conditions=['native','css','qasc','place'];tables={}
    from .pairs import summarize as pair_summary
    from .utility import summarize as utility_summary
    from .fault import summarize as fault_summary
    for c in conditions:
        pe=primary_evals[c];ps=pair_summary(pair_rows[c]);us=utility_summary(utility_rows[c]);fs=fault_summary(fault_rows[c]);cl=closed_evals[c]
        tables[c]={
            'State F1':pe['state_f1_mean'],'OLR':ps['olr'],'OLR_n':ps['eligible_n'],'Unauthorized Support Exposure':pe['unauthorized_support_exposure_rate'],'GVR':pe['gvr_mean'],'Answer F1':pe['answer_f1_mean'],'Needed SRA':pe['needed_sra_mean'],'JAUR':pe['jaur_mean'],
            'Utility Needed SRA':us['needed_sra'],'Utility Needed SRA_numerator':us['needed_success_n'],'Utility Needed SRA_denominator':us['needed_n'],'Utility Needed SRA_total_cases':us.get('needed_total_n'),'Utility Needed SRA_inapplicable':us.get('needed_inapplicable_n'),'Utility Harm':us['harm'],'Utility Harm_numerator':us['conflict_harm_n'],'Utility Harm_denominator':us['conflict_n'],'Utility Exposure':us.get('exposure'),'Utility GVR':us.get('gvr'),'Utility Answer F1':us['answer_f1'],'Utility by family':us.get('by_family'),
            'Fault Full-state pre F1':fs['pre_state_f1'],'Fault Full-state post F1':fs['post_state_f1'],'Fault Repair dF1':fs['repair_delta_f1'],'Fault Slot pre F1':fs['pre_faulted_slot_f1'],'Fault Slot post F1':fs['post_faulted_slot_f1'],'Fault Slot dF1':fs['faulted_slot_delta_f1'],'Fault Repair Success':fs['repair_success'],'Fault Repair Success_numerator':fs['repair_success_n'],'Fault Repair Success_denominator':fs['repair_n'],'Fault Exposure':fs.get('exposure'),'Fault GVR':fs.get('gvr'),'Fault Answer F1':fs['post_answer_f1'],'Fault by class':fs.get('by_fault_class'),'Fault by family':fs.get('by_family'),
            'Closed State F1':cl['state_f1_mean'],'Closed Exposure':cl['unauthorized_support_exposure_rate'],'Closed GVR':cl['gvr_mean'],'Closed Answer F1':cl['answer_f1_mean'],'Closed JAUR':cl['jaur_mean'],'Closed OLR':None,'Closed OLR reason':'paired causal worlds not executed','runtime_reliability':_reliability(pe),'pair_family_metrics':ps.get('family_metrics'),'primary_family_metrics':pe.get('family_metrics')}

    main=[]
    ds,ids=_paired_state(primary_evals['native'],primary_evals['css']);main.append(_descriptive_effect('css_on_state_f1','CSS-Native','State F1',ds,ids,bootstrap_reps,20260971))
    ds,ids=_paired_olr_reduction(pair_rows['css'],pair_rows['place']);main.append(_pair_effect('qasc_on_olr_css_minus_place','CSS-PLACE','OLR reduction',ds,ids,bootstrap_reps,20260972))
    ds,ids=_paired_query(primary_evals['native'],primary_evals['place'],'jaur');main.append(_descriptive_effect('place_on_jaur','PLACE-Native','JAUR',ds,ids,bootstrap_reps,20260973))
    ds,ids=_paired_interaction(primary_evals,'jaur');main.append(_descriptive_effect('css_x_qasc_on_jaur','(PLACE-QASC)-(CSS-Native)','JAUR interaction',ds,ids,bootstrap_reps,20260974))

    # Secondary effects are descriptive except causal-pair OLR contrasts.
    exploratory=[]
    ds,ids=_paired_state(primary_evals['native'],primary_evals['css']);exploratory.append(_descriptive_effect('write_css','CSS-Native','State F1',ds,ids,bootstrap_reps,20261001))
    ds,ids=_paired_state(primary_evals['qasc'],primary_evals['place']);exploratory.append(_descriptive_effect('write_place','PLACE-QASC','State F1',ds,ids,bootstrap_reps,20261002))
    ds,ids=_paired_olr(pair_rows['native'],pair_rows['qasc']);exploratory.append(_pair_effect('read_qasc','QASC-Native','OLR',ds,ids,bootstrap_reps,20261003))
    ds,ids=_paired_olr(pair_rows['css'],pair_rows['place']);exploratory.append(_pair_effect('read_place','PLACE-CSS','OLR',ds,ids,bootstrap_reps,20261004))
    for key,name in [('unauthorized_support_exposure','exposure'),('gvr','gvr'),('answer_f1','answer'),('jaur','jaur')]:
        ds,ids=_paired_query(primary_evals['native'],primary_evals['qasc'],key);exploratory.append(_descriptive_effect(f'{name}_qasc','QASC-Native',key,ds,ids,bootstrap_reps,20261100+len(exploratory)))
        ds,ids=_paired_query(primary_evals['css'],primary_evals['place'],key);exploratory.append(_descriptive_effect(f'{name}_place','PLACE-CSS',key,ds,ids,bootstrap_reps,20261200+len(exploratory)))
    return {'schema_version':'hf13-aplus-statistics-v3','condition_table':tables,'main_prespecified_effects':main,'main_confirmatory_effects':main,'confirmatory_paired_effects':main,'exploratory_secondary_effects':exploratory,
            'statistical_unit_policy':{'longitudinal_trajectory_count':1,'conversation_count':4,'state_checkpoints':105,'query_checkpoints':48,'longitudinal_population_inference':False,'wording':'105 state checkpoints and 48 query checkpoints are serial observations nested within one prespecified four-conversation trajectory, not independent trajectories. Longitudinal intervals are within-trajectory conversation-block sensitivity intervals and must not be described as population 95% CIs. Formal HF13 inference is restricted to the independently constructed causal-pair OLR estimand.'},
            'notes':{'main_inference_policy':'State/JAUR longitudinal effects are descriptive with block sensitivity intervals; only causal-pair OLR receives pair-level p/CI. Controlled HF11/HF12 replications should carry population-style confirmatory inference.','boundary_values':'Rates are reported exactly as observed; no jittering, smoothing, or manipulation is applied.','p_display':'Formal p-values below 0.001 are rendered as <0.001, never 0.000.','bootstrap_reps':bootstrap_reps}}
