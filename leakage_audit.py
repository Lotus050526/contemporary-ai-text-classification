"""仅核查现有方案，不调参：python leakage_audit.py。旧结果保持不变。"""
import argparse
import hashlib
import itertools
import json
import logging
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import main

ROOT = Path('results')
AUDIT = ROOT/'leakage_audit_run'
CONTEXT = {}
EVENTS = []

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def matrix_sha(x):
    x=x.tocsr()
    h=hashlib.sha256(str(x.shape).encode())
    for a in [x.indptr,x.indices,x.data]:
        h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()

def canonical(cfg):
    cfg=dict(cfg); cfg['dtype']=np.float64; cfg['ngram_range']=tuple(cfg['ngram_range'])
    return cfg

def set_context(name,d,tr,va,cfg,mode):
    texts=d.text.fillna('').astype(str).map(lambda s:main.clean(s,mode))
    CONTEXT.clear()
    CONTEXT.update(experiment=name,tr=tr,va=va,train_texts=main.text_digest(texts.iloc[tr]),
                   val_texts=main.text_digest(texts.iloc[va]), labels=d.target.iloc[tr].to_numpy(),cfg=cfg,mode=mode)

ORIGINAL_FT=TfidfVectorizer.fit_transform
ORIGINAL_TRANSFORM=TfidfVectorizer.transform
ORIGINAL_FITS={cls:cls.fit for cls in [MultinomialNB,LinearSVC,LogisticRegression]}

def traced_fit_transform(self, documents, y=None):
    assert len(documents)==len(CONTEXT['tr'])
    digest=main.text_digest(documents)
    assert digest==CONTEXT['train_texts'], '特征拟合输入不是指定训练子集'
    assert self.get_params()==TfidfVectorizer(**CONTEXT['cfg']).get_params(), '拟合配置错位'
    if isinstance(documents,pd.Series):
        np.testing.assert_array_equal(documents.index,CONTEXT['tr'])
    x=ORIGINAL_FT(self,documents,y)
    CONTEXT['train_matrix']=matrix_sha(x)
    EVENTS.append(dict(experiment=CONTEXT['experiment'],operation='vectorizer.fit_transform',rows=len(documents),
                       indices_sha256=main.indices_digest(CONTEXT['tr']),text_sha256=digest,matrix_sha256=CONTEXT['train_matrix']))
    return x

def traced_transform(self,documents):
    assert len(documents)==len(CONTEXT['va'])
    digest=main.text_digest(documents)
    assert digest==CONTEXT['val_texts'], '验证 transform 输入错位'
    if isinstance(documents,pd.Series):
        np.testing.assert_array_equal(documents.index,CONTEXT['va'])
    x=ORIGINAL_TRANSFORM(self,documents)
    CONTEXT['val_matrix']=matrix_sha(x)
    EVENTS.append(dict(experiment=CONTEXT['experiment'],operation='vectorizer.transform',rows=len(documents),
                       indices_sha256=main.indices_digest(CONTEXT['va']),text_sha256=digest,matrix_sha256=CONTEXT['val_matrix']))
    return x

def traced_model_fit(self,x,y,*args,**kwargs):
    assert x.shape[0]==len(CONTEXT['tr']) and matrix_sha(x)==CONTEXT['train_matrix']
    np.testing.assert_array_equal(np.asarray(y),CONTEXT['labels'])
    if isinstance(y,pd.Series):
        np.testing.assert_array_equal(y.index,CONTEXT['tr'])
    EVENTS.append(dict(experiment=CONTEXT['experiment'],operation=type(self).__name__+'.fit',rows=len(y),
                       matrix_sha256=matrix_sha(x),labels_sha256=main.indices_digest(y),
                       train_indices_sha256=main.indices_digest(CONTEXT['tr']),max_iter=getattr(self,'max_iter',None)))
    return ORIGINAL_FITS[type(self)](self,x,y,*args,**kwargs)

def split_check(d):
    result=[]; splits={}
    for seed in [42,21,84]:
        tr,va=train_test_split(np.arange(len(d)),test_size=.2,stratify=d.target,random_state=seed)
        saved=pd.read_csv(ROOT/f'split_{seed}.csv')
        assert saved.row_index.is_unique and len(saved)==len(d)
        np.testing.assert_array_equal(saved.row_index,np.arange(len(d)))
        np.testing.assert_array_equal(saved.target,d.target)
        assert set(saved.loc[saved.subset=='train','row_index'])==set(tr)
        assert set(saved.loc[saved.subset=='validation','row_index'])==set(va)
        intersection=len(set(tr)&set(va)); coverage=len(set(tr)|set(va))
        assert intersection==0 and coverage==7368 and len(tr)==5894 and len(va)==1474
        result.append(dict(seed=seed,train=len(tr),validation=len(va),row_intersection=intersection,coverage=coverage,
                           text_target_alignment='passed',matches_stratified_seed='passed'))
        splits[seed]=(tr,va)
    pd.DataFrame(result).to_csv(AUDIT/'split_checks.csv',index=False)
    return splits,result

def replay(d,splits):
    original=pd.read_csv(ROOT/'experiments.csv')
    main.OUT=AUDIT; main.ROWS.clear(); (AUDIT/'configs').mkdir(exist_ok=True)
    comparison=[]; basecache=None
    with patch.object(TfidfVectorizer,'fit_transform',traced_fit_transform), patch.object(TfidfVectorizer,'transform',traced_transform), \
         patch.object(MultinomialNB,'fit',traced_model_fit),patch.object(LinearSVC,'fit',traced_model_fit),patch.object(LogisticRegression,'fit',traced_model_fit):
        for old in original.to_dict('records'):
            ident=f"{old['group']}_{old['seed']}_{old['variant']}_{old['model']}_{old['parameter']}"
            config=json.loads((ROOT/f'configs/{ident}.json').read_text(encoding='utf-8'))
            cfg=canonical(config['features']); tr,va=splits[old['seed']]
            set_context(ident,d,tr,va,cfg,old['mode'])
            prepared=None
            if old['group']=='baseline':
                if basecache is None:
                    basecache=main.features(d.text.fillna('').astype(str),tr,va,cfg)
                else:
                    CONTEXT['train_matrix']=matrix_sha(basecache.train)
                    CONTEXT['val_matrix']=matrix_sha(basecache.validation)
                    EVENTS.append(dict(experiment=ident,operation='reuse_checked_baseline_features',
                                       train_matrix_sha256=CONTEXT['train_matrix'],validation_matrix_sha256=CONTEXT['val_matrix']))
                prepared=basecache
            row,vec,model,pred=main.run_one(d,tr,va,old['seed'],old['group'],old['variant'],old['model'],old['parameter'],cfg,old['mode'],prepared)
            deltas={k:row[k]-old[k] for k in ['train_accuracy','train_macro_f1','val_accuracy','val_macro_f1']}
            saved=pd.read_csv(AUDIT/f'validation_predictions/{ident}.csv')
            np.testing.assert_array_equal(saved.row_index,va)
            np.testing.assert_array_equal(saved.true_label,d.target.iloc[va])
            recomputed=main.metrics(saved.true_label,saved.predicted_label)
            assert abs(recomputed['macro_f1']-row['val_macro_f1'])<1e-12
            assert abs(recomputed['accuracy']-row['val_accuracy'])<1e-12
            comparison.append(dict(experiment=ident,seed=old['seed'],variant=old['variant'],
                                    replay_val_accuracy=row['val_accuracy'],replay_val_macro_f1=row['val_macro_f1'],
                                    max_metric_delta=max(abs(v) for v in deltas.values()),**deltas,
                                    final_converged=row['final_converged'],
                                    train_matrix_sha256=CONTEXT['train_matrix'],validation_matrix_sha256=CONTEXT['val_matrix']))
            # 对每个持久化模型按原训练文本独立核对词表和 IDF（不使用测试集）。
            manifest=dict(experiment=ident,seed=old['seed'],train_indices=tr.tolist(),validation_indices=va.tolist(),
                          train_text_sha256=CONTEXT['train_texts'],validation_text_sha256=CONTEXT['val_texts'],
                          train_matrix_sha256=CONTEXT['train_matrix'],validation_matrix_sha256=CONTEXT['val_matrix'],
                          vocabulary_sha256=hashlib.sha256(json.dumps({k:int(v) for k,v in vec.vocabulary_.items()},sort_keys=True).encode()).hexdigest(),
                          idf_sha256=hashlib.sha256(vec.idf_.tobytes()).hexdigest(),test_rows_fit=0,test_rows_validation=0)
            (AUDIT/'provenance').mkdir(exist_ok=True)
            write_json(AUDIT/f'provenance/{ident}.json',manifest)
            logging.info('REPLAY COMPARE %s max_delta=%.12g',ident,comparison[-1]['max_metric_delta'])
    pd.DataFrame(comparison).to_csv(AUDIT/'replay_comparison.csv',index=False)
    write_json(AUDIT/'fit_trace.json',EVENTS)
    main.OUT=ROOT
    return comparison

def check_saved_models(d,t,splits):
    locked=json.loads((ROOT/'locked_scheme.json').read_text(encoding='utf-8'))
    tr,va=splits[42]
    best=joblib.load(ROOT/'best_validation.joblib')
    cfg=canonical(locked['features'])
    texts=d.text.fillna('').astype(str).map(lambda s:main.clean(s,locked['mode']))
    independent=TfidfVectorizer(**cfg)
    a=independent.fit_transform(texts.iloc[tr]); b=independent.transform(texts.iloc[va])
    assert independent.vocabulary_==best['vectorizer'].vocabulary_
    np.testing.assert_allclose(independent.idf_,best['vectorizer'].idf_,rtol=0,atol=0)
    assert best['vectorizer'].get_params()==TfidfVectorizer(**cfg).get_params()
    assert best['model'].get_params()==main.model_for(locked['model'],locked['parameter']).get_params()
    pred=best['model'].predict(best['vectorizer'].transform(texts.iloc[va]))
    path=ROOT/'best_validation_predictions.csv'
    if path.exists():
        backup=AUDIT/'best_validation_predictions_before_audit.csv'
        backup.write_bytes(path.read_bytes())
    saved=pd.DataFrame(dict(row_index=va,true_label=d.target.iloc[va].to_numpy(),predicted_label=pred,
                            text_sha256=[hashlib.sha256(s.encode()).hexdigest() for s in d.text.iloc[va]]))
    saved.to_csv(path,index=False)
    # 重新从磁盘读取预测和原始 target，通过行号连接后评分，排除内存变量混用。
    restored=pd.read_csv(path)
    np.testing.assert_array_equal(restored.true_label,d.target.iloc[restored.row_index])
    scores=main.metrics(d.target.iloc[restored.row_index],restored.predicted_label)
    errors=int((restored.true_label!=restored.predicted_label).sum())
    for key,value in scores.items():
        assert abs(value-locked['seed42_metrics']['val_'+key])<1e-12
    assert errors==len(pd.read_csv(ROOT/'misclassified.csv'))
    np.testing.assert_array_equal(confusion_matrix(restored.true_label,restored.predicted_label,labels=range(10)),
                                  pd.read_csv(ROOT/'confusion_matrix.csv',index_col=0).to_numpy())
    selected=pd.read_csv(AUDIT/'validation_predictions/control_42_sublinear_LinearSVC_10.0.csv')
    joined=restored.merge(selected,on='row_index',suffixes=('_saved','_replay'),validate='one_to_one')
    assert len(joined)==1474 and (joined.predicted_label_saved==joined.predicted_label_replay).all()
    mlp=joblib.load(ROOT/'mlp_best.joblib'); mvec=mlp['vectorizer']
    check=TfidfVectorizer(**mvec.get_params()); check.fit(d.text.iloc[tr])
    assert check.vocabulary_==mvec.vocabulary_
    np.testing.assert_allclose(check.idf_,mvec.idf_,rtol=0,atol=0)
    probabilities=mlp['model'].predict_proba(mvec.transform(d.text.iloc[va]))
    mp=mlp['model'].classes_[probabilities.argmax(axis=1)]
    pd.DataFrame(dict(row_index=va,true_label=d.target.iloc[va].to_numpy(),predicted_label=mp)).to_csv(AUDIT/'mlp_best_validation_predictions.csv',index=False)
    history=pd.read_csv(ROOT/'mlp_epochs.csv'); epoch=int(mlp['epoch']); h=history[history.epoch==epoch].iloc[0]
    assert epoch==int(history.loc[history.val_macro_f1.idxmax(),'epoch'])
    assert abs(log_loss(d.target.iloc[va],probabilities,labels=mlp['model'].classes_)-h.val_cross_entropy)<1e-12
    assert abs(main.metrics(d.target.iloc[va],mp)['macro_f1']-h.val_macro_f1)<1e-12
    return dict(**scores,errors=errors,validation_rows=len(restored),predictions_recovered_from_saved_model=True,
                replay_predictions_identical=True,best_vocabulary_idf_training_only=True,mlp_epoch=epoch,mlp_idf_training_only=True),a,b,independent

def negative_control(d,splits,vec,a,b):
    locked=json.loads((ROOT/'locked_scheme.json').read_text(encoding='utf-8'))
    tr,va=splits[42]; original=d.target.iloc[tr].to_numpy()
    rng=np.random.default_rng(20261013); shuffled=rng.permutation(original)
    np.testing.assert_array_equal(np.bincount(original,minlength=10),np.bincount(shuffled,minlength=10))
    pd.DataFrame(dict(row_index=tr,original_label=original,shuffled_label=shuffled)).to_csv(AUDIT/'negative_control_training_labels.csv',index=False)
    start=time.perf_counter(); model=main.model_for(locked['model'],locked['parameter'])
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); model.fit(a,shuffled)
    pred=model.predict(b)
    pd.DataFrame(dict(row_index=va,true_label=d.target.iloc[va].to_numpy(),predicted_label=pred)).to_csv(AUDIT/'negative_control_validation_predictions.csv',index=False)
    result=dict(**main.metrics(d.target.iloc[va],pred),permutation_seed=20261013,training_rows=len(tr),validation_rows=len(va),
                changed_training_labels=int((original!=shuffled).sum()),training_label_counts=np.bincount(shuffled,minlength=10).tolist(),
                validation_labels_unchanged=True,train_only_vocabulary=True,model_params=model.get_params(),
                seconds=time.perf_counter()-start,warnings=[str(w.message) for w in caught],used_for_tuning=False)
    joblib.dump(dict(vectorizer=vec,model=model,train_indices=tr,validation_indices=va),AUDIT/'negative_control_model.joblib')
    write_json(AUDIT/'negative_control.json',result)
    logging.info('NEGATIVE CONTROL accuracy=%.6f macro-F1=%.6f',result['accuracy'],result['macro_f1'])
    return result

def normalize(s):
    return re.sub(r'\s+',' ',s).strip().lower()

def pairs_for(values,left,right):
    lmap=defaultdict(list); rmap=defaultdict(list)
    for i in left: lmap[values[i]].append(int(i))
    for i in right: rmap[values[i]].append(int(i))
    shared=sorted(set(lmap)&set(rmap))
    return dict(shared_unique=len(shared),left_rows=sum(len(lmap[v]) for v in shared),right_rows=sum(len(rmap[v]) for v in shared),
                sample_pairs=[dict(left_index=i,right_index=j) for v in shared[:10] for i in lmap[v][:2] for j in rmap[v][:2]])

def duplicate_diagnostics(d,t,splits):
    texts=d.text.astype(str).tolist()+t.text.astype(str).tolist(); n=len(d)
    raw=texts; norm=[normalize(s) for s in texts]
    body=[normalize(main.clean(s,'body')) for s in texts]
    report={'index_convention':'联合诊断索引0–7367为有标签行；7368–9824为测试行，测试原始行号=联合索引-7368。预测顺序清单单独用测试零基行号。'}
    for method,values in [('exact_full_text',raw),('lowercase_whitespace_full_text',norm),('lowercase_whitespace_body',body)]:
        row=dict(labeled_duplicate_extra_rows=int(pd.Series(values[:n]).duplicated().sum()),
                 test_duplicate_extra_rows=int(pd.Series(values[n:]).duplicated().sum()),
                 labeled_test=pairs_for(values,range(n),range(n,len(texts))))
        row['splits']={str(seed):pairs_for(values,tr,va) for seed,(tr,va) in splits.items()}
        nonempty_labeled=[i for i in range(n) if values[i].strip()]
        nonempty_test=[i for i in range(n,len(texts)) if values[i].strip()]
        row['nonempty_labeled_test']=pairs_for(values,nonempty_labeled,nonempty_test)
        row['nonempty_splits']={str(seed):pairs_for(values,[i for i in tr if values[i].strip()],[i for i in va if values[i].strip()])
                                for seed,(tr,va) in splits.items()}
        report[method]=row
    groups=defaultdict(list)
    for i,value in enumerate(body): groups[value].append(i)
    bodyrows=[]
    tr42,va42=splits[42]; trainset=set(tr42); valset=set(va42)
    for value,ids in groups.items():
        if len(ids)<2: continue
        group=hashlib.sha256(value.encode()).hexdigest()
        for i in ids:
            bodyrows.append(dict(group_sha256=group,dataset='labeled' if i<n else 'test',row_index=i if i<n else i-n,
                                 main_subset='train' if i in trainset else 'validation' if i in valset else 'test',
                                 target=int(d.target.iloc[i]) if i<n else None,normalized_body_chars=len(value),
                                 empty_body=not bool(value),normalized_body_snippet=value[:2000]))
    pd.DataFrame(bodyrows).to_csv(AUDIT/'normalized_body_duplicate_groups.csv',index=False)
    # 明确引用关系：完整匹配头部消息 ID 与 References/In-reply-to/正文 In article 的 ID。
    owners=defaultdict(set); refs=[]; subjectgroups=defaultdict(list)
    for i,s in enumerate(texts):
        p=main.header(s)
        if p:
            fields,pbody=p
            for k,v in fields:
                content=v.partition(':')[2].strip()
                if k in {'article-i.d.','message-id'}:
                    owners[content.strip('<> ').lower()].add(i)
                if k in {'references','in-reply-to'}:
                    refs += [(i,z.lower(),'header_reference') for z in re.findall(r'<([^<>\s]+)>',content)]
                if k=='subject':
                    title=normalize(content)
                    title=re.sub(r'^(?:(?:re|fw|fwd)(?:\[\d+\])?\s*:\s*)+','',title)
                    if title: subjectgroups[title].append(i)
        refs += [(i,z.lower(),'body_in_article') for z in re.findall(r'(?im)\bin article\s*<([^<>\s]+)>',s)]
    links=set()
    for src,ref,kind in refs:
        for dst in owners.get(ref,[]):
            if src!=dst: links.add((src,dst,kind,ref))
    tr42,va42=splits[42]; trainset=set(tr42); valset=set(va42)
    linkrows=[dict(source_index=i,referenced_index=j,kind=kind,message_id=ref,
                   source_dataset='labeled' if i<n else 'test',referenced_dataset='labeled' if j<n else 'test',
                   cross_main_split=(i in trainset and j in valset) or (j in trainset and i in valset))
              for i,j,kind,ref in sorted(links)]
    pd.DataFrame(linkrows,columns=['source_index','referenced_index','kind','message_id','source_dataset','referenced_dataset','cross_main_split']).to_csv(AUDIT/'message_reference_links.csv',index=False)
    # 逐行长引用匹配：至少 100 字符和 15 单词；至少一侧原行带 > 引用符。
    # 不比较普通词汇的重合；高频模板行（出现于超过10文档）排除。
    lines=defaultdict(dict)
    for i,s in enumerate(texts):
        for line in main.clean(s,'body').splitlines():
            quoted=bool(re.match(r'^\s*(?:[A-Za-z]{1,3})?\s*[>|]',line))
            stripped=re.sub(r'^\s*(?:(?:[A-Za-z]{1,3})?\s*[>|:]\s*)+','',line)
            linekey=normalize(stripped)
            if len(linekey)>=100 and len(linekey.split())>=15:
                lines[linekey][i]=lines[linekey].get(i,False) or quoted
    quote_pairs=defaultdict(list)
    for linekey,docs in lines.items():
        if 2<=len(docs)<=10 and any(docs.values()):
            for i,j in itertools.combinations(sorted(docs),2):
                if docs[i] or docs[j]: quote_pairs[(i,j)].append(linekey)
    qrows=[]
    for (i,j),matches in sorted(quote_pairs.items()):
        qrows.append(dict(left_index=i,right_index=j,left_dataset='labeled' if i<n else 'test',right_dataset='labeled' if j<n else 'test',
                          cross_main_split=(i in trainset and j in valset) or (j in trainset and i in valset),
                          shared_long_lines=len(matches),sample=matches[0]))
    pd.DataFrame(qrows,columns=['left_index','right_index','left_dataset','right_dataset','cross_main_split','shared_long_lines','sample']).to_csv(AUDIT/'long_quote_pairs.csv',index=False)
    samples=[]
    for r in [r for r in qrows if r['cross_main_split']][:5]:
        i,j=r['left_index'],r['right_index']
        samples.append(dict(**r,left_original=texts[i][:3500],right_original=texts[j][:3500]))
    write_json(AUDIT/'quote_pair_samples.json',samples)
    cross_titles=[dict(subject=title,train_indices=[i for i in ids if i in trainset],validation_indices=[i for i in ids if i in valset])
                  for title,ids in subjectgroups.items() if any(i in trainset for i in ids) and any(i in valset for i in ids)]
    write_json(AUDIT/'shared_subject_groups.json',cross_titles)
    report['relationships']=dict(exact_message_reference_links=len(linkrows),cross_main_message_reference_links=sum(r['cross_main_split'] for r in linkrows),
                               long_quote_pairs=len(qrows),cross_main_long_quote_pairs=sum(r['cross_main_split'] for r in qrows),
                               shared_subject_groups_main=len(cross_titles),
                               interpretation='引用匹配和主题分组是关联候选，不等于全文重复或证明程序泄露；普通词汇共享不计为泄露。')
    write_json(AUDIT/'duplicate_diagnostics.json',report)
    return report

def serialized_model_check(d,splits):
    checks=[]; independent_features={}
    experiments=pd.read_csv(AUDIT/'experiments.csv')
    for row in experiments.to_dict('records'):
        ident=f"{row['group']}_{row['seed']}_{row['variant']}_{row['model']}_{row['parameter']}"
        stored=joblib.load(AUDIT/f'models/{ident}.joblib')
        provenance=json.loads((AUDIT/f'provenance/{ident}.json').read_text(encoding='utf-8'))
        cfg=json.loads((AUDIT/f'configs/{ident}.json').read_text(encoding='utf-8'))
        tr,va=splits[row['seed']]
        np.testing.assert_array_equal(stored['train_indices'],tr)
        np.testing.assert_array_equal(stored['validation_indices'],va)
        assert stored['seed']==row['seed'] and stored['mode']==row['mode']
        assert stored['vectorizer'].get_params()==TfidfVectorizer(**canonical(cfg['features'])).get_params()
        assert stored['model'].get_params()==cfg['model']
        assert hashlib.sha256(stored['vectorizer'].idf_.tobytes()).hexdigest()==provenance['idf_sha256']
        texts=d.text.astype(str).map(lambda s:main.clean(s,row['mode']))
        assert stored['train_text_sha256']==main.text_digest(texts.iloc[tr])
        assert stored['validation_text_sha256']==main.text_digest(texts.iloc[va])
        a=stored['vectorizer'].transform(texts.iloc[tr]); b=stored['vectorizer'].transform(texts.iloc[va])
        key=(main.indices_digest(tr),row['mode'],json.dumps(cfg['features'],sort_keys=True))
        if key not in independent_features:
            independent=TfidfVectorizer(**canonical(cfg['features']))
            refit=independent.fit_transform(texts.iloc[tr])
            independent_features[key]=(independent,refit)
        independent,refit=independent_features[key]
        assert independent.vocabulary_==stored['vectorizer'].vocabulary_
        np.testing.assert_allclose(independent.idf_,stored['vectorizer'].idf_,rtol=0,atol=0)
        assert matrix_sha(refit)==provenance['train_matrix_sha256']
        # fit_transform 的 CSR 列索引可未排序；transform 会排序，L2求和顺序改变末位浮点数。
        # 保留原拟合摘要，另以稀疏差分验证相同数学特征，而不是伪造字节一致性。
        delta=a-refit
        max_delta=float(np.max(np.abs(delta.data))) if delta.nnz else 0.
        assert max_delta<1e-12
        assert matrix_sha(b)==provenance['validation_matrix_sha256']
        saved=pd.read_csv(AUDIT/f'validation_predictions/{ident}.csv')
        np.testing.assert_array_equal(saved.row_index,va)
        np.testing.assert_array_equal(saved.true_label,d.target.iloc[va])
        np.testing.assert_array_equal(saved.predicted_label,stored['model'].predict(b))
        for prefix,x,y in [('train',a,d.target.iloc[tr]),('val',b,d.target.iloc[va])]:
            for metric,value in main.metrics(y,stored['model'].predict(x)).items():
                assert abs(value-row[prefix+'_'+metric])<1e-12
        checks.append(dict(experiment=ident,seed=row['seed'],status='passed',indices_match=True,config_match=True,
                           idf_digest_match=True,independent_train_refit_digest_match=True,
                           transform_vs_fit_train_max_abs_difference=max_delta,validation_matrix_digest_match=True,
                           saved_predictions_match=True,train_validation_scores_match=True))
    pd.DataFrame(checks).to_csv(AUDIT/'serialized_model_checks.csv',index=False)
    return checks

def header_check(d,t):
    records=[]; samples=[]
    for name,frame in [('labeled',d),('test',t)]:
        ok=0; failed=0; colon_bodies=0
        sample_indices=set(np.random.default_rng(42).choice(len(frame),size=8,replace=False).tolist())
        for i,s in enumerate(frame.text.astype(str)):
            original=s.replace('\r\n','\n').replace('\r','\n')
            parsed=main.header(s)
            if parsed:
                ok+=1; fields,body=parsed
                assert body==original.partition('\n\n')[2]
                assert main.clean(s,'body')==body
                expected='\n'.join(v for k,v in fields if k not in main.SOURCE)+'\n\n'+body
                assert main.clean(s,'source')==expected
                assert any(k=='subject' for k,_ in fields)
                assert any(k=='subject' for k,_ in main.header(s)[0] if k not in main.SOURCE)
                if re.search(r'(?m)^\S[^\n]*:',body): colon_bodies+=1
                if i in sample_indices or (len([z for z in samples if z['dataset']==name and z['case']=='body_colon'])<2 and re.search(r'(?m)^\S[^\n]*:',body)):
                    samples.append(dict(dataset=name,row_index=i,case='body_colon' if re.search(r'(?m)^\S[^\n]*:',body) else 'random_success',
                                        original=s[:3000],source_removed=main.clean(s,'source')[:3000],body_only=body[:3000]))
            else:
                failed+=1; assert main.clean(s,'source')==s and main.clean(s,'body')==s
                if len([z for z in samples if z['dataset']==name and z['case']=='failed_unchanged'])<3:
                    samples.append(dict(dataset=name,row_index=i,case='failed_unchanged',original=s[:3000],cleaned=s[:3000]))
        records.append(dict(dataset=name,rows=len(frame),parsed=ok,failed_unchanged=failed,success_rate=ok/len(frame),
                            colon_body_documents_preserved=colon_bodies,body_exactly_preserved=True,subject_retained=True))
    write_json(AUDIT/'header_checks.json',records); write_json(AUDIT/'header_check_samples.json',samples)
    return records

def final_check(d,t):
    locked=json.loads((ROOT/'locked_scheme.json').read_text(encoding='utf-8'))
    bundle=joblib.load(ROOT/'final_model.joblib'); vec=bundle['vectorizer']; model=bundle['model']
    assert bundle['mode']==locked['mode']
    assert vec.get_params()==TfidfVectorizer(**canonical(locked['features'])).get_params()
    assert model.get_params()==main.model_for(locked['model'],locked['parameter']).get_params()
    texts=d.text.fillna('').astype(str).map(lambda s:main.clean(s,locked['mode']))
    independent=TfidfVectorizer(**canonical(locked['features'])); independent.fit(texts)
    assert independent.vocabulary_==vec.vocabulary_
    np.testing.assert_allclose(independent.idf_,vec.idf_,rtol=0,atol=0)
    # 重训一个核查副本，不覆盖已保存最终模型，也不产生训练内“验证指标”。
    reference_model=main.model_for(locked['model'],locked['parameter'])
    reference_model.fit(independent.transform(texts),d.target)
    np.testing.assert_allclose(reference_model.coef_,model.coef_,rtol=0,atol=1e-12)
    np.testing.assert_allclose(reference_model.intercept_,model.intercept_,rtol=0,atol=1e-12)
    testtexts=t.text.fillna('').astype(str).map(lambda s:main.clean(s,locked['mode']))
    pred=model.predict(vec.transform(testtexts))
    earlier_predictions=ROOT/'final_predictions.csv'
    if not earlier_predictions.is_file():
        earlier_predictions=Path('prediction.csv')
    old=pd.read_csv(earlier_predictions,header=None)[0].to_numpy()
    np.testing.assert_array_equal(pred,old)
    # 逐样本重算，独立检查整批预测的顺序，没有重排或删行。
    byrow=np.asarray([model.predict(vec.transform([s]))[0] for s in testtexts])
    np.testing.assert_array_equal(pred,byrow)
    path=Path('prediction.csv')
    if path.exists():
        (AUDIT/'prediction_before_audit.csv').write_bytes(path.read_bytes())
    pd.DataFrame(pred.astype(np.int64)).to_csv(path,index=False,header=False)
    lines=path.read_text(encoding='utf-8').splitlines()
    assert len(lines)==2457 and all(re.fullmatch(r'[0-9]',line) for line in lines)
    np.testing.assert_array_equal(pd.read_csv(path,header=None)[0],byrow)
    pd.DataFrame(dict(test_row_index=np.arange(len(t)),text_sha256=[hashlib.sha256(s.encode()).hexdigest() for s in t.text],
                      predicted_label=pred)).to_csv(AUDIT/'prediction_order_manifest.csv',index=False)
    result=dict(rows=len(lines),columns=1,no_header=True,no_index=True,integer_labels_0_to_9=True,
                all_2457_single_row_predictions_match=True,unchanged_test_order=True,old_final_predictions_identical=True,
                full_labeled_only_vocabulary_idf_match=True,locked_configuration_match=True,
                independent_full_labeled_refit_weights_match=True,
                output='prediction.csv',sha256=sha(path),sample_predictions_sha256=sha('predictions.csv') if Path('predictions.csv').is_file() else None,
                final_model_sha256=sha(ROOT/'final_model.joblib'),test_csv_sha256=sha('test_data_unlabeled.csv'),
                test_accuracy=None,note='从已保存最终模型生成，并非新选型；不计算最终模型的训练内指标作为验证指标。')
    write_json(AUDIT/'final_prediction_checks.json',result)
    return result

def source_positions():
    text=Path('main.py').read_text(encoding='utf-8').splitlines()
    def line(token): return next(i for i,s in enumerate(text,1) if token in s)
    tokens={'split':'def split(', 'feature_fit':'a = vec.fit_transform(texts.iloc[tr])','validation_transform':'b = vec.transform(texts.iloc[va])',
            'model_fit':'model.fit(a, d.target.iloc[tr])','validation_scores':'val_metrics = metrics(d.target.iloc[va], pred)',
            'cache_check':'def check(self, texts', 'clean':'def clean(', 'header':'def header(',
            'mlp':'def mlp_run(', 'mlp_fit':'model.partial_fit(a, d.target.iloc[tr]',
            'final_fit':'a=vec.fit_transform(d.text', 'final_model_fit':'model.fit(a,d.target)',
            'final_transform':'b=vec.transform(t.text', 'final_write':'pd.DataFrame(model.predict(b)).to_csv', 'load':'def load('}
    return {k:line(v) for k,v in tokens.items()}

def make_report(summary):
    p=source_positions(); s=summary; dup=s['duplicates']; rel=dup['relationships']
    exact=dup['exact_full_text']; norm=dup['lowercase_whitespace_full_text']; bod=dup['lowercase_whitespace_body']
    score=s['best_scores']; neg=s['negative_control']; replayrows=s['replay']; headers=s['headers']; final=s['final']
    lines=['# 数据泄露核查记录', '', '本文件是流程审计记录，不是最终实验报告。没有发布仓库、生成 PDF 或打包提交。', '',
           '实际命令：`.\\.venv\\Scripts\\python.exe leakage_audit.py`。审计独立输出至 `results/leakage_audit_run/`；旧模型、锁定配置和原实验表未覆盖。源码及重要旧产物备份在 `results/leakage_audit_snapshot/`。', '',
           '## 1. 划分与对齐：通过', '',
           f'- 代码：`main.py:{p["split"]}`，先对原始数据索引分层划分，stratify=target、test_size=0.2。',
           '- 实际核查 seed=42、21、84 各训练 5894、验证 1474；行号交集均为 0，合并均恰好覆盖 0–7367，无重复或遗漏；全部保存 target 与原 CSV 同行一致。',
           '- 证据：`split_42.csv`、`split_21.csv`、`split_84.csv`；`leakage_audit_run/split_checks.csv`。重新按种子生成的划分与保存索引完全一致，并保留原随机排列顺序用于重放。',
           f'- 清洗使用保持 Series 索引的逐行 map（`main.py:{p["clean"]}`）；训练特征与标签都用同一个 tr，验证特征与标签都用同一个 va，运行时检查索引、文本摘要和标签数组。',
           '- 无标签测试 2457 条仅参加结构检查和锁定后的预测；重放训练/验证的运行时断言分别只接受指定 5894/1474 行，fit_trace.json 没有测试输入。数据结构诊断不用于选型。', '',
           '## 2. 特征拟合与缓存：通过；补齐追溯证据', '',
           f'- `main.py:{p["feature_fit"]}` 仅 fit_transform(texts.iloc[tr])，`main.py:{p["validation_transform"]}` 仅 transform(texts.iloc[va])。不存在实验先全量拟合后划分或拼接测试拟合的路径。',
           '- 逐组读取原 configs/ 的实际配置，重放全部 26 组：主划分 9 个基础候选+5 个对照，21/84 每个种子 6 组固定对照，不搜索新参数。用运行时拦截检查每次词表拟合的文档内容、索引和配置，每次分类器 fit 的稀疏矩阵与 target 的顺序。',
           f'- 26 组重放训练/验证 accuracy、macro-F1 与原结果最大绝对差值为 {max(r["max_metric_delta"] for r in replayrows):.12g}；见 `leakage_audit_run/replay_comparison.csv`。',
           '- 基础候选只共享 seed42、BASE 的同一特征对象。原代码未存持久化特征矩阵，没有跨 seed 的磁盘特征缓存；新增 PreparedFeatures.check 绑定配置、训练/验证索引及顺序、清洗文本 SHA256，错误缓存直接拒绝。',
           f'- 缓存守卫代码：`main.py:{p["cache_check"]}`。验证错误种子和错误特征配置都被拒绝，见 cache_guard_checks.json。',
           '- 重放每组模型与索引存于 `leakage_audit_run/models/`，逐条验证预测在 validation_predictions/，词表/IDF 摘要及矩阵摘要在 provenance/，调用证据在 fit_trace.json。另实际加载全部26个保存模型，核对其索引、配置、词表/IDF、独立训练子集重新fit_transform的原矩阵摘要、逐条预测及训练/验证分数；26个均通过，见 serialized_model_checks.csv。训练文本再次transform可能改变CSR列索引顺序及L2求和末位，因此使用稀疏差分检查其数值误差小于1e-12，而不冒称字节完全相同。',
           '- 原最佳验证模型的词表、IDF 与独立仅在训练子集拟合的结果完全相同。原 MLP 最佳轮模型的词表/IDF 也完全匹配训练子集；其 epoch16 的验证交叉熵与 macro-F1 重算匹配。',
           f'- MLP 代码：`main.py:{p["mlp_fit"]}` 始终 partial_fit(a,target.iloc[tr])；循环中训练和验证分别用 a/tr、b/va。原每一轮没有独立模型副本，不能事后逐轮动态证明；保存的最佳轮可实际核验，循环路径无验证参与拟合。', '',
           '## 3. 评分来源：通过；原先缺逐条预测，已恢复并核对', '',
           f'- 评分位置：`main.py:{p["validation_scores"]}`，真实验证 target.iloc[va] 对该 b 的预测，不使用训练预测。',
           '- 原产物只有最佳模型、混淆矩阵与错分记录，没有全量逐条验证预测。已从原 best_validation.joblib 恢复 1474 条预测，保存 `results/best_validation_predictions.csv`，随后重新从磁盘读入，通过 row_index 取原 CSV 的真实 target 评分。',
           f'- 实际 accuracy={score["accuracy"]:.15f}，macro-F1={score["macro_f1"]:.15f}，错分={score["errors"]}；与 locked_scheme.json、experiments.csv、93 条 misclassified.csv、confusion_matrix.csv 一致。',
           '- 全部 1474 条预测也与本次独立重放最优方案逐行相同。没有读取课程示例 predictions.csv 的标签来评分；它只在审计中计算文件哈希区分来源。', '',
           '## 4. 重复与头部：程序检查通过；存在评估关联局限', '',
           f'- 完全重复：有标签内部额外重复 {exact["labeled_duplicate_extra_rows"]} 条，测试内部 {exact["test_duplicate_extra_rows"]} 条，有标签/测试共享全文 {exact["labeled_test"]["shared_unique"]} 组。三个 train/validation 的共享全文均为 {", ".join(str(exact["splits"][str(z)]["shared_unique"]) for z in [42,21,84])}。',
           f'- 仅诊断的大小写+空白统一全文：有标签内部额外重复 {norm["labeled_duplicate_extra_rows"]}、测试内部 {norm["test_duplicate_extra_rows"]}、有标签/测试共享 {norm["labeled_test"]["shared_unique"]} 组；三个划分共享 {", ".join(str(norm["splits"][str(z)]["shared_unique"]) for z in [42,21,84])}。',
           f'- 去头部后大小写+空白统一正文：有标签内部额外重复 {bod["labeled_duplicate_extra_rows"]}，测试内部 {bod["test_duplicate_extra_rows"]}，有标签/测试共享 {bod["labeled_test"]["shared_unique"]} 组；三个划分共享 {", ".join(str(bod["splits"][str(z)]["shared_unique"]) for z in [42,21,84])}。这些只是诊断，不更改数据或预测行序。',
           f'- 排除空正文后，主划分实际非空正文共享 {bod["nonempty_splits"]["42"]["shared_unique"]} 组，涉及验证 {bod["nonempty_splits"]["42"]["right_rows"]} 行；有标签/测试非空正文共享 {bod["nonempty_labeled_test"]["shared_unique"]} 组。所有组、数据集内行号和正文片段见 normalized_body_duplicate_groups.csv，空正文单独标记。',
           '- 人工抽查非空正文重复：训练行4887与验证行2009同为标签4，正文为同一帖子；训练行4323与验证行7309正文也相同，但标签分别为2和7。这是具体内容重合及标签差异，不是共享普通词汇；未据此改标签。实际内容只保存在不公开的样例产物中。',
           f'- 消息 ID 引用关系：匹配到 {rel["exact_message_reference_links"]} 条有向引用，其中 {rel["cross_main_message_reference_links"]} 条跨主训练/验证。长引用行配对 {rel["long_quote_pairs"]} 对，其中 {rel["cross_main_long_quote_pairs"]} 对跨主划分；主划分共享规范化 Subject 组 {rel["shared_subject_groups_main"]}。',
           '- 长引用条件：至少一侧带引用符、规范化整行至少100字符且15词，排除出现于超过10文档的高频模板；普通词汇共享不算引用或泄露。文件：message_reference_links.csv、long_quote_pairs.csv、quote_pair_samples.json、shared_subject_groups.json。',
           '- 人工抽查长引用：行502与5532引用同一句问题；行1089与3204也共享长引用。两对均跨主划分，实际内容只在不公开的quote_pair_samples.json。消息ID精确链接为0不代表没有引用。',
           '- 这些关系说明随机文本划分可能把同一线程、引用或相近帖子分到两侧；不能把现有分数解释为全新作者/线程的泛化，也不能仅凭关联就断言程序泄露。未擅自换划分、删文本或用这些诊断继续调参。',
           f'- 头部解析：`main.py:{p["header"]}`，仅解析首段到第一个空行，失败原样保留；逐条检查成功文本正文与分隔后的原正文完全相同，source 模式保留 Subject，正文中的冒号行完整保留。',
           *[f'- {h["dataset"]}：成功 {h["parsed"]}/{h["rows"]}，失败原样保留 {h["failed_unchanged"]}；含正文冒号行且确认保留的文档 {h["colon_body_documents_preserved"]}。' for h in headers],
           '- 实际随机样例、含冒号正文和失败样例见 header_check_samples.json。正文签名/引用仍可能含来源字段，因此清洗不是彻底匿名化。', '',
           '- 人工抽查行1、3：去来源字段后Subject保留，正文中的发言者引用行、冒号及多层引用均保留，没有全局删除冒号正文。实际样例只保存在不公开产物中。', '',
           '## 5. 打乱训练标签负对照：已运行，不参与调参', '',
           '- 使用主划分及锁定配置；词表/IDF 独立只拟合训练文本，随机种子20261013只排列训练 target，保持类别计数，验证 target 不变。模型类型、C、特征和验证样本完全相同。',
           f'- 实际验证 accuracy={neg["accuracy"]:.9f}，macro-F1={neg["macro_f1"]:.9f}；训练标签改变 {neg["changed_training_labels"]}/5894。警告：{neg["warnings"]}。',
           '- 负对照训练标签、验证预测、模型和配置分别保存在 negative_control_training_labels.csv、negative_control_validation_predictions.csv、negative_control_model.joblib、negative_control.json；没有写入选型实验表或更新锁定方案。',
           '- 接近十分类随机水平时，说明没有观察到“打乱训练标签仍保持原高分”的明显流程异常，但不能单独证明没有全部形式的泄露或相关性偏差。', '',
           '## 6. 最终预测：通过', '',
           f'- 最终全标签重训位置：`main.py:{p["final_fit"]}` 和 `main.py:{p["final_model_fit"]}`，仅全部7368条有标签文本/target；测试仅在 `main.py:{p["final_transform"]}` transform，并 predict 后写出。全量拟合只在锁定后的预测阶段允许，绝不用于原验证指标。',
           '- 原 final_model.joblib 的实际模型参数、特征参数和清洗模式与 locked_scheme.json 一致；词表/IDF 与独立仅全部有标签文本拟合的结果完全一致。另仅用全部有标签文本及原 target 重训核查副本，系数和截距与原最终模型在1e-12绝对容差内一致；原模型未覆盖。',
           '- 从原最终模型重新生成预测，并与旧 results/final_predictions.csv 全部逐行相同；另对2457条测试逐条 transform/predict，确认与整批预测逐行相同。',
           '- 项目根目录 `prediction.csv`：2457行、单列、无表头、无索引，每行匹配单个0–9整数。每条测试的原始行索引/文本SHA256/预测保存于 prediction_order_manifest.csv。',
           '- 根目录 predictions.csv 是课程示例，原样保留；prediction.csv 才是本次核验后的最终文件。没有报告无标签测试准确率，也未用最终重训训练内指标替换原验证分数。',
           f'- 最终文件SHA256：`{final["sha256"]}`，具体核查见 final_prediction_checks.json。', '',
           '## 未解决问题和处置范围', '',
           '- 未发现需要重选模型的流程泄露或评分错位；26组重放与原指标一致，锁定方案不变。新增缓存守卫、模型/预测/来源清单补齐追溯缺口，保留旧结果。',
           '- 原始运行没有逐组模型或完整动态调用记录，因此不能倒推出历史进程所有行为；结论基于当前源码、原最佳/MLP/最终模型的实际核验和全26组一致重放。',
           '- 明确存在帖子/引用/主题关联的评估局限。若研究新线程泛化，应另做预声明的分组划分实验；本次保持用户指定的80/20分层索引，未擅自去重、换划分、重选方案或修改测试集。',
           '- draft_main.py 是老师示例，直接拟合全部有标签文本；若拿它再划分验证会产生泄露风险。当前 main.py 不导入该脚本、不读取其 predictions.csv 来评估或调参；README 已明确它不属于复现入口。',
           '- 没有穷尽语义级近重复、作者身份或所有长帖中间位置的引用，长行匹配只是可解释的诊断。负对照单次结果和源码检查也不是“绝对无泄露”的证明。', '',
           '- 审计首轮在词表哈希 JSON 序列化时遇到 numpy.int64 类型错误，已将词表索引显式转换为整数后重跑。失败日志和部分产物保留在 results/leakage_audit_run_failed_01/，不与成功审计混合。', '',
           '- 增补序列化模型核查时，首次把fit_transform与再次transform的原始CSR字节摘要当成必然相同，断言失败。实际检查发现列索引顺序不同且L2末位误差约1e-15；修正为独立训练fit_transform核对原摘要，并以稀疏差分验证再次transform的数学等价。该失败记录保存在 leakage_audit_run/failure_matrix_check.json；不涉及模型选型或数据泄露。', '',
           '机器可读汇总：`results/leakage_audit_run/summary.json`。完整日志：`results/leakage_audit_run/audit.log`。']
    (ROOT/'leakage_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def cache_guard_check(d,splits):
    tr,va=splits[42]; cached=main.features(d.text,tr,va,main.BASE)
    checks={}
    for name,tr2,va2,cfg in [('wrong_seed',*splits[21],main.BASE),('wrong_configuration',tr,va,dict(main.BASE,sublinear_tf=True))]:
        try: cached.check(d.text,tr2,va2,cfg)
        except ValueError as error: checks[name]=dict(rejected=True,message=str(error))
        else: raise AssertionError('缓存守卫未拒绝错误输入')
    try: cached.check(d.text.map(lambda s:main.clean(s,'body')),tr,va,main.BASE)
    except ValueError as error: checks['wrong_cleaning']=dict(rejected=True,message=str(error))
    else: raise AssertionError('缓存守卫未拒绝错误清洗')
    write_json(AUDIT/'cache_guard_checks.json',checks)
    return checks

def run(reuse):
    AUDIT.mkdir(exist_ok=reuse)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(AUDIT/'audit.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)])
    logging.info('COMMAND %s',sys.argv)
    d,t=main.load(); assert len(d)==7368 and len(t)==2457
    required=['train_data.csv','test_data_unlabeled.csv','results/experiments.csv','results/locked_scheme.json','results/best_validation.joblib','results/final_model.joblib','results/mlp_best.joblib']
    before={f:sha(f) for f in required}
    if Path('predictions.csv').is_file(): before['predictions.csv']=sha('predictions.csv')
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    assert sha('train_data.csv')==protocol['files']['train_data.csv']
    assert sha('test_data_unlabeled.csv')==protocol['files']['test_data_unlabeled.csv']
    with threadpool_limits(limits=2):
        splits,splitrows=split_check(d)
        comparisons=pd.read_csv(AUDIT/'replay_comparison.csv').to_dict('records') if reuse else replay(d,splits)
        assert len(comparisons)==26
        if max(r['max_metric_delta'] for r in comparisons)>1e-10:
            raise AssertionError('重放指标与原结果不一致，需保留现场进一步排查')
        serialized_checks=serialized_model_check(d,splits)
        score,a,b,vec=check_saved_models(d,t,splits)
        negative=json.loads((AUDIT/'negative_control.json').read_text(encoding='utf-8')) if reuse else negative_control(d,splits,vec,a,b)
        if negative['accuracy']>.2 or negative['macro_f1']>.2:
            raise AssertionError('负对照明显高于随机水平，需继续排查')
        duplicates=duplicate_diagnostics(d,t,splits)
        headers=header_check(d,t)
        final=final_check(d,t)
        guard=cache_guard_check(d,splits)
    for file,digest in before.items(): assert sha(file)==digest, '旧数据/结果被修改：'+file
    summary=dict(status='passed_with_relationship_limitations',splits=splitrows,replay=comparisons,best_scores=score,
                 negative_control=negative,duplicates=duplicates,headers=headers,final=final,cache_guard=guard,
                 serialized_model_checks=serialized_checks,
                 source_sha256=sha('main.py'),audit_script_sha256=sha('leakage_audit.py'),unchanged_files_sha256=before,
                 original_provenance_gap='No original full validation predictions or 26 per-experiment models; independently replayed and retained originals.')
    write_json(AUDIT/'summary.json',summary); make_report(summary)
    logging.info('AUDIT COMPLETE scores=%s negative=%s',score,{k:negative[k] for k in ['accuracy','macro_f1']})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-replay',action='store_true',help='保留且复用已完成重放和负对照，仅重新做产物诊断')
    args=parser.parse_args()
    try: run(args.reuse_replay)
    except Exception:
        if AUDIT.exists(): write_json(AUDIT/'failure.json',dict(command=sys.argv,traceback=traceback.format_exc()))
        raise
