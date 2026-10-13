"""提交前整理：只读历史预测并评分、导出公开证据；没有模型训练或调参。"""
import ast
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import main

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT/'results'
PUBLIC = ROOT/'public_results'
UPLOAD = ROOT/'github_upload'
CODE_FILES = ['main.py', 'verify_results.py', 'leakage_audit.py', 'pack_review.py', 'prepare_submission.py',
              'README.md', 'requirements.txt', '.gitignore']
NUMERIC_COLUMNS = ['seed','parameter','train_accuracy','train_macro_f1','val_accuracy','val_macro_f1',
                   'fit_seconds','feature_seconds','elapsed_seconds','feature_plus_fit_seconds','features',
                   'train_nnz','train_density','validation_density','retries']
METRIC_COLUMNS = ['epoch','fit_seconds','internal_loss','train_cross_entropy','train_accuracy','train_macro_f1',
                  'val_cross_entropy','val_accuracy','val_macro_f1','epoch_total_seconds']

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def check_dependencies():
    requirements={}
    for line in (ROOT/'requirements.txt').read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name,version=line.strip().split('=='); requirements[name]=version
    expected=set(main.CORE_DEPENDENCIES)|{'pypdf'}
    assert set(requirements)==expected
    installed={}; unavailable=[]
    for p in requirements:
        try: installed[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            if p!='pypdf': raise
            unavailable.append(p)
    assert all(installed[p]==requirements[p] for p in installed)
    modules=set()
    for name in CODE_FILES:
        if not name.endswith('.py'): continue
        for node in ast.walk(ast.parse((ROOT/name).read_text(encoding='utf-8'))):
            if isinstance(node,ast.Import): modules.update(n.name.split('.')[0] for n in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module: modules.add(node.module.split('.')[0])
    mapping={'numpy':'numpy','pandas':'pandas','sklearn':'scikit-learn','matplotlib':'matplotlib',
             'joblib':'joblib','threadpoolctl':'threadpoolctl','pypdf':'pypdf'}
    assert set(mapping.values()).issubset(requirements)
    assert set(mapping).issubset(modules)
    return dict(status='passed',installed_versions=installed,unavailable_optional_dependencies=unavailable,
                direct_external_imports=sorted(set(mapping)&modules),
                scipy='稀疏矩阵和scikit-learn所需依赖',pypdf='仅可选PDF记录')

def check_setup_without_training():
    """隔离目录只含两个CSV和源码，PDF缺失时能到达训练边界；在边界主动停止。"""
    folder=PRIVATE/'pre_submission_checks'; folder.mkdir(exist_ok=True)
    absent=folder/'pdf_missing'; absent.mkdir(exist_ok=True)
    for name in ['train_data.csv','test_data_unlabeled.csv','main.py']:
        shutil.copy2(ROOT/name,absent/name)
    (absent/'results').mkdir(exist_ok=True)
    class TrainingBoundaryReached(Exception): pass
    previous=Path.cwd(); previous_out=main.OUT
    try:
        os.chdir(absent); main.OUT=Path('results')
        # 不拟合特征或模型，不进入候选选择；只检验初始化能越过可选材料读取。
        with patch.object(main,'audit',return_value={}), patch.object(main,'features',side_effect=TrainingBoundaryReached):
            try: main.experiment(30)
            except TrainingBoundaryReached: pass
            else: raise AssertionError('未到达预期训练边界')
        missing=json.loads((absent/'results/assignment_record.json').read_text(encoding='utf-8'))
        assert missing['status']=='missing' and missing['message']=='未提供要求PDF'
        protocol=json.loads((absent/'results/protocol.json').read_text(encoding='utf-8'))
        assert set(protocol['files'])=={'train_data.csv','test_data_unlabeled.csv','main.py'}
        actual_version=importlib.metadata.version
        def optional_missing(name):
            if name=='pypdf': raise importlib.metadata.PackageNotFoundError(name)
            return actual_version(name)
        with patch.object(main.importlib.metadata,'version',side_effect=optional_missing):
            assert 'pypdf' not in main.dependency_versions()
        present=folder/'pdf_present'; present.mkdir(exist_ok=True)
        main.OUT=present
        if (ROOT/'实验一要求.pdf').is_file():
            provided=main.record_assignment(ROOT/'实验一要求.pdf')
            assert provided['status'] in {'provided','read_failed'}
            if provided['status']=='provided': assert provided['pages']>0
        else:
            provided=dict(status='not_available_for_present_branch_check')
    finally:
        main.OUT=previous_out; os.chdir(previous)
    return dict(status='passed',missing_pdf_message=missing['message'],only_two_course_csv_required=True,
                missing_optional_pypdf_not_fatal=True,present_pdf_status=provided['status'],training_performed=False)

def sensitivity():
    # 标记条件只读取有标签文本和主划分，不接触测试文本，也不读取标签来决定排除。
    d=pd.read_csv(ROOT/'train_data.csv')
    split=pd.read_csv(PRIVATE/'split_42.csv')
    assert len(d)==7368 and split.row_index.is_unique
    np.testing.assert_array_equal(split.row_index,np.arange(len(d)))
    np.testing.assert_array_equal(split.target,d.target)
    tr=split.loc[split.subset=='train','row_index'].to_numpy()
    va=split.loc[split.subset=='validation','row_index'].to_numpy()
    assert len(tr)==5894 and len(va)==1474 and not set(tr)&set(va)
    bodies=d.text.fillna('').astype(str).map(lambda s:re.sub(r'\s+',' ',main.clean(s,'body')).strip().lower())
    train_bodies=set(bodies.iloc[tr]); train_bodies.discard('')
    marked=np.asarray([i for i in va if bodies.iloc[i] and bodies.iloc[i] in train_bodies],dtype=int)
    assert len(marked)==5
    predictions=pd.read_csv(PRIVATE/'best_validation_predictions.csv')
    assert predictions.row_index.is_unique and set(predictions.row_index)==set(va)
    np.testing.assert_array_equal(predictions.true_label,d.target.iloc[predictions.row_index])
    assert set(predictions.predicted_label).issubset(set(range(10)))
    results=[]
    for name,frame in [('original_main_validation',predictions),
                       ('posthoc_excluding_body_overlap',predictions[~predictions.row_index.isin(marked)])]:
        results.append(dict(evaluation=name,samples=len(frame),accuracy=accuracy_score(frame.true_label,frame.predicted_label),
                            macro_f1=f1_score(frame.true_label,frame.predicted_label,labels=range(10),average='macro',zero_division=0),
                            errors=int((frame.true_label!=frame.predicted_label).sum()),excluded_samples=len(predictions)-len(frame)))
    locked=json.loads((PRIVATE/'locked_scheme.json').read_text(encoding='utf-8'))
    for metric in ['accuracy','macro_f1']:
        assert abs(results[0][metric]-locked['seed42_metrics']['val_'+metric])<1e-12
    pd.DataFrame({'row_index':marked,'reason':'nonempty_normalized_body_overlap_training'}).to_csv(PRIVATE/'scoring_sensitivity_marked.csv',index=False)
    pd.DataFrame(results).to_csv(PRIVATE/'scoring_sensitivity.csv',index=False)
    result=dict(status='posthoc_diagnostic_only',marking_uses='主训练/验证非空规范化正文，无测试数据或标签选取条件',
                excluded_validation_row_indices=marked.tolist(),metrics=results,model_retrained=False,
                main_metric_changed=False,selection_changed=False,limitation='不能排除全部引用、线程、作者或语义近重复偏差')
    write_json(PRIVATE/'scoring_sensitivity.json',result)
    return result

def png_text(path):
    data=path.read_bytes(); assert data[:8]==b'\x89PNG\r\n\x1a\n'
    offset=8; text=[]
    while offset+12<=len(data):
        length=struct.unpack('>I',data[offset:offset+4])[0]; kind=data[offset+4:offset+8]
        payload=data[offset+8:offset+8+length]
        if kind in {b'tEXt',b'iTXt'}: text.append(payload.decode('utf-8',errors='replace'))
        offset+=length+12
    return '\n'.join(text)

def privacy_check(folder):
    email=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    absolute=re.compile(r'(?:\b[A-Za-z]:[\\/]|/(?:Users|home|mnt|tmp)/)')
    forbidden_names={'train_data.csv','test_data_unlabeled.csv','prediction.csv','predictions.csv','实验一要求.pdf','Project1_review.zip'}
    files=sorted(p for p in folder.rglob('*') if p.is_file())
    for p in files:
        assert p.name not in forbidden_names
        assert p.suffix.lower() not in {'.joblib','.pkl','.pickle','.npz','.npy','.zip','.pdf','.pyc'}
        assert not any(part in {'results','.venv','__pycache__','.git'} for part in p.relative_to(folder).parts)
        text=png_text(p) if p.suffix=='.png' else p.read_text(encoding='utf-8')
        assert not email.search(text), '邮箱检测失败：'+p.name
        assert not absolute.search(text), '绝对路径检测失败：'+p.name
        if p.suffix=='.csv':
            columns=set(pd.read_csv(p,nrows=0).columns)
            denied={'text','text_snippet','original','source_removed','body_only','row_index','true_label','predicted_label','text_sha256'}
            if p.name=='confusion_matrix.csv':
                matrix=pd.read_csv(p)
                assert list(matrix.columns)==['true_label']+[str(i) for i in range(10)] and len(matrix)==10
                np.testing.assert_array_equal(matrix.true_label,np.arange(10))
                values=matrix.drop(columns='true_label').to_numpy()
                assert np.issubdtype(values.dtype,np.integer) and (values>=0).all()
                denied.remove('true_label')  # 此处是10类汇总矩阵的轴标签，不是逐样本原标签。
            assert not columns&denied
    return len(files)

def public_export(sens):
    PUBLIC.mkdir(exist_ok=True)
    e=pd.read_csv(PRIVATE/'experiments.csv')
    assert set(e.model).issubset({'MultinomialNB','LinearSVC','LogisticRegression'})
    assert set(e.variant).issubset({'base','sublinear','bigram','no_norm','source_removed','body_only'})
    assert set(e['group']).issubset({'baseline','control','robustness'}) and set(e['mode']).issubset({'full','source','body'})
    numeric=e[NUMERIC_COLUMNS].apply(pd.to_numeric,errors='raise')
    clean=pd.concat([e[['group','variant','model','mode']],numeric],axis=1)
    clean['had_warning']=e.warnings.fillna('').ne('')
    clean['final_converged']=e.final_converged
    clean[clean.seed==42].to_csv(PUBLIC/'experiments.csv',index=False)
    clean[(clean['group']=='robustness')|((clean.seed==42)&(clean.model=='LinearSVC')&(clean.parameter==10))].to_csv(PUBLIC/'three_split_results.csv',index=False)
    summary=pd.read_csv(PRIVATE/'robustness_summary.csv')
    assert len(summary)==6 and summary.val_macro_f1_count.eq(3).all()
    summary.to_csv(PUBLIC/'robustness_summary.csv',index=False)
    h=pd.read_csv(PRIVATE/'mlp_epochs.csv')[METRIC_COLUMNS].apply(pd.to_numeric,errors='raise')
    h.to_csv(PUBLIC/'mlp_epochs.csv',index=False)
    for filename in ['mlp_curves.png','confusion_matrix.png','confusion_matrix.csv','per_class_metrics.csv']:
        shutil.copy2(PRIVATE/filename,PUBLIC/filename)
    locked=json.loads((PRIVATE/'locked_scheme.json').read_text(encoding='utf-8'))
    # 配置中只有算法参数、数字指标；不公开警告原文和任何路径。
    published={key:locked[key] for key in ['model','parameter','features','mode','fixed_linear_model','fixed_linear_parameter']}
    published['seed42_metrics']={k:v for k,v in locked['seed42_metrics'].items() if k not in {'warnings','error'}}
    write_json(PUBLIC/'locked_scheme.json',published)
    env=json.loads((PRIVATE/'environment.json').read_text(encoding='utf-8'))
    write_json(PUBLIC/'environment.json',dict(result_origin='原Windows实际运行',python=env['python'].split()[0],
                platform=env['platform'],versions=env['versions'],thread_limit=env['thread_limit'],
                limitation='跨操作系统、版本、CPU/BLAS的数值、耗时和预测可能不同；不保证逐条完全一致'))
    pd.DataFrame(sens['metrics']).to_csv(PUBLIC/'scoring_sensitivity.csv',index=False)
    audit_file=PRIVATE/'leakage_audit_run/summary.json'
    audit=json.loads(audit_file.read_text(encoding='utf-8')) if audit_file.is_file() else None
    a,b=sens['metrics']
    lines=['# 原Windows证据与事后评分敏感性摘要','',
           '公开证据来自原Windows环境；提交前只整理和重新评分，没有训练、调参或改变锁定方案。', '',
           f'锁定方案：{locked["model"]}，C={locked["parameter"]}，完整文本、unigram、20000维、min_df=2、sublinear_tf=True、L2。',
           '主选型指标保持原验证macro-F1。三划分均值/样本标准差不是独立外部测试或置信区间。', '',
           '| 评价范围 | 样本数 | accuracy | macro-F1 | 错分数 |',
           '|---|---:|---:|---:|---:|',
           f'| 原主验证集 | {a["samples"]} | {a["accuracy"]:.9f} | {a["macro_f1"]:.9f} | {a["errors"]} |',
           f'| 事后排除5条正文重合样本 | {b["samples"]} | {b["accuracy"]:.9f} | {b["macro_f1"]:.9f} | {b["errors"]} |','',
           '标记只依据主训练文本与验证文本的非空规范化正文重合：谨慎去开头邮件头、统一大小写和空白。没有使用测试文本或标签决定标记；分数使用已保存最佳验证预测，classes为0–9。',
           '这是事后评分敏感性检查，不替换原主指标、不改变选型；不能排除全部引用、作者/线程关联或语义近重复偏差。正文清洗失败的文本原样保留。','']
    if audit:
        negative=audit['negative_control']; relation=audit['duplicates']['relationships']
        lines += ['历史审计：三个划分训练/验证行号交集均0，训练5894、验证1474，覆盖7368条；词表/IDF仅拟合训练子集，测试未参与调参。',
                  f'26组固定配置重放与旧指标最大差异{max(r["max_metric_delta"] for r in audit["replay"]):.3g}；26个保存模型核查通过。',
                  f'历史训练标签打乱负对照：accuracy={negative["accuracy"]:.9f}，macro-F1={negative["macro_f1"]:.9f}；本次没有重跑负对照。接近随机不能单独证明完全无泄露。',
                  f'全文完全重复及大小写/空白规范化全文重合为0；主划分非空正文重合5组，长引用匹配{relation["cross_main_long_quote_pairs"]}对。随机文本划分不等同于新线程或新作者泛化。']
    else:
        lines += ['未提供历史深入审计产物；不声称已重放模型或运行负对照。本次只完成保存预测的事后评分。']
    lines += ['', '原LinearSVC不归一化对照曾有收敛警告，提高迭代预算后收敛；公共表只保留警告标记，私有日志保留原文。',
              'MLP逐轮损失是固定该轮模型重新评估的平均交叉熵，总交叉熵=平均×样本数；内部loss_另列，不能混同。最佳验证macro-F1为第16轮，原模型副本在私有results。',
              '公开目录不含文本、邮件地址、姓名、原始数据、逐条预测、模型、本机绝对路径。其他环境可能有数值和预测差异。']
    (PUBLIC/'audit_sensitivity_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=sorted(p for p in PUBLIC.iterdir() if p.is_file() and p.name!='manifest.json')
    write_json(PUBLIC/'manifest.json',dict(origin='原Windows运行，提交前只整理',files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in files]))
    return privacy_check(PUBLIC)

def make_upload():
    UPLOAD.mkdir(exist_ok=True)
    for name in CODE_FILES: shutil.copy2(ROOT/name,UPLOAD/name)
    target=UPLOAD/'public_results'; target.mkdir(exist_ok=True)
    for p in PUBLIC.iterdir():
        if p.is_file(): shutil.copy2(p,target/p.name)
    names=sorted(p.relative_to(UPLOAD).as_posix() for p in UPLOAD.rglob('*') if p.is_file() and p.name!='UPLOAD_FILES.txt')
    names.append('UPLOAD_FILES.txt'); names.sort()
    (UPLOAD/'UPLOAD_FILES.txt').write_text('适合手动上传的文件清单（相对路径）\n'+'\n'.join(names)+'\n',encoding='utf-8')
    assert sorted(p.relative_to(UPLOAD).as_posix() for p in UPLOAD.rglob('*') if p.is_file())==names
    for name in CODE_FILES: assert digest(ROOT/name)==digest(UPLOAD/name)
    for p in PUBLIC.iterdir():
        if p.is_file(): assert digest(p)==digest(target/p.name)
    return privacy_check(UPLOAD),names

def run():
    os.chdir(ROOT)
    baseline_path=PRIVATE/'pre_submission_original_hashes.json'
    if not baseline_path.is_file():
        protected=['prediction.csv','results/locked_scheme.json','results/experiments.csv',
                   'results/robustness_summary.csv','results/mlp_epochs.csv','results/best_validation_predictions.csv']
        write_json(baseline_path,{name:digest(ROOT/name) for name in protected})
    baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    for name,value in baseline.items(): assert digest(ROOT/name)==value
    dependencies=check_dependencies()
    setup=check_setup_without_training()
    sens=sensitivity()
    public_count=public_export(sens)
    upload_count,names=make_upload()
    for name,value in baseline.items(): assert digest(ROOT/name)==value, '受保护文件改变：'+name
    ignore=(ROOT/'.gitignore').read_text(encoding='utf-8')
    for pattern in ['.venv/','results/','train_data.csv','test_data_unlabeled.csv','prediction.csv','*.joblib','*.zip','!public_results/']:
        assert pattern in ignore
    write_json(PRIVATE/'pre_submission_checks.json',dict(status='passed',dependencies=dependencies,setup=setup,
                original_hashes_unchanged=True,training_performed=False,tuning_performed=False,
                public_files=public_count,upload_files=upload_count,sensitivity_metrics=sens['metrics'],upload_listing=names))
    print('Preparation passed; no training or tuning.')
    print(pd.DataFrame(sens['metrics']).to_string(index=False))
    print('Public files:',public_count,'Upload files:',upload_count)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    run()
