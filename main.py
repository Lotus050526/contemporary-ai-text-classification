"""实验一：稀疏文本分类。所有路径相对于项目目录；运行 python main.py --help。"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
# 字体缓存放在项目内，避免 Windows 用户目录写入权限问题。
os.environ.setdefault('MPLCONFIGDIR', str(Path('results') / 'mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits, threadpool_info

OUT = Path('results')
BASE = dict(lowercase=True, max_features=20000, ngram_range=(1, 1), min_df=2,
            sublinear_tf=False, norm='l2', input='content', encoding='utf-8',
            decode_error='strict', strip_accents=None, analyzer='word',
            token_pattern=r'(?u)\b\w\w+\b', stop_words=None, max_df=1.0,
            binary=False, use_idf=True, smooth_idf=True, dtype=np.float64)
ROWS = []
CORE_DEPENDENCIES = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'threadpoolctl']

def dependency_versions():
    versions = {name: importlib.metadata.version(name) for name in CORE_DEPENDENCIES}
    try:
        versions['pypdf'] = importlib.metadata.version('pypdf')
    except importlib.metadata.PackageNotFoundError:
        pass  # PDF 工具是可选记录工具，不参与特征或模型计算。
    return versions

def input_file_hashes():
    names = ['train_data.csv', 'test_data_unlabeled.csv', 'main.py']
    names += [name for name in ['draft_main.py', 'TUNING.md', '实验一要求.pdf'] if Path(name).is_file()]
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}

def record_assignment(pdf_path=Path('实验一要求.pdf')):
    """要求 PDF 仅作为附加记录；缺失或读取失败均不阻断实验。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        record = dict(status='missing', message='未提供要求PDF')
        extracted = record['message']
        logging.info(record['message'])
    else:
        record = dict(status='provided', file=pdf_path.name, sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest())
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            extracted = '\n'.join(page.extract_text() or '' for page in reader.pages)
            record['pages'] = len(reader.pages)
            logging.info('已读取要求PDF：%s', pdf_path.name)
        except Exception as error:
            record.update(status='read_failed', message='要求PDF读取失败，继续实验', error=f'{type(error).__name__}: {error}')
            extracted = record['message']
            logging.warning('%s: %s', record['message'], error)
    (OUT/'assignment_extracted.txt').write_text(extracted+'\n', encoding='utf-8')
    save_json('assignment_record.json', record)
    return record

def text_digest(texts):
    digest = hashlib.sha256()
    for text in texts:
        data = str(text).encode('utf-8')
        digest.update(len(data).to_bytes(8, 'little'))
        digest.update(data)
    return digest.hexdigest()

def indices_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()

@dataclass
class PreparedFeatures:
    vectorizer: object
    train: object
    validation: object
    seconds: float
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_text_sha256: str
    validation_text_sha256: str

    def __iter__(self):
        return iter((self.vectorizer, self.train, self.validation, self.seconds))

    def check(self, texts, tr, va, config):
        if not np.array_equal(self.train_indices, tr) or not np.array_equal(self.validation_indices, va):
            raise ValueError('缓存特征的训练/验证索引或顺序与当前实验不一致')
        if self.train_text_sha256 != text_digest(texts.iloc[tr]) or self.validation_text_sha256 != text_digest(texts.iloc[va]):
            raise ValueError('缓存特征的清洗文本与当前实验不一致')
        if self.vectorizer.get_params() != TfidfVectorizer(**config).get_params():
            raise ValueError('缓存特征配置与当前实验不一致')

def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')

def header(text):
    """只接受从首行开始、以空行结束、含 From/Subject 的标准邮件头；失败原样保留。"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    head, sep, body = text.partition('\n\n')
    if not sep:
        return None
    fields = []
    for line in head.split('\n'):
        if line[:1] in (' ', '\t') and fields:
            fields[-1][1] += '\n' + line
        else:
            match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_.-]*):[ \t]*(.*)', line)
            if not match:
                return None
            fields.append([match[1].lower(), line])
    if not {'from', 'subject'}.issubset({k for k, _ in fields}):
        return None
    return fields, body

SOURCE = {'from', 'sender', 'reply-to', 'organization', 'organisation', 'nntp-posting-host',
          'nntp-posting-user', 'path', 'message-id', 'references', 'xref', 'return-path',
          'received', 'distribution', 'newsgroups', 'originator', 'x-originating-ip', 'apparently-to', 'to', 'cc', 'bcc',
          'article-i.d.', 'in-reply-to', 'nf-id', 'nf-from', 'supersedes', 'x-posted-from',
          'x-address', 'x-fax', 'x-phone', 'replyto', 'x-received', 'x-to', 'x-auth-user',
          'x-cc', 'x-apparently-to', 'x-ftn-to', 'x-xxmessage-id', 'x-altnet-id', 'x-gated-by',
          'rganization', 'oganization'}

def clean(text, mode):
    if mode == 'full':
        return text
    parsed = header(text)
    if parsed is None:
        return text
    fields, body = parsed
    if mode == 'body':
        return body
    return '\n'.join(v for k, v in fields if k not in SOURCE) + '\n\n' + body

def load():
    d = pd.read_csv('train_data.csv')
    t = pd.read_csv('test_data_unlabeled.csv')
    if list(d.columns) != ['text', 'target'] or list(t.columns) != ['text']:
        raise ValueError('CSV 字段不符合要求')
    if d.target.isna().any() or set(d.target.unique()) != set(range(10)):
        raise ValueError('标签应为完整的 0–9，无缺失')
    return d, t

def split(d, seed):
    tr, va = train_test_split(np.arange(len(d)), test_size=.2, stratify=d.target, random_state=seed)
    frame = pd.DataFrame({'row_index': np.arange(len(d)), 'target': d.target, 'subset': 'train'})
    frame.loc[va, 'subset'] = 'validation'
    frame.to_csv(OUT / f'split_{seed}.csv', index=False)
    return tr, va

def audit(d, t):
    report = {}
    for name, frame in [('labeled', d), ('unlabeled', t)]:
        texts = frame.text.fillna('').astype(str)
        lengths = texts.str.len()
        report[name] = dict(rows=len(frame), columns=list(frame.columns), missing=frame.isna().sum().to_dict(),
                            blank=int(texts.str.strip().eq('').sum()), length_chars=lengths.describe(percentiles=[.5, .9, .99]).to_dict(),
                            duplicate_extra_rows=int(texts.duplicated().sum()), duplicate_involved_rows=int(texts.duplicated(False).sum()),
                            actual_newline_documents=int(texts.str.contains('\n', regex=False).sum()),
                            literal_backslash_n_documents=int(texts.str.contains(r'\n', regex=False).sum()),
                            actual_tab_documents=int(texts.str.contains('\t', regex=False).sum()),
                            literal_backslash_t_documents=int(texts.str.contains(r'\t', regex=False).sum()),
                            header_success=int(sum(header(s) is not None for s in texts)))
        report[name]['header_success_rate'] = report[name]['header_success'] / len(frame)
        examples = list(range(min(3, len(frame))))
        for pattern in [r'\n', r'\t']:
            examples += texts[texts.str.contains(pattern, regex=False)].index[:3].tolist()
        save_json(f'{name}_escape_samples.json', [{'row_index': int(i), 'repr': repr(texts.iloc[i][:1600])} for i in sorted(set(examples))])
    report['labels'] = d.target.value_counts().sort_index().to_dict()
    texts = d.text.fillna('').astype(str)
    duplicates = d[texts.duplicated(False)].copy()
    duplicates.insert(0, 'row_index', duplicates.index)
    duplicates.to_csv(OUT / 'duplicate_texts.csv', index=False)
    report['duplicate_text_groups_with_conflicting_labels'] = int(d.assign(text=texts).groupby('text').target.nunique().gt(1).sum())
    report['labeled_test_exact_overlap_unique_texts'] = len(set(texts) & set(t.text.fillna('').astype(str)))
    fields = {}
    examples = []
    failures = []
    for i, s in enumerate(texts):
        p = header(s)
        if p:
            for k, _ in p[0]:
                fields[k] = fields.get(k, 0) + 1
            if len(examples) < 5:
                examples.append(dict(row_index=i, original=s[:2500], source_removed=clean(s, 'source')[:2500], body_only=clean(s, 'body')[:2500]))
        elif len(failures) < 5:
            failures.append(dict(row_index=i, original=s[:1800]))
    report['header_fields'] = fields
    report['source_fields_removed'] = sorted(SOURCE)
    save_json('header_samples.json', {'successful': examples, 'failed_unchanged': failures})
    save_json('data_audit.json', report)
    return report

def model_for(name, value):
    if name == 'MultinomialNB':
        return MultinomialNB(alpha=value, fit_prior=True)
    if name == 'LinearSVC':
        return LinearSVC(C=value, dual='auto', max_iter=20000, tol=1e-4, random_state=42)
    return LogisticRegression(C=value, solver='lbfgs', max_iter=2000, tol=1e-4, random_state=42)

def metrics(y, pred):
    return dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, labels=np.arange(10), average='macro', zero_division=0))

def features(texts, tr, va, config):
    if len(np.intersect1d(tr, va)) or len(np.unique(tr)) != len(tr) or len(np.unique(va)) != len(va):
        raise ValueError('训练/验证索引重复或交叉')
    start = time.perf_counter()
    vec = TfidfVectorizer(**config)
    a = vec.fit_transform(texts.iloc[tr])
    b = vec.transform(texts.iloc[va])
    return PreparedFeatures(vec, a, b, time.perf_counter() - start, np.asarray(tr).copy(), np.asarray(va).copy(),
                            text_digest(texts.iloc[tr]), text_digest(texts.iloc[va]))

def run_one(d, tr, va, seed, group, variant, name, value, config, mode='full', prepared=None):
    ident = f'{group}_{seed}_{variant}_{name}_{value}'
    logging.info('START %s', ident)
    start = time.perf_counter()
    try:
        texts = d.text.fillna('').astype(str).map(lambda s: clean(s, mode))
        prepared = prepared if prepared is not None else features(texts, tr, va, config)
        prepared.check(texts, tr, va, config)
        vec, a, b, fs = prepared
        model = model_for(name, value)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit_start = time.perf_counter()
            model.fit(a, d.target.iloc[tr])
            fit_seconds = time.perf_counter() - fit_start
        warning_list = [f'{w.category.__name__}: {w.message}' for w in caught]
        retries = 0
        final_converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        # 只提高求解预算，不改变统计模型的 C、损失或特征；保留首次警告。
        if not final_converged and hasattr(model, 'max_iter'):
            retries = 1
            old_limit = model.max_iter
            logging.warning('%s retry: max_iter %d -> %d; original warnings %s', ident, old_limit, old_limit*5, warning_list)
            model.set_params(max_iter=old_limit*5)
            with warnings.catch_warnings(record=True) as retry_warnings:
                warnings.simplefilter('always')
                retry_start = time.perf_counter()
                model.fit(a, d.target.iloc[tr])
                fit_seconds += time.perf_counter()-retry_start
            warning_list += [f'retry {w.category.__name__}: {w.message}' for w in retry_warnings]
            final_converged = not any(issubclass(w.category, ConvergenceWarning) for w in retry_warnings)
        for w in warning_list:
            logging.warning('%s: %s', ident, w)
        pred = model.predict(b)
        train_metrics = metrics(d.target.iloc[tr], model.predict(a))
        val_metrics = metrics(d.target.iloc[va], pred)
        row = dict(group=group, seed=seed, variant=variant, model=name, parameter=value, mode=mode,
                   train_accuracy=train_metrics['accuracy'], train_macro_f1=train_metrics['macro_f1'],
                   val_accuracy=val_metrics['accuracy'], val_macro_f1=val_metrics['macro_f1'],
                   fit_seconds=fit_seconds, feature_seconds=fs, elapsed_seconds=time.perf_counter()-start,
                   feature_plus_fit_seconds=fs+fit_seconds,
                   features=a.shape[1], train_nnz=a.nnz, train_density=a.nnz/(a.shape[0]*a.shape[1]),
                   validation_density=b.nnz/(b.shape[0]*b.shape[1]), iterations=str(getattr(model, 'n_iter_', 'NA')),
                   warnings=' | '.join(warning_list), retries=retries, final_converged=final_converged, status='ok')
        ROWS.append(row)
        pd.DataFrame(ROWS).to_csv(OUT / 'experiments.csv', index=False)
        save_json(f'configs/{ident}.json', {'features': vec.get_params(), 'model': model.get_params(), 'header_mode': mode,
                  'train_shape': a.shape, 'validation_shape': b.shape,
                  'train_indices_sha256': indices_digest(tr), 'validation_indices_sha256': indices_digest(va),
                  'train_text_sha256': prepared.train_text_sha256, 'validation_text_sha256': prepared.validation_text_sha256})
        (OUT/'validation_predictions').mkdir(exist_ok=True)
        pd.DataFrame({'row_index':va, 'true_label':d.target.iloc[va].to_numpy(), 'predicted_label':pred,
                      'text_sha256':[hashlib.sha256(s.encode('utf-8')).hexdigest() for s in d.text.fillna('').astype(str).iloc[va]]}).to_csv(
                          OUT/f'validation_predictions/{ident}.csv',index=False)
        (OUT/'models').mkdir(exist_ok=True)
        joblib.dump({'vectorizer':vec,'model':model,'mode':mode,'seed':seed,
                     'train_indices':np.asarray(tr),'validation_indices':np.asarray(va),
                     'train_text_sha256':prepared.train_text_sha256,'validation_text_sha256':prepared.validation_text_sha256},
                    OUT/f'models/{ident}.joblib')
        logging.info('DONE %s macro-F1=%.6f fit=%.2fs', ident, val_metrics['macro_f1'], fit_seconds)
        return row, vec, model, pred
    except Exception:
        logging.exception('FAILED %s', ident)
        ROWS.append(dict(group=group, seed=seed, variant=variant, model=name, parameter=value, status='failed', error=traceback.format_exc()))
        pd.DataFrame(ROWS).to_csv(OUT / 'experiments.csv', index=False)
        raise

def diagnostics(d, va, vec, model, pred, mode='full'):
    y = d.target.iloc[va]
    cm = confusion_matrix(y, pred, labels=np.arange(10))
    pd.DataFrame(cm, index=np.arange(10), columns=np.arange(10)).to_csv(OUT / 'confusion_matrix.csv', index_label='true_label')
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=8)
    ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted label', ylabel='True label')
    fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(OUT / 'confusion_matrix.png', dpi=160); plt.close(fig)
    pd.DataFrame(classification_report(y, pred, labels=np.arange(10), output_dict=True, zero_division=0)).T.to_csv(OUT / 'per_class_metrics.csv')
    errors = pd.DataFrame({'row_index': va, 'csv_record_number': va+2, 'true_label': y.to_numpy(), 'predicted_label': pred,
                           'text_snippet': d.text.iloc[va].astype(str).str.slice(0, 3000).to_numpy()})
    errors = errors[errors.true_label != errors.predicted_label].sort_values('row_index')
    errors.to_csv(OUT / 'misclassified.csv', index=False)
    chosen = errors.drop_duplicates(['true_label', 'predicted_label']).head(5)
    chosen.to_csv(OUT / 'representative_errors.csv', index=False)
    if not hasattr(model, 'coef_'):
        return
    words = vec.get_feature_names_out()
    rows = []
    for label, coef in zip(model.classes_, model.coef_):
        for direction, indices in [('positive', np.argsort(coef)[-15:][::-1]), ('negative', np.argsort(coef)[:15])]:
            rows += [dict(label=int(label), direction=direction, word=words[j], weight=float(coef[j])) for j in indices]
    pd.DataFrame(rows).to_csv(OUT / 'class_weight_words.csv', index=False)
    evidence = []
    for r in chosen.itertuples():
        sample = vec.transform([clean(str(d.text.iloc[r.row_index]), mode)])
        true_idx = np.flatnonzero(model.classes_ == r.true_label)[0]
        pred_idx = np.flatnonzero(model.classes_ == r.predicted_label)[0]
        difference = model.coef_[pred_idx] - model.coef_[true_idx]
        values = sample.data * difference[sample.indices]
        for order in np.argsort(values)[-10:][::-1]:
            evidence.append(dict(row_index=r.row_index, true_label=r.true_label, predicted_label=r.predicted_label,
                                 term=words[sample.indices[order]], predicted_minus_true_contribution=values[order]))
    pd.DataFrame(evidence).to_csv(OUT / 'error_term_contributions.csv', index=False)

def mlp_run(d, tr, va, epochs):
    cfg = dict(BASE, max_features=5000)
    vec, a, b, fs = features(d.text.fillna('').astype(str), tr, va, cfg)
    model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=.0001,
                          batch_size=128, learning_rate_init=.001, shuffle=True, random_state=42,
                          max_iter=1, early_stopping=False)
    save_json('mlp_config.json', dict(features=vec.get_params(), model=model.get_params(), max_epochs=epochs,
              selection='validation macro-F1; ties earliest epoch', probability_columns=list(range(10)),
              loss_definition='log_loss(labels=model.classes_): sample mean cross entropy; sum = mean * number of samples. Internal loss_ includes regularization and batch-time weights.', feature_seconds=fs))
    history = []; best = -1
    for epoch in range(1, epochs+1):
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.partial_fit(a, d.target.iloc[tr], classes=np.arange(10))
        fit_seconds = time.perf_counter()-start
        for w in caught:
            logging.warning('MLP epoch %d %s: %s', epoch, w.category.__name__, w.message)
        row = dict(epoch=epoch, fit_seconds=fit_seconds, internal_loss=float(model.loss_), warnings=' | '.join(str(w.message) for w in caught))
        for prefix, x, y in [('train', a, d.target.iloc[tr]), ('val', b, d.target.iloc[va])]:
            prob = model.predict_proba(x)
            row[prefix+'_cross_entropy'] = log_loss(y, prob, labels=model.classes_)
            row.update({prefix+'_'+k:v for k,v in metrics(y, model.classes_[prob.argmax(axis=1)]).items()})
        row['epoch_total_seconds'] = time.perf_counter()-start
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / 'mlp_epochs.csv', index=False)
        if row['val_macro_f1'] > best:
            best = row['val_macro_f1']; best_epoch = epoch
            joblib.dump({'vectorizer': vec, 'model': copy.deepcopy(model), 'epoch': epoch, 'classes': model.classes_}, OUT / 'mlp_best.joblib')
        logging.info('MLP epoch=%d val-F1=%.5f train-loss=%.5f val-loss=%.5f', epoch, row['val_macro_f1'], row['train_cross_entropy'], row['val_cross_entropy'])
    hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ['cross_entropy', 'accuracy', 'macro_f1']):
        for prefix in ['train', 'val']:
            ax.plot(hist.epoch, hist[prefix+'_'+key], label=prefix)
        ax.axvline(best_epoch, color='gray', linestyle='--', label='best epoch')
        ax.set(xlabel='Epoch', ylabel=key); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'mlp_curves.png', dpi=160); plt.close(fig)
    save_json('mlp_best_epoch.json', history[best_epoch-1])

def experiment(epochs):
    d, t = load()
    versions = dependency_versions()
    Path('requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n', encoding='utf-8')
    save_json('environment.json', dict(python=sys.version, platform=platform.platform(), processor=platform.processor(), versions=versions, threadpools=threadpool_info(), thread_limit=2))
    save_json('protocol.json', dict(selection_metric='validation macro-F1', split=dict(test_size=.2, stratify='target', random_state=42),
              robustness_seeds=[21,84], tie_break='first candidate in declared order', mlp_epochs=epochs,
              selection_scope='9 baseline candidates and 5 one-factor/header variants on seed 42; MLP is learning-curve illustration only',
              files=input_file_hashes(),
              hypotheses={'sublinear':'降低重复词频影响是否改善泛化？', 'bigram':'短语信息是否改善分类？', 'no_norm':'长度与尺度变化是否影响性能？', 'headers':'模型是否依赖来源字段？'}))
    record_assignment()
    audit(d, t)
    tr, va = split(d, 42)
    texts = d.text.fillna('').astype(str)
    prepared = features(texts, tr, va, BASE)
    candidates = []
    for name, values in [('MultinomialNB', [.1,.5,1.]), ('LinearSVC', [.1,1.,10.]), ('LogisticRegression', [.1,1.,10.])]:
        for v in values:
            candidates.append(run_one(d,tr,va,42,'baseline','base',name,v,BASE,prepared=prepared))
    linear = max([c for c in candidates if c[0]['model'] != 'MultinomialNB'], key=lambda c:c[0]['val_macro_f1'])
    name, value = linear[0]['model'], linear[0]['parameter']
    variants = [('base',BASE,'full'), ('sublinear',dict(BASE,sublinear_tf=True),'full'),
                ('bigram',dict(BASE,ngram_range=(1,2)),'full'), ('no_norm',dict(BASE,norm=None),'full'),
                ('source_removed',BASE,'source'), ('body_only',BASE,'body')]
    comparisons = [linear]
    for variant,cfg,mode in variants[1:]:
        comparisons.append(run_one(d,tr,va,42,'control',variant,name,value,cfg,mode))
    best = max(candidates+comparisons[1:], key=lambda c:c[0]['val_macro_f1'])
    best_row,vec,model,pred = best
    cfg = BASE if best_row['group']=='baseline' else next(c for v,c,m in variants if v==best_row['variant'])
    locked = dict(model=best_row['model'], parameter=best_row['parameter'], features=cfg, mode=best_row['mode'],
                  seed42_metrics=best_row, fixed_linear_model=name, fixed_linear_parameter=value)
    save_json('locked_scheme.json',locked)
    joblib.dump({'vectorizer':vec,'model':model,'mode':best_row['mode']},OUT/'best_validation.joblib')
    pd.DataFrame({'row_index':va,'true_label':d.target.iloc[va].to_numpy(),'predicted_label':pred}).to_csv(
        OUT/'best_validation_predictions.csv',index=False)
    # 类别权重只对线性模型导出；若 NB 最优，另保存最佳线性模型的权重。
    diagnostics(d,va,vec,model,pred,best_row['mode'])
    if not hasattr(model,'coef_'):
        words=linear[1].get_feature_names_out()
        pd.DataFrame([dict(label=int(label),word=words[j],weight=float(coef[j]))
                      for label,coef in zip(linear[2].classes_,linear[2].coef_)
                      for j in np.argsort(coef)[-15:][::-1]]).to_csv(OUT/'class_weight_words.csv',index=False)
    overlap = {}
    for seed in [42,21,84]:
        tr_s,va_s = (tr,va) if seed==42 else split(d,seed)
        common = set(texts.iloc[tr_s]) & set(texts.iloc[va_s])
        overlap[str(seed)] = dict(unique_shared_texts=len(common), validation_rows_with_training_duplicate=int(texts.iloc[va_s].isin(common).sum()))
        pd.DataFrame({'row_index':va_s,'has_training_duplicate':texts.iloc[va_s].isin(common).to_numpy()}).to_csv(OUT/f'duplicate_overlap_{seed}.csv',index=False)
        if seed != 42:
            for variant,c,m in variants:
                run_one(d,tr_s,va_s,seed,'robustness',variant,name,value,c,m)
            if best_row['model']!=name or best_row['parameter']!=value:
                run_one(d,tr_s,va_s,seed,'robustness','selected',best_row['model'],best_row['parameter'],cfg,best_row['mode'])
    save_json('split_duplicate_overlap.json',overlap)
    robust_rows = [c[0] for c in comparisons]+[r for r in ROWS if r['group']=='robustness']
    if best_row['model']!=name or best_row['parameter']!=value:
        robust_rows.append(dict(best_row,variant='selected'))
    summary = pd.DataFrame(robust_rows).groupby(['variant','model','parameter'])[['val_macro_f1','val_accuracy','fit_seconds']].agg(['mean','std','count'])
    summary.columns = ['_'.join(pair) for pair in summary.columns]
    summary.reset_index().to_csv(OUT/'robustness_summary.csv',index=False)
    mlp_run(d,tr,va,epochs)
    save_json('completion.json',dict(status='complete', note='No test accuracy; predictions require separate command. Raw CSV files unchanged.'))

def predict(output, config_path='results/locked_scheme.json'):
    """锁定配置后重拟合全部标签数据；无标签数据仅 transform。"""
    d,t = load()
    locked=json.loads(Path(config_path).read_text(encoding='utf-8'))
    cfg=locked['features']; cfg['dtype']=np.float64
    cfg['ngram_range']=tuple(cfg['ngram_range'])
    vec=TfidfVectorizer(**cfg)
    mode=locked['mode']
    a=vec.fit_transform(d.text.fillna('').astype(str).map(lambda s:clean(s,mode)))
    model=model_for(locked['model'],locked['parameter'])
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); model.fit(a,d.target)
    b=vec.transform(t.text.fillna('').astype(str).map(lambda s:clean(s,mode)))
    pd.DataFrame(model.predict(b)).to_csv(output,index=False,header=False)
    joblib.dump({'vectorizer':vec,'model':model,'mode':mode},OUT/'final_model.joblib')
    save_json('prediction_run.json',dict(output=output,rows=len(t),fit_and_predict_seconds=time.perf_counter()-start,
              fit_rows=len(d), features=a.shape[1],warnings=[str(w.message) for w in caught],test_accuracy=None))
    logging.info('Saved %d predictions to %s',len(t),output)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['experiment','predict'])
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--output',default='prediction.csv')
    parser.add_argument('--config',default='results/locked_scheme.json',help='predict 使用的锁定配置；可指定 public_results/locked_scheme.json')
    args=parser.parse_args()
    if not 1<=args.epochs<=50:
        parser.error('--epochs 必须为 1–50')
    OUT.mkdir(exist_ok=True); (OUT/'configs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)])
    logging.info('COMMAND %s',sys.argv)
    with threadpool_limits(limits=2):
        try:
            experiment(args.epochs) if args.command=='experiment' else predict(args.output,args.config)
        except Exception:
            logging.exception('RUN FAILED')
            save_json('failure.json',dict(command=sys.argv,traceback=traceback.format_exc()))
            raise

if __name__=='__main__':
    main()
