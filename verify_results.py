"""复核已保存产物；不调参。运行 .venv 的 python verify_results.py。"""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
import main

with threadpool_limits(limits=2):
    d,t=main.load()
    out=Path('results')
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    for f,digest in protocol['files'].items():
        if f in {'draft_main.py','TUNING.md','实验一要求.pdf'} and not Path(f).is_file():
            continue  # 可选作业材料缺失不阻断数据及模型核验。
        if f!='main.py':
            assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest, f
    for seed in [42,21,84]:
        s=pd.read_csv(out/f'split_{seed}.csv')
        assert len(s)==len(d) and s.row_index.is_unique
        assert set(s.row_index)==set(range(len(d)))
        assert s.subset.eq('validation').sum()==1474
        assert s.subset.eq('train').sum()==5894
        np.testing.assert_array_equal(s.target,d.target)
    s=pd.read_csv(out/'split_42.csv')
    tr=s.loc[s.subset=='train','row_index'].to_numpy()
    va=s.loc[s.subset=='validation','row_index'].to_numpy()
    bundle=joblib.load(out/'best_validation.joblib')
    vec,model,mode=bundle['vectorizer'],bundle['model'],bundle['mode']
    texts=d.text.fillna('').astype(str).map(lambda x:main.clean(x,mode))
    # 独立仅在训练子集重新拟合，验证保存词表和 IDF 没有使用验证集。
    independent=TfidfVectorizer(**vec.get_params())
    independent.fit(texts.iloc[tr])
    assert independent.vocabulary_==vec.vocabulary_
    np.testing.assert_allclose(independent.idf_,vec.idf_)
    pred=model.predict(vec.transform(texts.iloc[va]))
    persisted=pd.read_csv(out/'best_validation_predictions.csv')
    np.testing.assert_array_equal(persisted.true_label,d.target.iloc[persisted.row_index])
    persisted_sorted=persisted.sort_values('row_index')
    np.testing.assert_array_equal(persisted_sorted.row_index,va)
    np.testing.assert_array_equal(persisted_sorted.predicted_label,pred)
    locked=json.loads((out/'locked_scheme.json').read_text(encoding='utf-8'))
    scores=main.metrics(d.target.iloc[va],pred)
    for k,v in scores.items():
        assert abs(v-locked['seed42_metrics']['val_'+k])<1e-12
    cm=pd.read_csv(out/'confusion_matrix.csv',index_col=0).to_numpy()
    assert cm.sum()==len(va)
    errors=pd.read_csv(out/'misclassified.csv')
    assert len(errors)==int((d.target.iloc[va].to_numpy()!=pred).sum())
    # 用已保存的最优轮模型重新计算，确认并非最后一轮模型冒充。
    mlp=joblib.load(out/'mlp_best.joblib')
    hist=pd.read_csv(out/'mlp_epochs.csv')
    assert mlp['epoch']==int(hist.loc[hist.val_macro_f1.idxmax(),'epoch'])
    assert len(hist)==protocol['mlp_epochs']
    probs=mlp['model'].predict_proba(mlp['vectorizer'].transform(d.text.iloc[va]))
    loss=log_loss(d.target.iloc[va],probs,labels=mlp['model'].classes_)
    expected=hist.loc[hist.epoch==mlp['epoch']].iloc[0]
    assert abs(loss-expected.val_cross_entropy)<1e-12
    e=pd.read_csv(out/'experiments.csv')
    assert len(e)==26 and e.status.eq('ok').all()
    predictions=pd.read_csv('prediction.csv',header=None)
    assert predictions.shape==(len(t),1)
    assert set(predictions[0]).issubset(set(range(10)))
    for name in ['mlp_curves.png','confusion_matrix.png']:
        assert (out/name).stat().st_size>1000
    # 以相同锁定模型重新生成错分诊断并增加词贡献证据。
    main.diagnostics(d,va,vec,model,pred,mode)
    main.save_json('verification.json',dict(status='passed',rows_labeled=len(d),rows_unlabeled=len(t),
                   experiment_count=len(e),mlp_epochs=len(hist),mlp_best_epoch=int(mlp['epoch']),
                   validation_scores=scores,misclassification_count=len(errors),
                   final_main_sha256=hashlib.sha256(Path('main.py').read_bytes()).hexdigest(),
                   checked=['raw data unchanged','split partitions','training-only vocabulary and IDF',
                            'saved validation model metrics','confusion matrix and error counts',
                            'MLP best epoch checkpoint and cross entropy','prediction format','PNG existence']))
    print('Verification passed:',scores,'MLP best epoch:',mlp['epoch'],'errors:',len(errors))
