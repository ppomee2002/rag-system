import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from simulator import Job, Call, job_from_row, attach_oof_predictions, fit_cost_model, simulate, summarize, poisson_arrivals
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def synthetic_jobs():
    rng=np.random.default_rng(42)
    jobs=[]
    for i in range(80):
        corrected=i%5==0
        phases=[('delay',.02)]+[('gpu',[Call('grade',j,100+j*30,float(rng.uniform(.5,1.5)))]) for j in range(5)]
        if corrected:phases += [('gpu',[Call('rewrite',0,100,2.)]),('delay',.5)]
        phases += [('gpu',[Call('generate',0,1000 if corrected else 500,float(rng.uniform(6,10) if corrected else rng.uniform(1,3)))])]
        work=sum(c.service for k,cs in phases if k=='gpu' for c in cs)
        solo=work+sum(v for k,v in phases if k=='delay')
        jobs.append(Job(str(i),'incorrect' if corrected else 'correct',np.array([1.,rng.random(),rng.random(),rng.random(),.1,1.]),phases,work,solo,n_docs=5))
    return jobs

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--demo',action='store_true')
    p.add_argument('--profile',type=Path)
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if not args.demo and args.profile is None:p.error('Specify --profile for real trace replay; --demo is SYNTHETIC validation only')
    jobs=synthetic_jobs() if args.demo else [job_from_row(json.loads(l)) for l in args.profile.read_text(encoding='utf-8-sig').splitlines() if l.strip() and json.loads(l)['pipeline']=='crag']
    if len(jobs)<10:p.error('At least 10 query traces required')
    attach_oof_predictions(jobs)
    model=fit_cost_model(jobs)
    mean_work=float(np.mean([j.work for j in jobs]))
    results=[]
    policies=[('fifo_call',0),('sjf_pred',0),('srpt_est',0)]+[('srpt_est',a) for a in cfg['aging_alphas']]
    for load in cfg['loads']:
        for seed in cfg['seeds']:
            rng=np.random.default_rng(seed)
            sample=[jobs[i] for i in rng.integers(0,len(jobs),cfg['n_queries'])]
            arrivals=poisson_arrivals(len(sample),load/mean_work,rng)
            for policy,alpha in policies:
                runs=simulate(sample,arrivals,policy,model,aging_alpha=alpha)
                assert all(np.isfinite(r.finish) and r.finish>=r.arrival for r in runs)
                results.append({'mode':'SYNTHETIC_VALIDATION' if args.demo else 'REAL_TRACE_SIMULATION','load':load,'seed':seed,'policy':policy,'aging_alpha':alpha,**summarize(runs,cfg['warmup_frac'])})
    out=ROOT/'results/scheduling_optimization'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
    with (out/'sim_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    lines=['# Scheduling comparison', '', 'SYNTHETIC VALIDATION ONLY; not PopQA or measured GPU results.' if args.demo else 'Trace replay simulation; not live vLLM validation.', '', '| load | policy | aging alpha | mean E2E | correction mean | correction p95 |','|---|---|---|---|---|---|']
    for load in cfg['loads']:
        for policy,alpha in policies:
            subset=[r for r in results if r['load']==load and r['policy']==policy and r['aging_alpha']==alpha]
            means=[np.nanmean([r[m] for r in subset]) for m in ['mean','mean_corrected','p95_corrected']]
            lines.append(f'| {load} | {policy} | {alpha} | '+ ' | '.join(f'{v:.3f}' for v in means)+' |')
    lines+=['','Age = cumulative GPU queue waiting seconds, including prior stages. Priority = baseline estimated remaining work - alpha * age. Queue priorities refreshed at every dispatch. Existing correction-aware SRPT is baseline. Calls are nonpreemptive.','Calibrate alpha on separate data, not final evaluation queries. No live vLLM scheduling integration implemented in this simulator.','Call cost model follows the inherited baseline and is fitted to the trace pool; query-level predictions are out-of-fold. This is trace replay, not validation on unseen hardware.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print('Saved:',out)



def threshold_study():
    """Four-arm replay with calibration seeds separated from evaluation seeds."""
    import hashlib
    import shutil
    from datetime import datetime
    from zoneinfo import ZoneInfo

    p = argparse.ArgumentParser()
    p.add_argument('--threshold-study', action='store_true')
    p.add_argument('--config', type=Path, default=HERE/'config.local.json')
    p.add_argument('--records', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    study = cfg['threshold_study']
    thresholds = study['threshold_candidates_seconds']
    calibration_seeds = study['calibration_seeds']
    if set(calibration_seeds) & set(cfg['seeds']):
        p.error('Calibration and evaluation seeds must be disjoint')
    if not thresholds or any(not np.isfinite(t) or t < 0 for t in thresholds):
        p.error('Threshold candidates must be finite, nonnegative seconds')
    if cfg['n_queries'] < 1 or not 0 <= cfg['warmup_frac'] < 1:
        p.error('Invalid sample size or warmup fraction')
    raw = args.records.read_bytes()
    records = [json.loads(l) for l in raw.decode('utf-8-sig').splitlines() if l.strip()]
    variants = ['full', study['grading_variant']]
    pools, audits, models = {}, {}, {}
    for variant in variants:
        rows = [r for r in records if r['variant'] == variant and r['repeat'] == study['trace_repeat']]
        by_id = {r['query_id']: r for r in rows}
        if len(by_id) != len(rows) or len(rows) < 10:
            p.error('Need >=10 unique, nonduplicated question traces per variant')
        pools[variant] = by_id
    ids = sorted(pools['full'])
    if set(ids) != set(pools[variants[1]]):
        p.error('Four-arm comparison requires identical question IDs; no silent dropping')
    for variant in variants:
        rows = [pools[variant][qid] for qid in ids]
        for row in rows:
            if not row['calls'] or any(not np.isfinite(c['server_total']) or c['server_total'] <= 0 for c in row['calls']):
                p.error('Trace requires positive measured server_total for every call')
        jobs = [job_from_row(r) for r in rows]
        if variant != 'full':
            # The early-stop position is unknown at retrieval time. Do not expose
            # the future observed grading-call count to the SRPT estimate.
            for job, row in zip(jobs, rows, strict=True):
                job.n_docs = len(row['retrieval_chars'])
        attach_oof_predictions(jobs)
        models[variant] = fit_cost_model(jobs)
        pools[variant] = jobs
        audits[variant] = {
            'unique_questions': len(jobs),
            'unique_correction_questions': sum(j.action != 'correct' for j in jobs),
            'historical_accuracy': float(np.mean([r['acc'] for r in rows])),
            'mean_server_work_seconds': float(np.mean([j.work for j in jobs])),
            'source_variant': variant, 'source_repeat': study['trace_repeat'],
            'model': models[variant].to_dict(),
            'prediction_scope': 'query predictions: 5-fold OOF seed 0; call model: full variant pool (in-pool replay, not unseen-question validation)',
        }
    # A common arrival rate on Full workload preserves offered traffic across arms.
    mean_full_work = audits['full']['mean_server_work_seconds']
    out = ROOT/'results/scheduling_optimization'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_threshold')
    out.mkdir(parents=True, exist_ok=False)
    (out/'source_snapshot').mkdir()
    for name in ['run.py', 'simulator.py', 'verify.py', 'plot.py', 'config.local.json']:
        shutil.copy2(HERE/name, out/'source_snapshot'/name)
    (out/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    metadata = {'source': str(args.records.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'variants': audits, 'question_ids': ids,
                'accuracy_scope': 'Historical source scores only; replay creates no new answers or accuracy measurements',
                'load_basis': 'arrival_rate = rho / mean Full server work; same IDs and arrivals across all arms',
                'calibration_scope': 'disjoint arrival seeds, SAME original question pool; not an independent question test set',
                'selection_rule': study['selection_rule'], 'selection_status': 'pending'}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out/'workloads').mkdir()
    results = []
    alpha = study['aging_alpha']

    def replay(stage, loads, seeds, arms, selected_t=None):
        for load in loads:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                indices = rng.integers(0, len(ids), cfg['n_queries'])
                arrivals = poisson_arrivals(len(indices), load/mean_full_work, rng)
                workload_hash = hashlib.sha256(indices.tobytes()+np.asarray(arrivals).tobytes()).hexdigest()
                np.savez_compressed(out/'workloads'/f'{stage}_rho{load}_seed{seed}.npz', indices=indices, arrivals=arrivals)
                for arm, variant, policy, aging, threshold in arms:
                    sample = [pools[variant][i] for i in indices]
                    runs = simulate(sample, arrivals, policy, models[variant], aging_alpha=aging,
                                    wait_threshold=threshold)
                    assert all(np.isfinite(r.finish) and r.finish >= r.arrival for r in runs)
                    assert all(abs(r.finish-r.arrival-r.job.solo-r.gpu_wait) < 1e-6 for r in runs)
                    monitor_t = threshold if threshold is not None else selected_t
                    results.append({'stage': stage, 'arm': arm, 'variant': variant, 'load': load,
                                    'seed': seed, 'policy': policy, 'aging_alpha': aging,
                                    'threshold_seconds': threshold, 'metric_threshold_seconds': monitor_t,
                                    'workload_sha256': workload_hash,
                                    'historical_source_accuracy': audits[variant]['historical_accuracy'],
                                    'meets_historical_full_accuracy': audits[variant]['historical_accuracy'] >= audits['full']['historical_accuracy'],
                                    **summarize(runs, cfg['warmup_frac'], monitor_t)})
                print(f'{stage}: rho={load}, seed={seed}, {len(arms)} arms complete', flush=True)

    calibration_arms = [('Scheduling Only', 'full', 'srpt_est', 0, None),
                        ('Scheduling Only', 'full', 'srpt_est', alpha, None)]
    calibration_arms += [('Scheduling Only', 'full', 'srpt_threshold', 0, t) for t in thresholds]
    replay('calibration', study['calibration_loads'], calibration_seeds, calibration_arms)
    # Predeclared equal weight of mean latency and Correction p95, normalized
    # within each identical workload to plain SRPT. Evaluation never selects T.
    scores = {}
    for t in thresholds:
        ratios = []
        for row in results:
            if row['policy'] != 'srpt_threshold' or row['threshold_seconds'] != t:
                continue
            base = next(r for r in results if r['load'] == row['load'] and r['seed'] == row['seed']
                        and r['policy'] == 'srpt_est' and r['aging_alpha'] == 0)
            ratios.append(0.5*row['mean']/base['mean'] + 0.5*row['p95_corrected']/base['p95_corrected'])
        scores[t] = float(np.mean(ratios))
    if not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('Calibration requires finite nonzero Correction p95; no usable threshold selected')
    selected_t = min(thresholds, key=lambda t: (scores[t], t))
    metadata.update(selected_threshold_seconds=selected_t, calibration_scores=scores, selection_status='frozen_before_evaluation')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Frozen T={selected_t}s from calibration only: {scores}', flush=True)
    eval_arms = [('Baseline', 'full', 'fifo_call', 0, None),
                 ('Grading Only', variants[1], 'fifo_call', 0, None)]
    for arm, variant in [('Scheduling Only', 'full'), ('Combined', variants[1])]:
        eval_arms += [(arm, variant, 'srpt_est', 0, None), (arm, variant, 'srpt_est', alpha, None),
                     (arm, variant, 'srpt_threshold', 0, selected_t)]
    replay('evaluation', cfg['loads'], cfg['seeds'], eval_arms, selected_t)
    with (out/'sim_results.csv').open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    metrics = ['mean', 'p95', 'mean_corrected', 'p95_corrected', 'mean_gpu_wait', 'max_gpu_wait', 'threshold_exceed_fraction', 'throughput']
    lines = ['# 대기 임계값 SRPT Trace 실험', '',
             f'선택 T: {selected_t}초. Aging α={alpha}. 선택용 시드={calibration_seeds}, 평가용 시드={cfg["seeds"]}.',
             'T는 응답 완료 보장이 아니라, 다음 호출 선택 시 누적 큐 대기가 T를 초과한 요청을 우선하는 기준입니다.',
             '', '## 평가 결과 (시드별 통계의 평균)', '',
             '| rho | 구성 | 정책 | 전체 평균 | 전체 p95 | Correction 평균 | Correction p95 | 평균 대기 | 최대 대기* | T 초과 비율 | 처리량(req/s) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary = []
    for load in cfg['loads']:
        for arm, variant, policy, aging, threshold in eval_arms:
            group = [r for r in results if r['stage'] == 'evaluation' and r['load'] == load and r['arm'] == arm
                     and r['policy'] == policy and r['aging_alpha'] == aging]
            label = policy + (f' alpha={aging}' if aging else '') + (f' T={threshold}' if threshold is not None else '')
            agg = {m: float(np.mean([r[m] for r in group])) for m in metrics}
            summary.append({'load': load, 'arm': arm, 'variant': variant, 'label': label, **agg,
                            'worst_seed_max_wait': max(r['max_gpu_wait'] for r in group)})
            lines.append(f'| {load} | {arm} | {label} | '+' | '.join(f'{agg[m]:.4f}' for m in metrics)+' |')
    with (out/'evaluation_summary.csv').open('x', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    lines += ['', '*최대 대기 열은 시드별 최대값의 평균입니다. 모든 시드 중 최대값은 evaluation_summary.csv의 worst_seed_max_wait에 있습니다.',
              'p95도 시드별 p95의 평균이며 전체 요청을 합친 p95가 아닙니다.',
              '', '## 해석과 제한', '',
              '- Full 원본 50문항, Correction 5개. 각 조건 요청 3,000개는 반복 재사용이며, 초기 300개 제외 후 2,700개를 집계합니다.',
              '- Full 원본 정확도 84%, Lazy-hit 원본 정확도 80%. Baseline/Scheduling Only가 주 평가입니다. Grading Only/Combined는 정확도 유지 기준 미충족 참고 결과입니다.',
              '- 정확도는 원본 기록의 과거 측정값입니다. 재생 중 새 답변 생성, API 호출, GPU 사용률 측정은 없습니다.',
              '- Full 서버 작업량 기준으로 동일한 도착률·질문 추출 인덱스를 모든 정책과 4종 구성에 공유했습니다. Lazy-hit의 실제 작업 부하는 더 작습니다.',
              '- T 초과 비율은 제외 구간 이후 요청 중 최종 누적 큐 대기시간 > 선택 T인 비율입니다. 실행시간·검색 지연은 대기에 포함하지 않습니다.',
              '- 처리량은 집계 요청 수 / (해당 요청의 마지막 완료 시각 - 최초 도착 시각)이며, 유한 Trace의 drain 구간을 포함합니다.',
              '- 선택 조건과 평가 도착 시드는 분리했지만 원본 질문 풀은 같습니다. 새 질문에 대한 일반화·통계적 유의성을 입증하지 않습니다.',
              '- 호출 비용 모델은 각 variant 전체 Trace로 학습하며 질문 전체/후반 작업량 예측만 5-fold OOF입니다. 기존 학습 방식을 보존한 in-pool 비교입니다.',
              '- Lazy-hit 스케줄링에서 미래 중단 위치를 알지 못하도록 잔여 grading 추정에 전체 검색 문서 수를 사용합니다.',
              '- Baseline은 현재 코드의 fifo_call로 정의했습니다. 다른 팀원 환경의 통합 실험과 동일한 구현이라는 뜻은 아닙니다.',
              '', '## Lazy-rest 기존 결과와 미실행 항목', '',
              '현재 원본 기록에서 Full → Lazy-rest: 평균 grading 호출 5.00 → 1.52, 평균 생성 단계 6.39 → 21.23초, 전체 평균 응답 24.55 → 27.07초, 정확도 84% → 82%.',
              '첫 통과 문서 + 미평가 문서 2개/3개로 제한하는 재측정은 아직 수행하지 않았습니다. 이는 생성 입력을 바꾸므로 LLM 새 실행과 정확도 재평가가 필요하며 이번 SRPT 실험에서 결과를 추정하지 않습니다.',
              '', '## 재현 자료', '',
              'config.json, metadata.json(Trace SHA256·학습 정보·T 선택 점수), source_snapshot/, workloads/*.npz(인덱스·도착 시각), sim_results.csv를 함께 보관합니다.']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved:', out, flush=True)



def rescue_study():
    """Mean-first, one-call rescue; no model-server/API calls."""
    import hashlib
    import shutil
    from datetime import datetime
    from zoneinfo import ZoneInfo

    p = argparse.ArgumentParser()
    p.add_argument('--rescue-study', action='store_true')
    p.add_argument('--config', type=Path, default=HERE/'config.local.json')
    p.add_argument('--records', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    study = cfg['rescue_study']
    cal_seeds, eval_seeds = study['calibration_seeds'], study['evaluation_seeds']
    if set(cal_seeds) & set(eval_seeds):
        p.error('Calibration/evaluation seeds must be disjoint')
    percentiles = study['threshold_percentiles']
    if not percentiles or any(not 0 < q < 100 for q in percentiles):
        p.error('Threshold percentiles must be between 0 and 100')
    if cfg['n_queries'] < 1 or not 0 <= cfg['warmup_frac'] < 1:
        p.error('Invalid sample size or warmup fraction')
    source = args.records.read_bytes()
    records = [json.loads(l) for l in source.decode('utf-8-sig').splitlines() if l.strip()]
    pools, models, audit = {}, {}, {}
    full_ids = None
    for variant in ['full', 'lazy_hit']:
        rows = [r for r in records if r['variant'] == variant and r['repeat'] == study['trace_repeat']]
        by_id = {r['query_id']: r for r in rows}
        if len(by_id) != len(rows) or len(rows) < 10:
            p.error('Need >=10 unique question traces per variant')
        ids = sorted(by_id)
        if full_ids is not None and ids != full_ids:
            p.error('Full and Lazy-hit must have identical question IDs')
        full_ids = ids
        rows = [by_id[qid] for qid in ids]
        if any(not r['calls'] or any(not np.isfinite(c['server_total']) or c['server_total'] <= 0 for c in r['calls']) for r in rows):
            p.error('Every call needs a positive finite measured server duration')
        jobs = [job_from_row(r) for r in rows]
        if variant == 'lazy_hit':
            for job, row in zip(jobs, rows, strict=True):
                job.n_docs = len(row['retrieval_chars'])
        attach_oof_predictions(jobs)
        pools[variant], models[variant] = jobs, fit_cost_model(jobs)
        audit[variant] = {'n_original_questions': len(jobs),
                          'n_original_correction': sum(j.action != 'correct' for j in jobs),
                          'historical_accuracy': float(np.mean([r['acc'] for r in rows])),
                          'mean_work': float(np.mean([j.work for j in jobs])),
                          'model': models[variant].to_dict()}
    out = ROOT/'results/scheduling_optimization'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_rescue')
    out.mkdir(parents=True, exist_ok=False)
    for folder in ['source_snapshot', 'workloads']:
        (out/folder).mkdir()
    for name in ['run.py', 'simulator.py', 'verify.py', 'plot.py', 'config.local.json', 'README.md']:
        shutil.copy2(HERE/name, out/'source_snapshot'/name)
    (out/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    metadata = {'source_path': str(args.records.resolve()), 'source_sha256': hashlib.sha256(source).hexdigest(),
                'question_ids': ids, 'variants': audit, 'selection_rule': study['selection_rule'],
                'prediction_scope': 'query-level 5-fold OOF seed 0; call model fit on complete variant pool; same original question pool in calibration/evaluation',
                'load_basis': 'Full mean server work; equal question samples and arrival timestamps across all arms',
                'threshold_scope': 'Single global T derived from equally sized calibration workload pools; not an online load estimator',
                'rescue_semantics': 'eligible if cumulative queue wait minus queue wait at last rescue > T; dispatch one nonpreemptive call, then reset only rescue eligibility counter; total queue wait unchanged',
                'selection_status': 'not_yet_selected'}
    results, cache, waits = [], {}, []
    warmup_n = int(cfg['n_queries']*cfg['warmup_frac'])

    def workload(stage, load, seed):
        key = (stage, load, seed)
        if key not in cache:
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(ids), cfg['n_queries'])
            arrivals = poisson_arrivals(len(indices), load/audit['full']['mean_work'], rng)
            digest = hashlib.sha256(indices.tobytes()+np.asarray(arrivals).tobytes()).hexdigest()
            cache[key] = indices, arrivals, digest
            np.savez_compressed(out/'workloads'/f'{stage}_rho{load}_seed{seed}.npz', indices=indices, arrivals=arrivals)
        return cache[key]

    def execute(stage, load, seed, arm, variant, policy, alpha=0, threshold=None, monitor_t=None):
        indices, arrivals, digest = workload(stage, load, seed)
        stats = {}
        runs = simulate([pools[variant][i] for i in indices], arrivals, policy, models[variant],
                        aging_alpha=alpha, wait_threshold=threshold, dispatch_stats=stats)
        assert all(np.isfinite(r.finish) and r.finish >= r.arrival for r in runs)
        assert all(abs(r.finish-r.arrival-r.job.solo-r.gpu_wait) < 1e-6 for r in runs)
        retained = runs[warmup_n:]
        row = {'stage': stage, 'load': load, 'seed': seed, 'arm': arm, 'variant': variant,
               'policy': policy, 'aging_alpha': alpha, 'threshold_seconds': threshold,
               'metric_threshold_seconds': monitor_t, 'workload_sha256': digest,
               **summarize(runs, cfg['warmup_frac'], monitor_t),
               'p99_gpu_wait': float(np.percentile([r.gpu_wait for r in retained], 99)),
               'rescue_request_fraction': float(np.mean([r.rescue_count > 0 for r in retained])),
               'mean_rescues_per_request': float(np.mean([r.rescue_count for r in retained])),
               **{k: v for k, v in stats.items() if k != 'current_consecutive_overrides'},
               'override_fraction_all_dispatches': stats['srpt_override_dispatches']/stats['total_dispatches']}
        results.append(row)
        return runs

    # Baseline runs both create candidate thresholds and provide paired references.
    for load in cfg['loads']:
        for seed in cal_seeds:
            runs = execute('calibration', load, seed, 'Scheduling Only', 'full', 'srpt_est')
            waits.extend(r.gpu_wait for r in runs[warmup_n:])
        print(f'Candidate distribution: rho={load} complete', flush=True)
    thresholds_by_percentile = {str(q): float(np.ceil(np.percentile(waits, q))) for q in percentiles}
    candidates = sorted(set(thresholds_by_percentile.values()))
    metadata['thresholds_by_percentile'] = thresholds_by_percentile
    alpha, old_t = study['aging_alpha'], study['old_threshold_seconds']
    for load in cfg['loads']:
        for seed in cal_seeds:
            execute('calibration', load, seed, 'Scheduling Only', 'full', 'srpt_est', alpha)
            execute('calibration', load, seed, 'Scheduling Only', 'full', 'srpt_threshold', threshold=old_t, monitor_t=old_t)
            for threshold in candidates:
                execute('calibration', load, seed, 'Scheduling Only', 'full', 'srpt_rescue', threshold=threshold, monitor_t=threshold)
            print(f'Calibration: rho={load} seed={seed} complete', flush=True)
    base = {(r['load'], r['seed']): r for r in results if r['policy'] == 'srpt_est' and r['aging_alpha'] == 0}
    scores = {t: float(np.mean([r['mean']/base[r['load'], r['seed']]['mean'] for r in results
                               if r['policy'] == 'srpt_rescue' and r['threshold_seconds'] == t])) for t in candidates}
    finite_t = min(candidates, key=lambda t: (scores[t], -t))
    winner = 'srpt_rescue' if scores[finite_t] < 1.0 else 'srpt_est'
    metadata.update(finite_challenger_T=finite_t, calibration_mean_ratio_scores=scores,
                    mean_first_winner=winner, selection_status='frozen_before_evaluation',
                    evaluation_seeds=eval_seeds, calibration_seeds=cal_seeds)
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Frozen finite challenger T={finite_t}s; scores={scores}; mean-first winner={winner}', flush=True)

    def write_csv(path, rows):
        with path.open('x', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    write_csv(out/'calibration_results.csv', results)
    for load in cfg['loads']:
        for seed in eval_seeds:
            execute('evaluation', load, seed, 'Baseline', 'full', 'fifo_call', monitor_t=finite_t)
            execute('evaluation', load, seed, 'Grading Only', 'lazy_hit', 'fifo_call', monitor_t=finite_t)
            for arm, variant in [('Scheduling Only', 'full'), ('Combined', 'lazy_hit')]:
                execute('evaluation', load, seed, arm, variant, 'srpt_est', monitor_t=finite_t)
                execute('evaluation', load, seed, arm, variant, 'srpt_est', alpha, monitor_t=finite_t)
                execute('evaluation', load, seed, arm, variant, 'srpt_threshold', threshold=old_t, monitor_t=finite_t)
                execute('evaluation', load, seed, arm, variant, 'srpt_rescue', threshold=finite_t, monitor_t=finite_t)
            print(f'Evaluation: rho={load} seed={seed} complete', flush=True)
    write_csv(out/'sim_results.csv', results)
    eval_rows = [r for r in results if r['stage'] == 'evaluation']
    metrics = ['mean', 'p95', 'mean_correct', 'p95_correct', 'mean_corrected', 'p95_corrected',
               'mean_gpu_wait', 'max_gpu_wait', 'p99_gpu_wait', 'threshold_exceed_fraction', 'throughput',
               'rescue_request_fraction', 'mean_rescues_per_request', 'override_fraction_all_dispatches',
               'max_consecutive_overrides', 'noncorrection_shorter_bypass_events']
    keys = list(dict.fromkeys((r['load'], r['arm'], r['policy'], r['aging_alpha'], r['threshold_seconds']) for r in eval_rows))
    summary = []
    for load, arm, policy, aging, threshold in keys:
        group = [r for r in eval_rows if (r['load'], r['arm'], r['policy'], r['aging_alpha'], r['threshold_seconds']) == (load, arm, policy, aging, threshold)]
        summary.append({'load': load, 'arm': arm, 'policy': policy, 'aging_alpha': aging, 'threshold_seconds': threshold,
                        **{k: float(np.mean([r[k] for r in group])) for k in metrics},
                        'worst_seed_max_wait': max(r['max_gpu_wait'] for r in group)})
    write_csv(out/'evaluation_summary.csv', summary)
    lines = ['# SRPT 1회 우선권 실험', '', f'유한 임계값 후보: {thresholds_by_percentile}',
             f'전체 평균 우선 선택 결과: {winner}. 비교용 유한 T={finite_t}초.',
             f'선택 점수(전체 평균 / 같은 조건 SRPT 평균의 조건 간 평균): {scores}', '',
             '| 부하 | 구성 | 정책 | alpha | T | 전체 평균 | 비보정 평균 | Correction 평균 | Correction p95 | 평균 대기 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['load']} | {r['arm']} | {r['policy']} | {r['aging_alpha']} | {r['threshold_seconds']} | " +
                     ' | '.join(f'{r[m]:.3f}' for m in ['mean', 'mean_correct', 'mean_corrected', 'p95_corrected', 'mean_gpu_wait'])+' |')
    lines += ['', '## 해석 범위', '',
              f"조건별 재사용 요청 {cfg['n_queries']}개, 초기 {warmup_n}개 제외. 원본 문항수·Correction 수는 metadata.json 참조.",
              'Full 기반 Scheduling Only가 주 평가이고 Lazy-hit Combined는 기존 정확도 저하가 있는 참고 결과입니다. 새 답변 생성·정확도 측정·GPU 사용률 측정은 하지 않습니다.',
              '전체 평균만으로 선택했습니다. 별도 Correction 가중치·임의 허용 손실률은 없습니다. SRPT가 선택되면 유한 T는 개선안으로 채택한 것이 아니라 가장 나은 유한 후보를 평가한 것입니다.',
              '임계값은 선택 시드의 모든 부하에서 모은 요청별 최종 누적 큐 대기 분포로 만들었습니다. 각 조건 요청 수가 같아 동일 가중입니다. 평가 시드로 T를 수정하지 않습니다.',
              '누적 대기 > T이면 한 호출 우선, 이후 추가 큐 대기 > T일 때 다시 자격을 얻습니다. 자격 초기화는 전체 누적 대기 통계를 지우지 않으며 실행시간은 대기에 포함하지 않습니다.',
              'SRPT는 호출 경계에서만 선택하며 실행 중인 호출은 중단하지 않습니다. 비용 모델과 원본 질문 풀은 이전 실험과 같고, 선택/평가 도착 시드는 분리했습니다.',
              '정책별 도착 시각·질문 인덱스는 동일합니다. Full 평균 작업량으로 도착률을 정해 4종 구성에도 동일 트래픽을 사용합니다.',
              '지연·대기 통계는 warmup 제외입니다. dispatch 진단은 warmup 포함 전체 실행이며, 우선순위 역전은 더 작은 예상 잔여량의 준비된 호출을 제치고 선택한 사건입니다. FIFO 구성의 진단값은 SRPT 비교에 사용하지 않습니다.',
              '비보정 shorter bypass는 사후 분석 분류이며 스케줄러는 미래 Correction 여부를 보지 않습니다. 같은 요청이 여러 번 밀리면 여러 사건으로 집계합니다.',
              '임계값 초과 비율은 모든 평가 정책에서 같은 비교용 T를 기준으로 최종 누적 대기 > T인 요청 비율입니다. 반복 자격 획득 비율과 다릅니다.',
              '표의 p95와 최대 대기는 시드별 통계의 평균이며, 전체 시드 최대 대기는 worst_seed_max_wait입니다. 처리량은 집계 요청 수 / (최초 집계 요청 도착부터 마지막 완료까지)입니다.',
              '한 호출씩 구제해도 서로 다른 초과 요청이 연속 선택될 수 있습니다. 이 정책은 전체 우선 호출 비율 상한이나 응답시간 보장을 제공하지 않습니다.',
              'Lazy-rest 미평가 문서 2~3개 제한 실험은 이번에도 실행하지 않았습니다.']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved:', out, flush=True)



def tradeoff_study():
    """All prespecified T values, original persistent-priority policy, Full trace."""
    import hashlib
    import shutil
    from datetime import datetime
    from zoneinfo import ZoneInfo

    p = argparse.ArgumentParser()
    p.add_argument('--tradeoff-study', action='store_true')
    p.add_argument('--config', type=Path, default=HERE/'config.local.json')
    p.add_argument('--records', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    study = cfg['tradeoff_study']
    thresholds, alphas, seeds = study['thresholds_seconds'], study['aging_alphas'], study['seeds']
    if not thresholds or any(not np.isfinite(t) or t < 0 for t in thresholds):
        p.error('Invalid thresholds')
    if len(set(thresholds)) != len(thresholds) or len(set(seeds)) != len(seeds):
        p.error('Duplicate thresholds or seeds')
    source = args.records.read_bytes()
    records = [json.loads(l) for l in source.decode('utf-8-sig').splitlines() if l.strip()]
    rows = [r for r in records if r['variant'] == 'full' and r['repeat'] == study['trace_repeat']]
    by_id = {r['query_id']: r for r in rows}
    if len(rows) != len(by_id) or len(rows) < 10:
        p.error('Need >=10 unique Full traces')
    ids = sorted(by_id)
    rows = [by_id[q] for q in ids]
    if any(not r['calls'] or any(not np.isfinite(c['server_total']) or c['server_total'] <= 0 for c in r['calls']) for r in rows):
        p.error('Every call requires positive measured server duration')
    jobs = [job_from_row(r) for r in rows]
    attach_oof_predictions(jobs)
    model = fit_cost_model(jobs)
    mean_work = float(np.mean([j.work for j in jobs]))
    out = ROOT/'results/scheduling_optimization'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_tradeoff')
    out.mkdir(parents=True, exist_ok=False)
    (out/'source_snapshot').mkdir(); (out/'workloads').mkdir()
    for name in ['run.py', 'simulator.py', 'verify.py', 'plot.py', 'config.local.json', 'README.md']:
        shutil.copy2(HERE/name, out/'source_snapshot'/name)
    (out/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    metadata = {'source_path': str(args.records.resolve()), 'source_sha256': hashlib.sha256(source).hexdigest(),
                'original_questions': len(jobs), 'original_correction_questions': sum(j.action != 'correct' for j in jobs),
                'historical_accuracy': float(np.mean([r['acc'] for r in rows])), 'question_ids': ids,
                'model': model.to_dict(), 'mean_server_work_seconds': mean_work,
                'prediction_scope': 'Same sorted Full pool; query predictions 5-fold OOF seed 0; call model fit on whole pool',
                'policy': 'Original srpt_threshold: cumulative queue wait > T, then greatest cumulative queue wait; no reset, no Aging combination; calls nonpreemptive',
                'study_scope': 'Prespecified multi-T sensitivity analysis; no composite score, no selected T, no independent final-test claim',
                'primary_reference': 'SRPT + Aging alpha=0.5', 'secondary_references': ['SRPT', 'Aging alpha=0.1', 'Aging alpha=1.0'],
                'dominance_rule': 'At the same load, seed-mean overall E2E strictly smaller AND seed-mean Correction p95 no larger. Descriptive comparison, not significance.',
                'seeds': seeds, 'thresholds_seconds': thresholds, 'aging_alphas': alphas}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    policies = [('srpt_est', 0, None)] + [('srpt_est', a, None) for a in alphas] + [('srpt_threshold', 0, t) for t in thresholds]
    results = []
    warmup = int(cfg['n_queries']*cfg['warmup_frac'])
    def name(policy, alpha, threshold):
        return f'T={threshold:g}s' if threshold is not None else (f'Aging={alpha:g}' if alpha else 'SRPT')
    for load in cfg['loads']:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(jobs), cfg['n_queries'])
            arrivals = poisson_arrivals(len(idx), load/mean_work, rng)
            digest = hashlib.sha256(idx.tobytes()+np.asarray(arrivals).tobytes()).hexdigest()
            np.savez_compressed(out/'workloads'/f'rho{load}_seed{seed}.npz', indices=idx, arrivals=arrivals)
            sample = [jobs[i] for i in idx]
            for policy, alpha, threshold in policies:
                stats = {}
                runs = simulate(sample, arrivals, policy, model, aging_alpha=alpha, wait_threshold=threshold, dispatch_stats=stats)
                assert all(np.isfinite(r.finish) and abs(r.finish-r.arrival-r.job.solo-r.gpu_wait) < 1e-6 for r in runs)
                retained = runs[warmup:]
                waits = np.array([r.gpu_wait for r in retained])
                results.append({'study': 'threshold_tradeoff', 'load': load, 'seed': seed,
                                'policy': policy, 'aging_alpha': alpha, 'threshold_seconds': threshold,
                                'label': name(policy, alpha, threshold), 'workload_sha256': digest,
                                **summarize(runs, cfg['warmup_frac'], threshold),
                                'p95_gpu_wait': float(np.percentile(waits, 95)),
                                'p99_gpu_wait': float(np.percentile(waits, 99)),
                                **{f'wait_exceed_{t:g}s_fraction': float(np.mean(waits > t)) for t in thresholds},
                                'override_fraction_all_calls': stats['srpt_override_dispatches']/stats['total_dispatches'],
                                'max_consecutive_overrides_all_calls': stats['max_consecutive_overrides']})
            print(f'Tradeoff: rho={load}, seed={seed}, {len(policies)} policies complete', flush=True)
    def csv_write(path, values):
        with path.open('x', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    csv_write(out/'sim_results.csv', results)
    metrics = ['mean', 'p95', 'mean_correct', 'p95_correct', 'mean_corrected', 'p95_corrected', 'mean_gpu_wait',
               'p95_gpu_wait', 'p99_gpu_wait', 'max_gpu_wait', 'throughput', 'override_fraction_all_calls']
    metrics += [f'wait_exceed_{t:g}s_fraction' for t in thresholds]
    summary = []
    for load in cfg['loads']:
        for policy, alpha, threshold in policies:
            label = name(policy, alpha, threshold)
            group = [r for r in results if r['load'] == load and r['label'] == label]
            summary.append({'load': load, 'label': label, 'policy': policy, 'aging_alpha': alpha, 'threshold_seconds': threshold,
                            **{m: float(np.mean([r[m] for r in group])) for m in metrics},
                            **{m+'_seed_sd': float(np.std([r[m] for r in group], ddof=1)) for m in ['mean', 'p95_corrected', 'mean_correct']},
                            'worst_seed_max_wait': max(r['max_gpu_wait'] for r in group)})
    csv_write(out/'evaluation_summary.csv', summary)
    comparisons = []
    for load in cfg['loads']:
        for t in thresholds:
            candidate = next(r for r in summary if r['load'] == load and r['threshold_seconds'] == t)
            for ref in [r for r in summary if r['load'] == load and r['threshold_seconds'] is None]:
                paired = []
                for seed in seeds:
                    c = next(r for r in results if r['load'] == load and r['seed'] == seed and r['label'] == candidate['label'])
                    b = next(r for r in results if r['load'] == load and r['seed'] == seed and r['label'] == ref['label'])
                    paired.append(c['mean'] < b['mean'] and c['p95_corrected'] <= b['p95_corrected'])
                comparisons.append({'load': load, 'threshold_seconds': t, 'reference': ref['label'],
                                    'mean_e2e_change_pct': 100*(candidate['mean']/ref['mean']-1),
                                    'correction_p95_change_pct': 100*(candidate['p95_corrected']/ref['p95_corrected']-1),
                                    'noncorrection_mean_change_pct': 100*(candidate['mean_correct']/ref['mean_correct']-1),
                                    'dominates_on_seed_means': candidate['mean'] < ref['mean'] and candidate['p95_corrected'] <= ref['p95_corrected'],
                                    'dominating_seed_count': sum(paired), 'total_seeds': len(seeds)})
    csv_write(out/'paired_comparisons.csv', comparisons)
    lines = ['# 대기 임계값의 성능 절충 분석', '',
             '처음 제안한 누적 대기 최장 요청 우선 정책만 사용합니다. T 초과 후 대기 기준을 초기화하지 않습니다.',
             f'T={thresholds}초를 모두 비교하며 종합 점수·최적 T 선택은 하지 않습니다. 주 비교 기준은 Aging α=0.5입니다.',
             '', '| 부하 | 정책 | 전체 E2E 평균 | 전체 p95 | 비보정 E2E 평균 | 비보정 p95 | Correction E2E 평균 | Correction p95 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['load']} | {r['label']} | "+' | '.join(f'{r[m]:.3f}' for m in ['mean','p95','mean_correct','p95_correct','mean_corrected','p95_corrected'])+' |')
    lines += ['', '## 두 지표 동시 개선 판정', '',
              '조건: 같은 부하에서 전체 평균 E2E가 엄격히 작고 Correction p95는 같거나 작음. 시드별 통계를 평균한 기술적 비교이며 통계적 유의성 주장이 아닙니다.',
              '| 부하 | T | 기준 | 전체 평균 변화 % | Correction p95 변화 % | 비보정 평균 변화 % | 동시 개선 | 개별 시드 동시 개선 |',
              '|---|---:|---|---:|---:|---:|---|---|']
    for r in comparisons:
        if r['reference'] == 'Aging=0.5':
            lines.append(f"| {r['load']} | {r['threshold_seconds']} | {r['reference']} | {r['mean_e2e_change_pct']:+.2f} | {r['correction_p95_change_pct']:+.2f} | {r['noncorrection_mean_change_pct']:+.2f} | {r['dominates_on_seed_means']} | {r['dominating_seed_count']}/{r['total_seeds']} |")
    lines += ['', '## 조건과 해석 범위', '',
              f"Full 원본 {len(jobs)}문항, Correction {metadata['original_correction_questions']}개. 조건별 {cfg['n_queries']}개 반복 추출, 앞 {warmup}개 제외. 부하 {cfg['loads']}, 시드 {seeds}.",
              'E2E = 완료 시각 - 도착 시각이며 모델링한 검색 지연·LLM 실행·큐 대기를 포함합니다. Correction 지표는 보정 단계만의 시간이 아니라 보정 요청의 전체 E2E입니다.',
              '전체 정책에 같은 Trace 표본·도착 시각·예측 모델을 공유했습니다. 전체 평균 E2E와 Correction p95를 합친 점수로 후보를 탈락시키지 않습니다.',
              '모든 T를 공개하는 탐색적 민감도 분석이며, 이 표에서 좋은 T를 골라 독립 최종 성능이라고 주장하지 않습니다.',
              '같은 Full 답변을 재생하므로 정책 간 답변은 동일합니다. LLM/API 재실행, 새 정확도·GPU 사용률 측정은 없습니다.',
              '예측 모델은 이전과 동일한 in-pool 방식입니다. 질문 단위 예측만 OOF이고 호출 비용은 전체 원본 풀로 학습합니다.',
              '통계의 p95·최대 대기는 시드별 통계의 평균입니다. 전체 시드 최대값은 worst_seed_max_wait입니다. 처리량은 집계 요청 최초 도착~마지막 완료 구간을 사용합니다.',
              'wait_exceed_* 열은 동일 T에서 정책 간 누적 대기를 비교하기 위한 요청 비율입니다. threshold_exceed_fraction은 각 임계값 정책의 자체 T 기준이며 SRPT/Aging은 미정의입니다.',
              '순서 역전 진단은 warmup 포함 전체 호출 기준입니다. 더 작은 예상 잔여 작업이 있는 호출을 제친 비율이며 E2E 통계와 집계 범위가 다릅니다.',
              '이번에는 스케줄링 절충만 분리하려고 Full을 사용했습니다. 기존 4종 통합 결과와 1회구제 실험은 별도로 보존하며 새 실험에 섞지 않습니다.']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('DOMINATING COMBINATIONS:', [r for r in comparisons if r['dominates_on_seed_means']], flush=True)
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    import sys
    if '--tradeoff-study' in sys.argv:
        tradeoff_study()
    elif '--rescue-study' in sys.argv:
        rescue_study()
    elif '--threshold-study' in sys.argv:
        threshold_study()
    else:
        main()
