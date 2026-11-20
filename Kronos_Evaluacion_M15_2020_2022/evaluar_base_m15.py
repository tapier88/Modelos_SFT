"""Offline, reproducible M15 validation. No optimizer, training or test-data reads."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

L, H, STEP = 64, 4, 15
SEED = 271828
SOURCE_HASHES = {
    'XAUUSD_M15_BID_2020.parquet': '449e19982f7c578ba7ff88f0c1f2c0e2b4363e08ded34e9ba05f4f0f321c560b',
    'XAUUSD_M15_BID_2021.parquet': 'd4605da5798fcbb76dda953c106cd248901003b6e68bb6b5de048e53b6ab192b',
    'XAUUSD_M15_BID_2022.parquet': 'b2f945631fb6a78cb2bc0ca591b65865a713380d661ee1e1632ebbfa63ecc170',
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def state_digest(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def select_nonoverlapping(ix):
    keep, last_end = [], None
    for i, row in ix.iterrows():
        if last_end is None or row.decision_time >= last_end:
            keep.append(i)
            last_end = row.target_end_exclusive
    return ix.loc[keep].reset_index(drop=True)


def prepare(source):
    frames, manifest = [], []
    for name, expected in SOURCE_HASHES.items():
        p = source / name
        actual = digest(p)
        assert actual == expected, f'Source hash mismatch: {name}'
        df = pd.read_parquet(p)
        assert list(df.columns) == ['timestamp_utc', 'open', 'high', 'low', 'close', 'volume', 'm1_candles']
        df['timestamp_utc'] = pd.to_datetime(df.timestamp_utc, utc=True)
        t = df.timestamp_utc
        assert t.dt.year.eq(int(name[-12:-8])).all()
        assert not df.isna().any().any()
        assert np.isfinite(df[['open', 'high', 'low', 'close', 'volume']]).all().all()
        assert t.is_monotonic_increasing and t.is_unique
        assert (t.astype('int64') % (STEP * 60 * 10**9) == 0).all()
        assert df.m1_candles.eq(STEP).all() and df.volume.ge(0).all()
        assert (df.low.gt(0) & df.low.le(df[['open', 'close']].min(axis=1)) &
                df.high.ge(df[['open', 'close']].max(axis=1)) & df.high.ge(df.low)).all()
        manifest.append({'file': name, 'sha256': actual, 'rows': len(df), 'bytes': p.stat().st_size})
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    t = df.timestamp_utc
    assert t.is_monotonic_increasing and t.is_unique
    assert t.ge(pd.Timestamp('2020-01-01', tz='UTC')).all()
    assert t.lt(pd.Timestamp('2023-01-01', tz='UTC')).all()
    segment = t.diff().ne(pd.Timedelta(minutes=STEP)).cumsum()
    starts = []
    for _, group in df.groupby(segment, sort=False):
        starts.extend(range(int(group.index[0]), int(group.index[-1]) - L - H + 2))
    starts = np.array(starts, dtype=np.int64)
    ix = pd.DataFrame({'start_row': starts, 'split': 'validation'})
    ix['decision_time'] = t.iloc[starts + L].to_numpy()
    ix['target_end_exclusive'] = (t.iloc[starts + L + H - 1] + pd.Timedelta(minutes=STEP)).to_numpy()
    ny = pd.to_datetime(ix.decision_time, utc=True).dt.tz_convert('America/New_York')
    minute = ny.dt.hour * 60 + ny.dt.minute
    ix['session_date_ny'] = ny.dt.strftime('%Y-%m-%d')
    ix['year_ny'] = ny.dt.year
    ix['ny_full_horizon'] = (minute >= 510) & (minute + H * STEP <= 780)
    assert len(ix) == 18168 and int(ix.ny_full_horizon.sum()) == 6565, 'Eligibility differs from audited pipeline'
    selected = select_nonoverlapping(ix)
    assert (selected.decision_time.iloc[1:].reset_index(drop=True) >=
            selected.target_end_exclusive.iloc[:-1].reset_index(drop=True)).all()
    selected.insert(0, 'window_id', np.arange(len(selected)))
    coverage = {'rows': len(df), 'candidate_windows': len(ix), 'candidate_ny_windows': int(ix.ny_full_horizon.sum()),
                'selected_windows': len(selected), 'selected_ny_windows': int(selected.ny_full_horizon.sum()),
                'eligible_days_ny': int(ix.session_date_ny.nunique()),
                'selected_days_ny': int(selected.session_date_ny.nunique()),
                'available_bar_days_ny': int(t.dt.tz_convert('America/New_York').dt.date.nunique()),
                'candidate_by_year': ix.groupby('year_ny').size().to_dict(),
                'selected_by_year': selected.groupby('year_ny').size().to_dict()}
    return df, selected, manifest, coverage


def cluster_interval(window_metrics, reps=2000, block=1):
    """Ratio of sampled cluster sums/counts, preserving the per-window estimand."""
    days = window_metrics.groupby('session_date_ny', sort=True).delta_mae.agg(['sum', 'count'])
    sums, counts = days['sum'].to_numpy(), days['count'].to_numpy()
    n = len(days)
    if n < 2:
        return [None, None]
    rng = np.random.default_rng(SEED + block)
    samples = []
    for _ in range(reps):
        if block == 1:
            ids = rng.integers(n, size=n)
        else:
            # Circular moving blocks of observed eligible days; not calendar days.
            anchors = rng.integers(n, size=math.ceil(n / block))
            ids = ((anchors[:, None] + np.arange(block)) % n).ravel()[:n]
        samples.append(float(sums[ids].sum() / counts[ids].sum()))
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def summarize(predictions):
    p = predictions.copy()
    p['model_abs'] = abs(p.predicted_close - p.actual_close)
    p['persist_abs'] = abs(p.persistence_close - p.actual_close)
    p['model_squared'] = (p.predicted_close - p.actual_close)**2
    p['persist_squared'] = (p.persistence_close - p.actual_close)**2
    windows = p.groupby('window_id', sort=True).agg(
        session_date_ny=('session_date_ny', 'first'), year_ny=('year_ny', 'first'),
        ny_full_horizon=('ny_full_horizon', 'first'), model_mae=('model_abs', 'mean'),
        persistence_mae=('persist_abs', 'mean'))
    windows['delta_mae'] = windows.model_mae - windows.persistence_mae
    groups = [('ALL', np.ones(len(p), dtype=bool)), ('NY', p.ny_full_horizon), ('OTHER', ~p.ny_full_horizon)]
    for year in [2020, 2021, 2022]:
        groups.extend([(str(year), p.year_ny.eq(year)), (f'{year}_NY', p.year_ny.eq(year) & p.ny_full_horizon)])
    rows = []
    for label, mask in groups:
        g = p.loc[mask]
        w = windows.loc[g.window_id.unique()]
        terminal = g[g.horizon == H]
        truth_move = terminal.actual_close - terminal.persistence_close
        predicted_move = terminal.predicted_close - terminal.persistence_close
        active = truth_move.abs() > 1e-8
        ci = cluster_interval(w)
        block_ci = cluster_interval(w, block=5)
        m, b = float(g.model_abs.mean()), float(g.persist_abs.mean())
        rows.append({'group': label, 'windows': len(w), 'days': int(w.session_date_ny.nunique()),
                     'close_mae': m, 'persistence_close_mae': b, 'mae_skill_pct': 100 * (1 - m / b),
                     'close_rmse': float(np.sqrt(g.model_squared.mean())),
                     'persistence_close_rmse': float(np.sqrt(g.persist_squared.mean())),
                     'delta_mae': m-b, 'delta_ci95_low': ci[0], 'delta_ci95_high': ci[1],
                     'delta_5day_ci95_low': block_ci[0], 'delta_5day_ci95_high': block_ci[1],
                     'terminal_mae': float(terminal.model_abs.mean()),
                     'persistence_terminal_mae': float(terminal.persist_abs.mean()),
                     'terminal_magnitude_mae': float((predicted_move.abs()-truth_move.abs()).abs().mean()),
                     'persistence_terminal_magnitude_mae': float(truth_move.abs().mean()),
                     'direction_accuracy_nonflat': float((np.sign(predicted_move[active]) == np.sign(truth_move[active])).mean()),
                     'always_up_accuracy_nonflat': float((truth_move[active] > 0).mean()),
                     'true_flat_windows': int((~active).sum()),
                     'predicted_flat_windows': int((predicted_move.abs() <= 1e-8).sum())})
    by_horizon = p.groupby('horizon').agg(model_mae=('model_abs', 'mean'), persistence_mae=('persist_abs', 'mean'))
    return pd.DataFrame(rows), windows.reset_index(), by_horizon.reset_index()


def self_test():
    times = pd.date_range('2020-01-01', periods=12, freq='15min', tz='UTC')
    ix = pd.DataFrame({'decision_time': times, 'target_end_exclusive': times + pd.Timedelta(hours=1)})
    assert select_nonoverlapping(ix).decision_time.tolist() == times[::4].tolist()
    synthetic = pd.DataFrame({'session_date_ny': ['2020-01-01', '2020-01-01', '2020-01-02'], 'delta_mae': [2., 2., 2.]})
    assert cluster_interval(synthetic, reps=100) == [2., 2.]
    assert cluster_interval(synthetic, reps=100, block=5) == [2., 2.]
    return {'nonoverlap_selection': True, 'bootstrap_constant_difference': True}


def write_report(out, summary, coverage, status):
    lines = ['# Evaluación M15 de Kronos: pesos base, 2020–2022', '',
             'Evaluación terminada sin entrenamiento. RUN_FULL=False; test 2023–2024 no leído.', '',
             f"Se evaluaron {coverage['selected_windows']:,} ventanas con objetivos no solapados, "
             f"de {coverage['candidate_windows']:,} candidatas elegibles. "
             f"Ventanas NY: {coverage['selected_ny_windows']:,}. Contexto: 64 velas (16 horas); horizonte: 4 velas (1 hora).", '',
             '| Grupo | Ventanas | MAE Kronos | MAE último cierre | Mejora MAE % | IC 95% diferencia MAE |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.group} | {row.windows} | {row.close_mae:.4f} | {row.persistence_close_mae:.4f} | '
                     f'{row.mae_skill_pct:.2f} | [{row.delta_ci95_low:.4f}, {row.delta_ci95_high:.4f}] |')
    allrow = summary[summary.group == 'ALL'].iloc[0]
    nyrow = summary[summary.group == 'NY'].iloc[0]
    lines.extend(['', 'MAE en unidades de precio XAUUSD; menor es mejor. Mejora positiva favorece Kronos. '
                  'Diferencia = error Kronos menos error de mantener el último cierre; un intervalo completamente negativo favorece Kronos.', '',
                  f"Resultado global: mejora MAE {allrow.mae_skill_pct:.2f}%; NY: {nyrow.mae_skill_pct:.2f}%.", '',
                  '## Protocolo y alcance', '',
                  '- Pesos locales de Kronos-small y Kronos-Tokenizer-base, sin cargar checkpoints entrenados. '
                  'Se conservaron los hashes de archivos y de todos los parámetros antes/después.',
                  '- Se evalúa el adaptador causal auditado del proyecto, con generación greedy determinista. '
                  'No es una reproducción del predictor oficial con muestreo; las conclusiones se limitan a esta configuración.',
                  '- Normalización float32 exclusivamente con el contexto, clip ±5, amount=0. '
                  'Los valores futuros se retiraron de la entrada de inferencia. La verdad y persistencia se puntúan con los valores originales float64.',
                  '- Selección cronológica de todas las ventanas elegibles con objetivos separados al menos 60 minutos; '
                  'sin selección basada en resultados. Los contextos pueden solaparse.',
                  '- NY exige que todo el horizonte esté entre 08:30 y 13:00 America/New_York, con DST. '
                  'El resto incluye horizontes que cruzan los bordes de sesión. Los años corresponden a la fecha de decisión NY.',
                  '- IC 95%: 2.000 remuestreos pareados de días NY, conservando todas las ventanas de cada día. '
                  'Se añade sensibilidad con bloques circulares de 5 días elegibles en summary.csv. '
                  'Los intervalos no eliminan sesgos de selección ni equivalen a beneficio de trading.',
                  f"- Cobertura: {coverage['selected_days_ny']} días con ventanas seleccionadas, "
                  f"{coverage['available_bar_days_ny']} fechas NY con al menos una vela. "
                  'El requisito de 17 horas continuas excluye sesiones con huecos; no representa todas las oportunidades de mercado.',
                  '- La exactitud direccional es descriptiva, excluye movimientos reales nulos y cuenta predicciones nulas como fallos. '
                  'Persistencia predice movimiento cero y se abstiene de dirección. Always-up solo describe el desequilibrio de clases.',
                  '- No se midieron spread, comisiones, deslizamiento, órdenes, rentabilidad ni drawdown. '
                  'Puede existir solapamiento entre el preentrenamiento del modelo y estas fechas históricas.', '',
                  '## Decisión', ''])
    if nyrow.delta_ci95_high < 0 and allrow.delta_ci95_high < 0:
        lines.append('Hay mejora de error de precio en esta muestra y configuración. El siguiente paso es una evaluación cronológica '
                     'walk-forward predefinida y pruebas de robustez; esto no demuestra una ventaja operable.')
    else:
        lines.append('No queda demostrada una mejora robusta global y NY frente al último cierre. '
                     'No activar entrenamiento completo por este resultado. El siguiente experimento es contrastar la '
                     'inferencia oficial y la representación de volumen/amount sobre las mismas ventanas, con protocolo fijo, '
                     'antes de un piloto SFT limitado. Conservar 2023–2024 reservado.')
    lines += ['', '## Reproducibilidad', '',
              'config.json fija el experimento antes de predecir; status.json registra verificaciones, entorno y hashes; '
              'predictions.parquet contiene cada predicción; window_metrics.csv permite recalcular los intervalos. '
              'Los fragmentos en shards permiten reanudar sin repetir inferencias terminadas.', '',
              f"Tiempo de inferencia: {status['inference_seconds']:.1f} segundos. Dispositivo: CPU.", '']
    (out/'INFORME.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, default=Path(r'C:\workspace\kronos_sft_20261107'))
    ap.add_argument('--source', type=Path, default=Path(r'C:\workspace\kronos_audit_20261107\parquet'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()
    out = args.output.resolve()
    assert out != args.source.resolve() and args.source.resolve() not in out.parents
    out.mkdir(parents=True, exist_ok=True)
    (out/'shards').mkdir(exist_ok=True)
    def log(message):
        text = time.strftime('%Y-%m-%dT%H:%M:%S') + ' ' + message
        print(text, flush=True)
        with (out/'execution.log').open('a', encoding='utf-8') as f:
            f.write(text+'\n')
    tests = self_test()
    spec = importlib.util.spec_from_file_location('kronos_verified_baseline', args.bundle/'kronos_pipeline.py')
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    kp.seed_all(SEED)
    df, ix, manifest, coverage = prepare(args.source)
    log('DATA_VERIFIED ' + json.dumps(coverage))
    weights = args.bundle/'weights'
    weight_hashes = {str(p.relative_to(weights)): digest(p) for p in sorted(weights.rglob('*')) if p.is_file()}
    config = {'protocol': 'M15_BASE_GREEDY_CAUSAL_ADAPTER_V1', 'RUN_FULL': False,
              'training': False, 'test_evaluated': False, 'years': [2020, 2021, 2022],
              'context': L, 'horizon': H, 'step_minutes': STEP, 'batch_size': args.batch_size,
              'selection': 'greedy_chronological_nonoverlapping_targets', 'seed': SEED, 'bootstrap_reps': 2000,
              'decoder': 'audited_causal_adapter_greedy', 'normalization': 'context_float32_std_ddof0_eps1e-5_clip5',
              'amount': 'zero_unavailable', 'torch': torch.__version__, 'numpy': np.__version__,
              'pandas': pd.__version__, 'device': 'cpu', 'threads': torch.get_num_threads(),
              'pipeline_sha256': digest(args.bundle/'kronos_pipeline.py'), 'evaluator_sha256': digest(Path(__file__)),
              'vendor_sha256': {str(p.relative_to(args.bundle)): digest(p) for p in sorted((args.bundle/'vendor').rglob('*.py'))},
              'model_revisions_recorded_by_pipeline': kp.REVISIONS, 'weight_files': weight_hashes, 'sources': manifest}
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text(encoding='utf-8')) == config, 'Resume config mismatch; use a fresh output directory'
    else:
        write_json(out/'config.json', config)
    ix.to_parquet(out/'selected_windows.parquet', index=False)
    write_json(out/'coverage.json', coverage)
    model, tokenizer = kp.load_models(weights, torch.device('cpu'))
    model.eval().requires_grad_(False)
    tokenizer.eval().requires_grad_(False)
    initial = {'model': state_digest(model), 'tokenizer': state_digest(tokenizer)}
    ds = kp.WindowDataset(df, ix, 'M15', 'validation')
    tests.update(kp.contract_tests(model, tokenizer, ds, torch.device('cpu')))
    assert state_digest(model) == initial['model'] and state_digest(tokenizer) == initial['tokenizer']
    model.eval()
    write_json(out/'tests.json', tests)
    log('CONTRACT_TESTS_PASSED; original weight files loaded; gradients disabled')
    started = time.monotonic()
    done, total = 0, len(ix)
    all_parts = []
    for start in range(0, total, args.batch_size):
        stop = min(start + args.batch_size, total)
        path = out/'shards'/f'{start:06d}_{stop:06d}.parquet'
        if path.exists():
            part = pd.read_parquet(path)
            assert part.window_id.tolist() == np.repeat(np.arange(start, stop), H).tolist()
            assert part.horizon.tolist() == np.tile(np.arange(1, H+1), stop-start).tolist()
        else:
            batch = next(iter(DataLoader(Subset(ds, range(start, stop)), batch_size=stop-start, shuffle=False)))
            # Physically remove all future prices and truth before model inference.
            infer = {'x': batch['x'][:, :L].contiguous(), 'stamp': batch['stamp'], 'mean': batch['mean'], 'scale': batch['scale']}
            with torch.inference_mode():
                prediction = kp.rollout(model, tokenizer, infer, L, H).cpu().numpy()
            assert np.isfinite(prediction).all()
            rows = []
            for bi, wi in enumerate(range(start, stop)):
                meta = ix.iloc[wi]
                pos = int(meta.start_row)
                last_close = float(df.close.iloc[pos+L-1])
                for h in range(H):
                    rows.append({'window_id': wi, 'decision_time': meta.decision_time,
                                 'target_close_time': df.timestamp_utc.iloc[pos+L+h] + pd.Timedelta(minutes=STEP),
                                 'session_date_ny': meta.session_date_ny, 'year_ny': int(meta.year_ny),
                                 'ny_full_horizon': bool(meta.ny_full_horizon), 'horizon': h+1,
                                 'actual_close': float(df.close.iloc[pos+L+h]),
                                 'predicted_close': float(prediction[bi, h, 3]), 'persistence_close': last_close})
            part = pd.DataFrame(rows)
            tmp = path.with_suffix('.tmp')
            part.to_parquet(tmp, index=False)
            tmp.replace(path)
        all_parts.append(part)
        done = stop
        if start == 0 or (start // args.batch_size) % 10 == 0 or stop == total:
            elapsed = time.monotonic() - started
            progress = {'state': 'running', 'done': done, 'total': total, 'percent': round(100*done/total, 2),
                        'elapsed_seconds': round(elapsed, 1), 'eta_seconds': round(elapsed/done*(total-done), 1)}
            write_json(out/'progress.json', progress)
            log('PROGRESS ' + json.dumps(progress))
    duration = time.monotonic() - started
    predictions = pd.concat(all_parts, ignore_index=True)
    assert len(predictions) == total*H and not predictions.duplicated(['window_id', 'horizon']).any()
    predictions.to_parquet(out/'predictions.parquet', index=False)
    summary, windows, horizons = summarize(predictions)
    summary.to_csv(out/'summary.csv', index=False)
    windows.to_csv(out/'window_metrics.csv', index=False)
    horizons.to_csv(out/'horizon_metrics.csv', index=False)
    assert initial == {'model': state_digest(model), 'tokenizer': state_digest(tokenizer)}
    assert all(digest(args.source/m['file']) == m['sha256'] for m in manifest)
    assert all(digest(weights/name) == h for name, h in weight_hashes.items())
    status = {'state': 'completed', 'RUN_FULL': False, 'training_done': False, 'test_evaluated': False,
              'source_hashes_match_latest_colab_manifest': True, 'sources_unchanged': True,
              'weight_files_unchanged': True, 'parameters_unchanged': True, 'parameter_sha256': initial,
              'contract_tests_passed': True, 'all_selected_windows_evaluated': True,
              'inference_seconds': duration, 'python': sys.version, 'platform': platform.platform(),
              'coverage': coverage, 'metrics': summary.to_dict(orient='records')}
    write_report(out, summary, coverage, status)
    write_json(out/'status.json', status)
    write_json(out/'progress.json', {'state': 'completed', 'done': total, 'total': total, 'percent': 100})
    log('COMPLETED ' + json.dumps(summary[['group', 'windows', 'close_mae', 'persistence_close_mae', 'mae_skill_pct']].to_dict(orient='records')))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
