#@title MISIÓN 06 · 1. Datos autorizados, índice nuevo y normalización causal
assert globals().get('M6_PREFLIGHT_OK') is True, 'Ejecutar primero M06 · 0'
import numpy as np
import pandas as pd

M6_SOURCE = M6_BASE / 'datasets/processed/kronos_ohlcv_v1'
M6_SOURCE_MANIFEST, m6_parts = [], []
# Lista cerrada; no se enumeran directorios de datos, archivos ZIP ni índices históricos.
def m6_authorized_source(year):
    if type(year) is not int or year not in range(2010, 2021):
        raise ValueError('M06 solo permite archivos fuente 2010–2020')
    return M6_SOURCE / f'XAUUSD_M15_BID_{year}.parquet'

for m6_year in range(2010, 2021):
    m6_path = m6_authorized_source(m6_year)
    m6_digest = m6_sha(m6_path)
    m6_part = pd.read_parquet(m6_path)
    m6_part['timestamp_utc'] = pd.to_datetime(m6_part.timestamp_utc, utc=True)
    assert len(m6_part) and m6_part.timestamp_utc.dt.year.eq(m6_year).all()
    assert m6_part.timestamp_utc.is_monotonic_increasing and m6_part.timestamp_utc.is_unique
    assert m6_part.m1_candles.eq(15).all()
    m6_prices = m6_part[['open', 'high', 'low', 'close', 'volume']]
    assert np.isfinite(m6_prices.to_numpy()).all()
    assert (m6_part.low > 0).all() and (m6_part.volume >= 0).all()
    assert (m6_part.low <= m6_part[['open', 'close']].min(axis=1)).all()
    assert (m6_part.high >= m6_part[['open', 'close']].max(axis=1)).all()
    assert (m6_part.low <= m6_part.high).all()
    assert m6_sha(m6_path) == m6_digest, 'Fuente cambió durante lectura'
    M6_SOURCE_MANIFEST.append({'year': m6_year, 'file': m6_path.name, 'rows': len(m6_part), 'sha256': m6_digest})
    m6_parts.append(m6_part[['timestamp_utc', 'open', 'high', 'low', 'close', 'volume']].copy())
m6_df = pd.concat(m6_parts, ignore_index=True)
del m6_parts, m6_part, m6_prices
assert m6_df.timestamp_utc.is_unique and m6_df.timestamp_utc.is_monotonic_increasing
m6_ts = m6_df.timestamp_utc.astype('int64').to_numpy()
M6_STEP = 15 * 60 * 10**9
assert np.all(m6_ts % M6_STEP == 0)
m6_df['split'] = np.where(m6_df.timestamp_utc.dt.year <= 2019, 'train', 'development')
m6_df['segment'] = (m6_df.timestamp_utc.diff().ne(pd.Timedelta(minutes=15)) |
                    m6_df.split.ne(m6_df.split.shift())).cumsum()
m6_windows = []
for m6_segment, m6_group in m6_df.groupby('segment', sort=False):
    m6_first, m6_last = int(m6_group.index[0]), int(m6_group.index[-1])
    m6_starts = np.arange(m6_first, m6_last - 59 + 1, dtype=np.int64)
    if not len(m6_starts):
        continue
    m6_decisions = m6_df.timestamp_utc.iloc[m6_starts + 56]
    m6_ny = m6_decisions.dt.tz_convert('America/New_York')
    for s, decision, ny in zip(m6_starts, m6_decisions, m6_ny):
        if ny.strftime('%H:%M') not in M6_POLICY['ny_anchors']:
            continue
        m6_windows.append({'start_row': int(s), 'split': m6_group.split.iloc[0],
                           'decision_utc': decision, 'decision_ny': ny.isoformat(),
                           'ny_date': ny.strftime('%Y-%m-%d'), 'ny_hour': ny.strftime('%H:%M'),
                           'context_start': m6_df.timestamp_utc.iloc[s],
                           'target_end': m6_df.timestamp_utc.iloc[s+59] + pd.Timedelta(minutes=15)})
m6_ix = pd.DataFrame(m6_windows).sort_values('decision_utc').reset_index(drop=True)
assert len(m6_ix) and m6_ix.decision_utc.is_unique
m6_train_ix = m6_ix[m6_ix.split.eq('train')].reset_index(drop=True)
m6_dev_ix = m6_ix[m6_ix.split.eq('development')].reset_index(drop=True)
assert len(m6_train_ix) and len(m6_dev_ix)
assert m6_dev_ix.decision_utc.dt.year.eq(2020).all()
assert m6_train_ix.target_end.max() < m6_dev_ix.context_start.min()
assert (m6_ix.decision_utc.iloc[1:].to_numpy() >= m6_ix.target_end.iloc[:-1].to_numpy()).all()
m6_starts = m6_ix.start_row.to_numpy(np.int64)
assert np.all(m6_ts[m6_starts[:, None] + np.arange(60)] == m6_ts[m6_starts, None] + np.arange(60) * M6_STEP)

def m6_normalize(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (60, 6):
        raise ValueError('Contrato M06: 60 filas, OHLCV + amount=0')
    mean = raw[:56].mean(axis=0)
    scale = raw[:56].std(axis=0) + np.float32(1e-5)
    return np.clip((raw - mean) / scale, -5, 5), mean, scale

class M6Dataset(torch.utils.data.Dataset):
    def __init__(self, frame, index):
        self.L, self.H = 56, 4
        self.index = index.reset_index(drop=True)
        self.starts = self.index.start_row.to_numpy(np.int64)
        self.values = np.column_stack([frame[['open','high','low','close','volume']].to_numpy(np.float32), np.zeros(len(frame), np.float32)])
        t = frame.timestamp_utc.dt
        self.stamp = np.column_stack([t.minute, t.hour, t.weekday, t.day, t.month]).astype(np.float32)
    def __len__(self):
        return len(self.starts)
    def __getitem__(self, i):
        s = int(self.starts[i])
        raw = self.values[s:s+60].copy()
        x, mean, scale = m6_normalize(raw)
        return {'x': torch.from_numpy(x), 'stamp': torch.from_numpy(self.stamp[s:s+60].copy()),
                'loss_mask': torch.from_numpy(np.arange(59) >= 55),
                'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
                'truth': torch.from_numpy(raw[56:].copy()), 'last_close': torch.tensor(raw[55,3])}

m6_train = M6Dataset(m6_df, m6_train_ix)
m6_dev = M6Dataset(m6_df, m6_dev_ix)
m6_probe = m6_train.values[m6_train.starts[0]:m6_train.starts[0]+60].copy()
m6_changed = m6_probe.copy()
m6_changed[56:] += np.float32(10000)
m6_before, m6_after = m6_normalize(m6_probe), m6_normalize(m6_changed)
assert np.array_equal(m6_before[0][:56], m6_after[0][:56])
assert np.array_equal(m6_before[1], m6_after[1]) and np.array_equal(m6_before[2], m6_after[2])
M6_DATA_CHECKS = {'loaded_years': list(range(2010, 2021)), 'normalization_future_invariant': True,
                  'train_windows': len(m6_train), 'development_windows': len(m6_dev),
                  'development_ny_sessions': int(m6_dev_ix.ny_date.nunique()),
                  'causal_model_checks': 'PENDING', 'predictive_experiments': 'NOT_EXECUTED'}
for rel in ['auditoria/ventanas_m06.parquet', 'auditoria/train_2010_2019.csv', 'auditoria/desarrollo_2020.csv']:
    assert not (M6_DIR / rel).exists(), 'Ejecutar M06 · 0 para crear otra ejecución; no sobrescribir'
m6_ix.to_parquet(M6_DIR / 'auditoria/ventanas_m06.parquet', index=False)
m6_train_ix.to_csv(M6_DIR / 'auditoria/train_2010_2019.csv', index=False)
m6_dev_ix.to_csv(M6_DIR / 'auditoria/desarrollo_2020.csv', index=False)
m6_json('auditoria/datasets_autorizados.json', M6_SOURCE_MANIFEST)
m6_json('auditoria/verificacion_datos.json', M6_DATA_CHECKS)
display(m6_ix.groupby(['split', 'ny_hour']).size().rename('ventanas').to_frame())
print('DATOS_AUTORIZADOS_PREPARADOS', M6_DATA_CHECKS)
print('No se ha evaluado ni entrenado ningún candidato. Las fases I–VI siguen pendientes.')
