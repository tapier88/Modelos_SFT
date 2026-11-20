#@title MISIÓN 06 · 0. Estado real del runtime y procedencia (sin entrenamiento)
# Ejecutar esta sección solamente: las recuperaciones históricas de M05 cargan 2021–2022.
import sys, os, json, hashlib, platform, uuid, importlib.metadata as m6_metadata
from pathlib import Path
from datetime import datetime, timezone
from IPython.display import display, Markdown

if 'google.colab' not in sys.modules:
    import importlib.util
    if importlib.util.find_spec('google.colab') is None:
        raise RuntimeError('M06 debe ejecutarse en el cuaderno principal de Google Colab.')

import torch
from google.colab import drive, _message
M6_BASE = Path('/content/drive/MyDrive/AXXEL_CAPITAL')
M6_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
M6_HW = {
    'observed_at_utc': datetime.now(timezone.utc).isoformat(),
    'python': platform.python_version(), 'torch': torch.__version__,
    'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
    'device': str(M6_DEVICE), 'cpu_count': os.cpu_count(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'gpu_total_GiB': torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None,
}
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    M6_HW.update(gpu_free_GiB=free / 2**30, gpu_allocated_GiB=torch.cuda.memory_allocated() / 2**30)
try:
    import psutil
    M6_HW.update(ram_total_GiB=psutil.virtual_memory().total / 2**30,
                 ram_available_GiB=psutil.virtual_memory().available / 2**30)
except ImportError:
    M6_HW['ram_measurement'] = 'psutil no instalado; no se infiere memoria'
print(json.dumps(M6_HW, indent=2))

if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')
if not M6_BASE.is_dir():
    raise FileNotFoundError(M6_BASE)
M6_RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
M6_DIR = M6_BASE / 'MISION06_ALINEACION' / M6_RUN_ID
M6_DIR.mkdir(parents=True, exist_ok=False)
for sub in ['auditoria', 'configuraciones', 'metricas', 'graficas', 'checkpoints', 'informes']:
    (M6_DIR / sub).mkdir()

def m6_json(relative, value):
    target = (M6_DIR / relative).resolve()
    if M6_DIR.resolve() not in target.parents:
        raise ValueError('Escritura fuera de la ejecución M06')
    # Cada ejecución es nueva; no sobrescribir archivos ni artefactos históricos.
    with target.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)

def m6_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

m6_json('auditoria/hardware_inicial.json', M6_HW)
M6_VERSIONS = {}
for package in ['torch', 'numpy', 'pandas', 'pyarrow', 'safetensors', 'huggingface_hub', 'einops', 'matplotlib', 'scipy', 'scikit-learn']:
    try:
        M6_VERSIONS[package] = m6_metadata.version(package)
    except m6_metadata.PackageNotFoundError:
        M6_VERSIONS[package] = None
m6_json('auditoria/versiones.json', M6_VERSIONS)
M6_HIST = M6_BASE / 'Kronos_SFT_verificado_20261107/ejecuciones/20261112T022707Z_55c83b'
M6_M05 = M6_BASE / 'MISION05_LORA/20261112T130149Z_80e373/continuacion_20261113T203705Z_d2f8d8'
M6_CHECKPOINT = M6_M05 / 'checkpoints/seed_271828/step_1000.pt'
M6_FILES = [M6_HIST / 'kronos_pipeline.py', M6_HIST / 'vendor/model/kronos.py',
            M6_HIST / 'vendor/model/module.py', M6_CHECKPOINT]
M6_PROVENANCE = []
for path in M6_FILES:
    if not path.is_file():
        m6_json('auditoria/bloqueo_procedencia.json', {'missing': str(path), 'experiments_executed': False})
        raise FileNotFoundError(path)
    M6_PROVENANCE.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': m6_sha(path)})
assert M6_PROVENANCE[0]['sha256'] == '1424fb451942c7e1e1823808405f428283f9b1aacf7b34d08df7ca9edb503016'
m6_json('auditoria/procedencia.json', M6_PROVENANCE)
M6_POLICY = {'L': 56, 'H': 4, 'seed': 271828, 'train_years': list(range(2010, 2020)),
             'development_years': [2020], 'allowed_source_years': list(range(2010, 2021)),
             'ny_anchors': ['08:30', '09:30', '10:30', '11:30'],
             'normalization': 'float32; solo contexto; std(ddof=0)+1e-5; clip [-5,5]',
             'research_only': True, 'orders_enabled': False,
             'historical_M05_artifacts': 'read_only', 'status': 'PREFLIGHT_ONLY'}
m6_json('configuraciones/politica.json', M6_POLICY)
# Copia de recuperación dentro de M06; no es un cuaderno de trabajo alternativo.
try:
    m6_json('auditoria/notebook_al_inicio.ipynb', _message.blocking_request('get_ipynb', timeout_sec=30)['ipynb'])
except Exception as exc:
    m6_json('auditoria/respaldo_no_disponible.json', {'error_type': type(exc).__name__, 'error': str(exc)})
M6_PREFLIGHT_OK = True
display(Markdown('**M06: diagnóstico inicial ejecutado.** No se han cargado datasets ni entrenado modelos.'))
print('EJECUCION_M06', M6_DIR)
print('VERSIONES', M6_VERSIONS)
print('CHECKPOINT_IDENTIFICADO_Y_HASH_CALCULADO', M6_CHECKPOINT)
print('GPU mayor no solicitada. Las fases predictivas siguen pendientes.')
