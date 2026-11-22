# M08C — sesión conectada; diagnóstico del kernel pendiente de autorización

Lectura actual: 2026-11-21T04:40:55.398645+00:00 (20-11-2026 Bogotá). Se reutilizó `scripts/axxel.ps1 status`, sin modificar el controlador ni repetir pruebas M08B o auditoría M07.

| Componente | Estado | Evidencia |
|---|---|---|
| Notebook | CURRENT: único y exacto | ID `1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss`; título Kronos_SFT_verificado.ipynb |
| Runtime | CURRENT: conectado según interfaz | «Conectado a (GPU) del backend de Google Compute Engine» |
| GPU | CURRENT: backend GPU; modelo UNKNOWN | Control de conexión; no consulta nvidia-smi |
| RAM | CURRENT: 1,56/12,67 GB usados/totales | Lectura redondeada de interfaz; no es VRAM |
| VRAM, CUDA, PyTorch | UNKNOWN | Requieren diagnóstico del kernel |
| Drive y carpeta AXXEL_CAPITAL | UNKNOWN | No se ejecutó montaje ni consulta de filesystem remoto |
| Procesos y entrenamientos | UNKNOWN | No se observó indicador positivo; no puede descartarse actividad de fondo |
| Modelos cargados | UNKNOWN | No se consultaron objetos del kernel |
| Checkpoints III–IV | HISTORICAL; existencia actual UNKNOWN | M07 auditó III70/70 e IV37/37; sin repetirlo hoy |

El runtime ya estaba conectado en la comprobación actual. No se pulsó reconectar, reiniciar, ejecutar ni montar Drive. La lectura anterior de M08B era desconectada; no se infiere quién o qué restableció la conexión. No se declara READY.

## Evidencias y hashes

- Lectura actual: `output/axxel/20261121T0440548200924Z/readonly_status.json`; SHA-256 `38142297f0a2fa9d15bd4d7833220a2c8eb76e0a177521c288414d7ad5b2cc5f`.
- Controlador sin cambios: SHA-256 `2b4accfff31394b76213f2063922ede656dce30628fd6733a84530dc8734262c`.
- Manifiesto III histórico: `731d1c5412d4836e16961ff2e3747e6fd4da124db81eaf20265a638b6c90d7a1`.
- Manifiesto IV histórico: `e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc`.
- Procedencia histórica: `m06_staging/m07_reconciliation_20261120T224952Z/remote_inventory.json`; cierre M07 2026-11-20T22:59:16Z. Las rutas canónicas siguen en el estado del proyecto.
- Copia del estado previo y hashes locales: `output/m08c_20261121T044216325917Z/`.

## Bloqueo y propuesta concreta de diagnóstico

El usuario exige autorización explícita para toda nueva celda diagnóstica. Se preparó **sin insertar ni ejecutar** `m06_staging/m08c_readonly_diagnostic.py`, SHA-256 `890b369671fa0a424f2389b3d12a9c33c5ef22f6ec389cd26501234cb1a34620`. Solo se validó su sintaxis local con AST; su comportamiento remoto no está certificado.

La propuesta consulta nvidia-smi, memoria y nombres de procesos; versiones instaladas/cargadas sin importar torch ni inicializar CUDA; nombres/tipos de objetos existentes; montaje y acceso de Drive; hashes de los dos manifiestos fijados y exclusivamente checkpoints `.pt` declarados dentro de cada fase. No carga checkpoints, no lee datasets, no monta Drive, no instala dependencias ni escribe archivos remotos. Imprime un resultado JSON en la salida de la nueva celda. Los nombres de objetos no acreditan compatibilidad; procesos de fondo y usabilidad de CUDA pueden seguir UNKNOWN.

Si se autoriza: volver a verificar identidad y estado; no poner la celda en cola si hay ejecución activa; comprobar que el marcador no existe; preservar el notebook y comparar fuente exacta antes de ejecutar una sola vez. Si Drive no está montado, detenerse para la intervención correspondiente. Si se desconecta y no puede descartarse actividad, solicitar la autorización de reconexión exigida por M08C. No incorporar un ejecutor a `start/status/recover`.

## Preparación de M08D — no iniciada

Entrada requerida: resultado autorizado de M08C con GPU/Drive comprobados, manifiestos coincidentes y checkpoints localizados; resolución de procesos activos o evidencia insuficiente. Después fijar revisiones de Kronos/tokenizador, hashes de pesos, cabezas y adaptadores, y configuraciones auditadas.

Restauración futura: objetos aislados, carga estricta y recuperación de optimizador/RNG solo cuando corresponda, sin repetir I–IV; demostrar equivalencia paso cero y causalidad con las tolerancias auditadas y datos autorizados. Persistir evidencia nueva y estado de recuperación. No restaurar ahora, no entrenar Fase V ni acceder al test 2023–2024.
