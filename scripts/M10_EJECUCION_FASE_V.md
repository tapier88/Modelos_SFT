# M10 — ejecución autorizada de Fase V

El usuario autorizó el siguiente paso después de la certificación READY_FOR_PHASE_V. El alcance es `M06_FV_JOINT_RETURN_V1`: dos brazos, tres semillas, 300 pasos adicionales por corrida. No se autoriza modificar datasets, acceder al test ni repetir I–IV.

Entrada explícita: `python automation/m10_execute_phase_v.py --authorize-phase-v --output <directorio_nuevo>/bootstrap_phase_v`.

Este comando no forma parte de `scripts/axxel.ps1 start`. Reutiliza el transporte existente, respalda el notebook, verifica su identidad y la fuente fijada, y rechaza actividad o transacciones inciertas. No reenviar mientras haya una ejecución activa. Una transacción incierta se inspecciona y recupera antes de cualquier nuevo envío.

La carpeta remota propuesta es `<canonical_run>/fase_V_joint_v1/<scientific_fingerprint>/`. Cada brazo/semilla tiene checkpoint inicial, checkpoints cada 25 pasos y estado COMPLETED verificable. Cada checkpoint conserva cabeza, adaptadores, optimizador, paso, RNG Python/NumPy/torchCPU/CUDA y trazas del orden de ventanas. `commit.json` se escribe después de guardar y releer el checkpoint; sus hashes deben verificarse para reanudar. Una carpeta incompleta o un conflicto de configuración detiene la ejecución. Los archivos previos no se sobrescriben.

Se usan tokens causales de los 56 pasos de contexto. El control reutiliza características del backbone congelado; el joint recalcula sus representaciones con autograd habilitado. La normalización de características replica los redondeos float32 de los scalers históricos. Las dos ramas reciben los mismos índices PCG64 por semilla; se comparan sus trazas al finalizar. La cabeza parte del best auditado de esa semilla, con un AdamW nuevo según el protocolo. No se cargan adaptadores M05/IV como inicialización de Fase V.

Desarrollo se evalúa únicamente en paso0 y paso300. El bootstrap pareado usa 10.000 réplicas y bloques circulares de cinco sesiones NY, con intervalos simultáneos para nueve contrastes. El informe distingue evidencia de desarrollo reutilizado de superioridad predictiva independiente. No se emplea el test reservado.

Validación local previa: 24 pruebas en tests y 5 de integridad M07 aprobadas; parser PowerShell aprobado. Incluyen normalización numérica, identidad inicial LoRA, gradientes A/B, reanudación exacta del optimizador/RNG sobre una copia CPU sintética, acumulación de cuatro microbatches, rechazo de sobrescritura y bloqueo de entrenamiento sin autorización explícita. No sustituyen los controles GPU ejecutados antes de optimizar.

Fuentes: `m06_staging/m10_phase_v.py`, `m06_staging/m10_evaluation.py`. Ensamblado y hashes: `automation/m10_build_phase_v.py`, `m06_staging/m10_source_manifest.json`. Estado real y métricas deben obtenerse del resultado remoto, no inferirse de esta documentación.
