# Inicio diario: certificación real del 21-11-2026

Se comprobó la restauración de Kronos desde Drive y su reutilización con `scripts/axxel.ps1 start`. No se repitieron entrenamientos ni experimentos I–IV, no se inició Fase V ni se abrió el test reservado.

## Resultado actual

| Componente | Estado | Fecha UTC / evidencia |
|---|---|---|
| Notebook maestro | CURRENT, identidad exacta y única | 21-11 14:58; controlador |
| GPU | CURRENT, Tesla T4, 15.360 MiB total, 14.648 MiB libres | 14:57:57; nvidia-smi en prueba remota |
| PyTorch / CUDA | CURRENT, 2.14.1+cu130 / 13.0 | 14:58:08; restauración y operaciones GPU completadas |
| Drive / AXXEL_CAPITAL | CURRENT, montado y accesible | 14:57:57; consulta desde kernel |
| Manifiestos III–IV / checkpoints | CURRENT, ambos manifiestos y 22 checkpoints coinciden con SHA-256 esperado | 14:57:57; health/result |
| AXXEL_SESSION | CURRENT, restaurado y luego reutilizado | 14:57:16 `reused=false`; 14:58:08 `reused=true` |
| Predicciones de cabezas | Verificadas 14:57:16; diferencias máxima y media 0,0 para tres semillas | Comparación sobre 950 características de desarrollo 2020 guardadas |
| Procesos de entrenamiento generales | UNKNOWN fuera del control de este procedimiento | No se confunde ausencia de procesos GPU ajenos con ausencia universal de trabajos CPU |
| Preflight científico completo de Fase V | Pendiente | No se habilita entrenamiento |

`AXXEL_SESSION` contiene backbone y tokenizador congelados, tres cabezas Huber, scalers y paquetes de reanudación/adaptadores. Los paquetes Adam/RNG se cargan en memoria, no se activan globalmente para entrenar. Los adaptadores IV no se aplican al backbone base. En reutilización se comprueban hashes del backbone/tokenizador y se conservan los objetos; la equivalencia de cabezas pertenece a la restauración anterior, no a una nueva evaluación completa.

## Evidencia recuperable

- Restauración actual: `output/axxel/20261121T1456393286790Z/bootstrap_restore/result.json`; SHA256 `ca56f0b33c2e0a1af6ef914175b95a0ff65f20eb999486fcf3569d24be8b4074`.
- Reutilización: `output/axxel/20261121T1457513684605Z/bootstrap_restore/result.json`; SHA256 `5ad6c79dcff0cacd04d2923d208a75305b07c5651125dd41c54af59f09638f9e`.
- GPU/Drive/checkpoints: `output/axxel/20261121T1457513684605Z/bootstrap/result.json`; SHA256 `81afa2d494f035582ca666f5ed396f115b945fbb33d9614307336eb72a9b3e56`.
- Certificación, comparación de notebook, logs y copia de memoria anterior: `output/daily_certification_20261121T145658764458Z/`.

Base hash: `a3c79ed2ca1f5b35c921fce7cf673db515dd7593e5552c643a977f8dc96ae6a9`.
Tokenizador: `e7885d9d7874cde53113b1cc43cbdf8f05941fb8d95bce0c38d03183665e4740`.
Manifiesto III: `731d1c5412d4836e16961ff2e3747e6fd4da124db81eaf20265a638b6c90d7a1`.
Manifiesto IV: `e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc`.

## Pruebas completadas

- Parser nativo de PowerShell aprobado y **24/24 pruebas locales** (19 controlador/clasificación/seguridad y 5 de integridad M07).
- Dos inicios completos consecutivos: restauración real y reutilización. Ambos conservaron 63 celdas, una por cada marcador gestionado. Se comprobó igualdad de fuentes y salidas de las 61 celdas históricas.
- Recuperación real tras pérdida del enlace local: CDP devolvía ECONNREFUSED; `start` abrió Chrome con el perfil existente y recuperó el notebook. No se simuló ese fallo. La sesión actual tenía otra GPU UUID y no contenía los objetos anteriores; se reconstruyeron desde pesos/revisiones fijos y checkpoints auditados.
- Se recuperó una salida ya ejecutada tras fallo de descarga sin reenviarla. Se corrigió la carrera de carga del control de conexión y el desfase de unas décimas entre reloj local/Colab. Un `outputId` nuevo acredita una salida nueva; los relojes quedan como procedencia.
- Las descargas de interfaz pueden reintentarse; el envío de ejecución no se reintenta si su entrega es incierta. Fuente alterada, duplicados y hashes incorrectos bloquean el flujo.

No se forzó un reinicio ni se destruyó una sesión para probar pérdida del backend. La reconexión con el botón de Colab está implementada, pero la recuperación real observada se produjo al abrir Chrome y esperar la conexión de la página. No se certifica un ciclo manual de desconexión del backend. La autorización OAuth/montaje desde cero de Drive sigue siendo intervención humana si Google la solicita; no se probaron permisos nuevos porque Drive estaba disponible.

## Uso y límites

Ejecutar `.\scripts\axxel.ps1 start`. Consultar `session.json`: `start_status=RESTORED` acredita recuperación; `training_enabled=false` sigue vigente. `status` no ejecuta celdas y etiqueta los últimos resultados guardados como HISTORICAL.

La siguiente misión científica debe comprobar las condiciones pendientes de Fase V, preparar datos autorizados y validar reanudación de optimizador/RNG y deduplicación científica. Restauración diaria no equivale a autorización para comenzar las seis corridas. Las comparaciones descritas no prueban superioridad frente a persistencia.
