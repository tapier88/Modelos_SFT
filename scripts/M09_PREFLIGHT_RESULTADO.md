# Misión 09 — BLOCKED

Revisión local: 2026-11-21 20:00 UTC. Ejecución GPU recuperada: 2026-11-21 15:14:31–15:15:41 UTC. Estas lecturas del kernel son evidencia de esa ejecución, no una consulta en vivo a las 20:00.

El último comando terminó con `PROBE_COMPLETED`; el resultado científico fue `NO_APTO`. No se encontró un error de transporte que requiriera corregir o reenviar el comando. Se recuperó `result.json` y se comparó íntegramente con la salida de la celda descargada. No se repitió la ejecución.

| Comprobación en GPU | Resultado |
|---|---|
| Infraestructura | Tesla T4, 15.359.344.640 bytes libres, PyTorch 2.14.1+cu130, CUDA 13.0; Drive montado |
| Identidad de pesos | Base y tokenizador coinciden con hashes del protocolo |
| Datos y objetivos | 8.428 ventanas train 2010–2019; 950 dev 2020 / 243 sesiones; índices auditados idénticos; H1–H4 alineados |
| Normalización y causalidad | Invariancia futura aprobada; diferencias causales eval/train 0; scalers recalculados exclusivamente con train coinciden |
| Paso cero desde precios | Tres cabezas reproducen referencias; diferencia máxima global 9,9209592e-7 unidades de precio |
| Arquitectura y gradientes | Control 33.092 parámetros; joint 164.164; paso cero idéntico; gradientes finitos cabeza/B; base/tokenizador sin gradientes |
| Reanudación sintética | Pérdida, índices y parámetros idénticos tras restaurar; gradiente A no nulo después del primer paso sintético |
| Conservación | Pesos base/tokenizador sin cambios; RNG restaurado; fuentes y salidas de las 63 celdas anteriores idénticas |
| Duplicados | Una sola celda M09; inventario remoto halló únicamente el protocolo propuesto, sin candidatos ambiguos dentro del alcance inspeccionado |

MAE reproducido por semilla: 271828 = 3,5551758106; 271829 = 3,5532714143; 271830 = 3,5526065591. Son referencias recuperadas, no resultados de entrenamiento de Fase V. Tolerancia aplicada: máximo 0,001 y media 0,00001 unidades de precio; el error observado quedó ampliamente por debajo de ambos límites. La tolerancia permite diferencias numéricas de inferencia por lotes, sin cambiar el objetivo ni las referencias.

## Bloqueos de aceptación

1. Falta ejecutar y certificar dos restauraciones de Fases III–IV en espacios de nombres vacíos, requisito explícito de `preflight_gates`. Reutilizar `AXXEL_SESSION` y probar una copia sintética no cumple este requisito.
2. El inventario de metadatos no certifica la deduplicación completa por huella científica: buscó identificadores y nombres de carpetas, pero no reconstruyó las huellas de métodos equivalentes con otros nombres.

El código agrega estos bloqueos explícitamente; no son fallos numéricos observados. Eliminarlos sin completar las verificaciones invalidaría la certificación. La prueba sintética realizó tres pasos desechables sobre copias; Fase V realizó cero pasos. No se accedió al test reservado. No se certifica ausencia universal de procesos de fondo mediante estas evidencias.

## Evidencia preservada

- Resultado: `output/axxel/20261121T1512053310924Z_m09/bootstrap_preflight/result.json`.
- SHA-256 del resultado: `2d93afbdc40012a5ac526c3a0397b45efdbd8ca2181bf340278d8edb6477fb08`.
- Verificación local de fuente, salida, unicidad y conservación: `output/m09_evidence_review_20261121T200047826068Z/evidence.json`.
- Payload ejecutado SHA-256: `609f16ce0ad204ffabafdfd44d14349e4c057974c9ae17d523d8748720e1fae1`.
- Artefactos remotos reportados: `/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/m09_preflight/20261121T151431978339Z/`.
- Manifiesto remoto reportado SHA-256: `800fd54cd746bcde889bd9ca5d032eadeb1c7215fdf37a0e73622874e6e698a4`. Se conserva el hash emitido por la ejecución; no se afirma una nueva lectura independiente de Drive.

Próximo trabajo científico: cubrir exclusivamente las dos puertas pendientes y cualquier fallo que revelen, preservando esta evidencia y sin repetir los experimentos I–IV. Hasta entonces, `BLOCKED`; no autorizar Fase V.
