# AGENTS.md — AXXEL CAPITAL / Kronos

## 1. Entorno de trabajo

- Proyecto: `C:\Users\HP\OneDrive\Escritorio\modelo_SFT`
- Modelo: `NeoQuasar/Kronos-small`
- Tokenizador: `NeoQuasar/Kronos-Tokenizer-base`
- Notebook maestro: [Kronos_SFT_verificado.ipynb](https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss)
- Drive: `/content/drive/MyDrive/AXXEL_CAPITAL/`
- Herramientas: VS Code, Codex CLI, Python, PyTorch, Google Colab Pro.

## 2. Misión científica

Entrenar y validar Kronos-small para predecir los próximos cuatro cierres M15 de XAUUSD, superando benchmarks de persistencia mediante evidencia estadística reproducible.

La investigación predictiva precede al desarrollo de estrategias de trading.

## 3. Reglas obligatorias

- `RESEARCH_ONLY=true`; `orders_enabled=false`; `test_read=false`.
- Prohibido acceder al test 2023–2024 o ejecutar órdenes de trading.
- No modificar datasets, fuentes, checkpoints auditados ni informes históricos.
- No repetir las Fases I–IV ni experimentos equivalentes.
- No iniciar Fase V sin comprobar integridad, causalidad, ausencia de leakage, gradientes y predicciones del paso cero.
- Respetar el protocolo `M06_FV_JOINT_RETURN_V1` sin inventar parámetros.
- Guardar resultados nuevos y estados recuperables con hashes SHA-256.
- Mantener un solo escritor sobre el notebook maestro.
- No declarar superioridad predictiva sin pruebas estadísticas.

## 4. Inicio diario de Codex

Al iniciar cualquier sesión:

1. Leer `ESTADO_ACTUAL_KRONOS.md` y verificarlo con evidencias.
2. Comprobar directorio de trabajo, dependencias y cambios pendientes de Git.
3. Detectar el estado de Colab, GPU y Google Drive.
4. Reutilizar el runtime existente; no reiniciarlo por defecto.
5. Verificar modelos, checkpoints y experimentos activos.
6. Recuperar únicamente lo necesario, evitando ejecuciones duplicadas.
7. Informar del estado y la próxima tarea antes de entrenar.

Si Google solicita autorización, requerir intervención humana.

## 5. Recuperación y automatización

- Utilizar el controlador de inicio diario cuando esté implementado y probado.
- Priorizar conexiones oficiales compatibles; Playwright/CDP como alternativa controlada.
- Nunca duplicar celdas ni ejecutar automáticamente entrenamientos.
- Recuperar modelos desde checkpoints y estados persistidos, verificando integridad.
- No asumir que las variables siguen en memoria después de un reinicio.
- No depender del historial de conversación para reconstruir el proyecto.

## 6. Git y seguridad

- Versionar código, pruebas, configuraciones y documentación.
- Excluir secretos, credenciales, datos voluminosos y checkpoints de Git.
- Revisar los cambios antes de cada commit.
- No ejecutar `git push`, eliminar archivos ni sobrescribir históricos sin autorización.
- Evitar conflictos entre Git, OneDrive y Colab.

## 7. Cierre de jornada

1. Verificar persistencia de checkpoints, resultados y manifiestos.
2. Registrar avances, errores y experimentos todavía activos.
3. Actualizar `ESTADO_ACTUAL_KRONOS.md` únicamente con evidencia.
4. Documentar el último checkpoint válido y la próxima acción.
5. No interrumpir entrenamientos remotos activos automáticamente.

## Principio fundamental

**Cada nueva sesión debe poder recuperar el último estado científico válido de Kronos sin repetir experimentos, perder resultados ni depender del contexto anterior de Codex.**
## 8. Misión 08 — controlador diario

- Entrada: `.\scripts\axxel.ps1 start`; consultar `scripts/README.md`.
- Distinguir datos históricos de inspecciones del runtime actual.
- Nunca duplicar pestaña/celda/runtime para reparar conexión.
- `recover` respalda y muestra estado; no ejecutar código científico ni entrenar sin autorización expresa y nuevo preflight.
- `output/` y notebooks descargados se excluyen de Git; preservar manifiestos e informes auditados.

## 9. Inicio integrado autorizado (21-11-2026)

- El usuario autorizó implementar y probar conexión, prueba remota y recuperación real mediante `start`. Esta autorización reemplaza la limitación histórica de `start` a solo lectura de interfaz; no autoriza entrenamiento.
- `start` puede ejecutar únicamente las dos celdas propias con fuente fijada: `AXXEL_DAILY_HEALTH_V1` y `AXXEL_RUNTIME_RESTORE_V1`. Nunca ejecutar celdas históricas ni aceptar cambios de fuente automáticamente.
- El estado restaurado se publica en `AXXEL_SESSION`; modelos congelados, tres cabezas Huber, scalers y paquetes de reanudación/adaptadores. Cero pasos de optimizador. No confundir restauración con preflight completo de Fase V.
- `status` mantiene lectura de interfaz; `recover/close` mantienen diagnóstico/respaldo. Las evidencias anteriores a una lectura actual son HISTORICAL.
- Ante entrega incierta, recuperar el resultado de la transacción antes de enviar otra ejecución. La frescura se comprueba por cambio de `outputId`, no solo por relojes de Windows/Colab.
- Chrome CDP usa el perfil existente `C:\ChromeAutomationProfile`, puerto local 9222. No copiar cookies/secretos al repositorio. Si Google exige autorización, interviene el usuario.

## 10. Ejecución Fase V autorizada

- Después de READY_FOR_PHASE_V, el usuario autorizó explícitamente el siguiente paso: ejecutar las seis corridas del protocolo M06_FV_JOINT_RETURN_V1. No volver a solicitar autorización científica para esas mismas corridas.
- Usar la entrada explícita `automation/m10_execute_phase_v.py --authorize-phase-v`; nunca incorporarla al inicio diario. Consultar `scripts/M10_EJECUCION_FASE_V.md` y las transacciones locales antes de actuar.
- Una entrega incierta o un entrenamiento activo impiden reenviar. Reanudar solo desde checkpoints confirmados por commit.json y SHA-256; no repetir corridas completadas.
