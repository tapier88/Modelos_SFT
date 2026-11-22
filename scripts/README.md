# Inicio diario AXXEL / Kronos

Desde la raíz del proyecto en Windows PowerShell:

```powershell
.\scripts\axxel.ps1 start
```

`start` comprueba CDP 9222. Si no está disponible, abre Chrome con el perfil existente `C:\ChromeAutomationProfile` y la URL maestra. Reutiliza una única pestaña; espera la conexión de Colab y utiliza el botón Conectar cuando corresponde. Nunca reinicia ni elimina un runtime.

Después descarga un respaldo, comprueba que no haya una celda en ejecución y ejecuta exclusivamente dos celdas propias de contenido fijado por SHA-256. Las reutiliza por marcador e ID, sin ejecutar celdas históricas:

1. `AXXEL_DAILY_HEALTH_V1`: GPU, memoria, versiones, Drive y hashes de 22 checkpoints III–IV.
2. `AXXEL_RUNTIME_RESTORE_V1`: Kronos/tokenizador con revisiones fijas, tres cabezas Huber, scalers y paquetes de reanudación/adaptadores, publicados en `AXXEL_SESSION`. Si ya existe la sesión, comprueba los pesos base/tokenizador y la reutiliza.

La primera restauración comprueba las predicciones de las tres cabezas contra las referencias, utilizando únicamente las características de desarrollo 2020 guardadas. No reentrena, no abre datasets originales ni test 2023–2024. Los paquetes de optimizador/RNG se conservan en memoria; no se ejecutan pasos de optimizador ni se activa automáticamente el estado RNG para entrenar. Los adaptadores históricos se cargan como paquetes y no se aplican al backbone base. El preflight completo de Fase V sigue siendo una tarea distinta.

## Comandos

- `start`: conecta, prueba y restaura. Una incidencia produce `start_status=BLOCKED` y salida 2; éxito produce `RESTORED`.
- `status`: consulta la interfaz sin ejecutar celdas y muestra la última restauración local como HISTORICAL, con fecha y origen. No prueba por sí solo que esos objetos continúen vivos.
- `recover`: conserva el modo diagnóstico/respaldo; no ejecuta restauración.
- `close`: respalda sin detener trabajos ni desconectar Colab.

`start` tiene autorización explícita para las dos celdas fijadas; `remote_execution_enabled=true` significa solo ese alcance. `training_enabled=false` se mantiene para todos los comandos. Una ejecución pendiente se recupera por ID de celda y cambio de `outputId` antes de enviar otra. No se infiere frescura comparando exclusivamente relojes Windows/Colab. Se bloquean fuentes alteradas, celdas duplicadas, manifiestos incorrectos y procesos GPU ajenos antes de restaurar.

## Requisitos e intervención humana

Python con Playwright, Chrome y el perfil ya usado por el proyecto. No se instalan dependencias automáticamente. El perfil de Chrome está fuera del repositorio y no se copian sus cookies ni credenciales. Colab debe disponer de GPU; el controlador no fuerza cambios de tipo de runtime ni consume un reinicio para obtenerla.

Drive se verifica antes de cargar objetos. Si el runtime nuevo no tiene Drive montado, autoriza/monta Drive mediante Colab y vuelve a ejecutar `start`; no se automatizan los permisos de Google. La rama de OAuth/montaje desde cero no está certificada automáticamente. Cuotas, login vencido y recursos no disponibles son bloqueos externos explícitos, no motivos para repetir experimentos.

## Evidencia y validación

Cada ejecución guarda `output/axxel/<UTC>/session.json`, `bootstrap/` y `bootstrap_restore/`, con respaldo, transacción, comparación de fuente y resultado remoto. Se conservan los intentos fallidos y sus causas. `output/` está excluido de Git.

```powershell
powershell.exe -NoProfile -File tests/validate_axxel_syntax.ps1
python -m unittest discover -s tests -v
python -m unittest discover -s automation -p test_m07_integrity.py -v
```

Las pruebas locales usan simulaciones y no operan Colab. La certificación real se documenta por separado en `scripts/CERTIFICACION_INICIO_DIARIO.md`. Las certificaciones M08B/M08C describen etapas históricas y no sustituyen este procedimiento vigente.
