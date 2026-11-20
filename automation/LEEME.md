# Control de Kronos en Colab mediante Playwright y CDP

## Resultado de la prueba realizada

El script completo `controlar_kronos_colab.py` se ejecutó en el entorno local de esta tarea.
Se detuvo en la comprobación previa de `http://localhost:9222` con:

`WinError 10013: Intento de acceso a un socket no permitido por sus permisos de acceso`.

Playwright tampoco está instalado en el intérprete Python utilizado. El error de socket no permite determinar si Chrome está escuchando en 9222: no demuestra que el puerto esté cerrado ni identifica por sí solo qué componente impone la restricción. No se cambiaron permisos, firewall ni configuración de Chrome.

No se conectó al navegador ni se ejecutaron celdas en esta prueba. `resultado_control_colab.json` conserva el resultado real. Pasaron las comprobaciones locales de sintaxis, identidad de URL, rechazo de cambios de código y rechazo de resultados antiguos. Los selectores de la interfaz de Colab siguen pendientes de prueba en vivo y pueden necesitar adaptación si Google cambia la interfaz.

## Alcance

1. Comprueba que el endpoint sea local y que CDP publique el cuaderno exacto.
2. Conecta con `chromium.connect_over_cdp`, comprueba URL y título, y trae la pestaña al frente.
3. Descarga una copia temporal del cuaderno mediante Archivo > Descargar > .ipynb para verificar su código actual y sus salidas.
4. Compara las celdas con las huellas del cuaderno revisado el 9 de noviembre de 2026. Se detiene si cambian; no modifica ni acepta automáticamente código distinto.
5. Si se solicita `--restart-runtime`, reinicia la sesión. Esto descarta las variables en memoria; no borra los archivos de Drive.
6. Ejecuta las secciones 1 a 8 individualmente, usando Control+Enter cuando encuentra su editor o el botón de ejecutar de esa misma celda.
7. La sección 3 extrae el pipeline embebido, verifica su integridad e importación y confirma su ruta. No es necesario subir el archivo local por separado.
8. Comprueba metadatos de ejecución nuevos y las salidas reales. No considera éxito las salidas guardadas de una ejecución anterior.
9. Confirma un run_id nuevo, las verificaciones de datos y tres pasos de entrenamiento en M5 y M15. Guarda diagnóstico local y deja Chrome abierto.

La sección 8 conserva `RUN_FULL=False`. La sección 9 no se ejecuta. Esto repite la prueba técnica corta: **no implementa ni ejecuta la evaluación ampliada pendiente**.

## Requisitos

- Python 3.10 o posterior.
- Chrome con el endpoint CDP local ya disponible y accesible desde el proceso Python.
- Una sola pestaña del cuaderno `1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss`.
- Colab autenticado y las autorizaciones de Drive necesarias completadas por el usuario.

El programa no abre un puerto de depuración, no lee credenciales y no cambia protecciones del sistema. Si la conexión está prohibida, se detiene. Los cuadros de autorización requieren intervención del usuario.

## Uso en PowerShell

Desde la carpeta que contiene el script, con el endpoint disponible:

```powershell
python -m pip install playwright
python .\controlar_kronos_colab.py --check-only
```

No hace falta `playwright install`: se utiliza el Chrome existente.

Si la comprobación termina correctamente y no hay un trabajo activo en Colab:

```powershell
python .\controlar_kronos_colab.py --restart-runtime
```

El reinicio es necesario al repetir estas secciones, porque la sección 2 crea una carpeta nueva y la sección 3 verifica que el módulo importado provenga de esa carpeta. Un módulo de una ejecución anterior puede permanecer en la memoria del runtime.

El límite predeterminado es de 30 minutos por celda y se revisa cada 10 segundos. Un tiempo agotado o Ctrl+C detiene el monitor local: la celda puede continuar en Colab. El programa no reenvía ejecuciones automáticamente.

Un cambio de código o de interfaz provoca una detención explícita; se debe revisar la diferencia antes de adaptar las huellas o los selectores.

## Documentación

- https://playwright.dev/python/docs/api/class-browsertype#browser-type-connect-over-cdp
- https://playwright.dev/python/docs/api/class-page#page-bring-to-front
- https://playwright.dev/python/docs/api/class-keyboard#keyboard-press
