# Fase V — seis corridas completadas y reconciliadas

Reconciliación: 22-11-2026 12:04 UTC / 07:04 Bogotá. Resultado de entrenamiento capturado el 22-11-2026 02:03:30 UTC.

El resultado local y la salida de la única celda `M10_PHASE_V_JOINT_RETURN_V1` del notebook maestro coinciden íntegramente. Las seis corridas tienen estado COMPLETED y paso300. No se reenvió ninguna ejecución durante esta reconciliación.

## Comunicación y bloqueo

- `http://127.0.0.1:9222/json/version` responde. El fallo histórico fue un timeout de Playwright tras conectar el WebSocket; no demuestra bloqueo de puerto/firewall.
- El controlador usa `timeout=60000` para CDP. Se verificó una conexión real con ese valor.
- Colab quedó visible, con `windowState=normal`. La entrada explícita de Fase V restaura ese estado y lleva la pestaña al frente antes de actuar.
- `output/axxel_start.lock` ya no existía y no había un controlador Python de Fase V activo. No fue necesario borrar otro archivo ni tocar bloqueos remotos.
- El runtime de la interfaz estaba conectado. No se reinició Colab ni se ejecutaron celdas para obtener esta evidencia.

## Resultado científico

Desarrollo2020:950 ventanas/243 sesionesNY;L56,H4,M15. MAE de persistencia: **3,560039287**. Mejor referencia HuberIII guardada: **3,552606561**, seleccionada previamente sobre este mismo desarrollo.

| Semilla | Control Huber MAE | Joint QV+Huber MAE | Joint−control |
|---|---:|---:|---:|
|271828|3,577933976|3,578414298|+0,000480322|
|271829|3,574401693|3,574432133|+0,000030440|
|271830|3,560099197|3,560406587|+0,000307390|

MAE medio±desviación entre semillas: control **3,570811622±0,009443849**; joint **3,571084339±0,009459133**. Las tres semillas joint tienen MAE mayor que persistencia, que la mejor cabezaIII y que su control emparejado. Los intervalos simultáneos de las nueve comparaciones incluyen cero. El criterio predefinido de seguimiento devuelve **false**. No hay evidencia de superioridad predictiva de Joint LoRA+Huber bajo este protocolo; tampoco corresponde afirmar deterioro estadísticamente significativo en todos los contrastes.

Los intervalos proceden de bootstrap pareado por sesiones, bloques circulares de5 sesiones,10.000 réplicas y ajuste Bonferroni para9 contrastes. Desarrollo2020 ya se había utilizado en selecciones anteriores; no es un holdout independiente. El baseline greedy de Kronos figura como `UNAVAILABLE_UNVERIFIED` en esta evaluación: no se incorpora una comparación nueva no verificada. La exactitud direccional de persistencia es0 por la convención sign(0) frente a movimientos no planos, no porque prediga la dirección contraria.

El resultado incluye RMSE, sesgo, error direccional, MAE por horizonte, diferencias por horaNY y dispersión por semilla. Se generaron seis gráficas reales en Colab; se exportaron sus representaciones PNG desde las salidas del notebook a la carpeta local de reconciliación.

## Evidencias y recuperación

- Resultado: `output/axxel/m10_phase_v_20261122/bootstrap_phase_v/result.json`.
- SHA256 del resultado: `e78f0bbbf46ab0693ced209b1b2642bd6b23960e922b0774463a83a000b89321`.
- Evidencia actual, notebook descargado, seis gráficas y hashes: `output/axxel/m10_reconciled_20261122T120350/`.
- Huella científica: `f7efc796337d3684b0641688c1fa98eca67d0a01814d714ce99cce78a49f1213`.
- Carpeta persistente reportada: `/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/fase_V_joint_v1/f7efc796337d3684b0641688c1fa98eca67d0a01814d714ce99cce78a49f1213/`.
- Evaluación: subcarpeta `evaluation/20261122T020302277505Z/`; manifiesto SHA256 `0f02af8aeefc2f794f5b39516835b471fec36484b1c920c300a12c2a285427f4`.

La ejecución guardó checkpoints/optimizador/RNG y manifiestos según el código fijado. En esta reconciliación se contrastó la salida actual del notebook; no se afirma una nueva relectura de todos los archivos de Drive a las12:04UTC. Antes de una futura restauración, verificar sus commits y hashes. No repetir estas seis corridas.

Comprobaciones locales de este cambio: sintaxis Python y seis pruebas de seguridad del transporte aprobadas. Los29tests y parser previos constan en la preparación. No hubo acceso al test2023–2024, órdenes, commit ni push.

Próxima misión sugerida: diagnosticar la escala de retornos, el efecto de continuar la cabeza y la magnitud efectiva de adaptación usando estos checkpoints y predicciones, antes de proponer una nueva ablación. No ampliar pasos ni buscar otra semilla para mejorar retrospectivamente este resultado.
