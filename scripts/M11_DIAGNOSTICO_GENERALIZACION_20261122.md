# M11 — Auditoría y propuesta controlada, 22-11-2026

**Dictamen:** Fase V no aporta evidencia de mejora predictiva. La degradación puntual asociada a continuar la cabeza domina la diferencia Joint–control. No está demostrado que la causa sea sobreajuste, adaptación insuficiente o desalineación de la pérdida. La generalización fuera de desarrollo 2020 sigue sin medirse de forma independiente.

RESEARCH_ONLY=true; orders_enabled=false; test_read=false. Esta misión ejecutó análisis local de resultados y lectura de interfaz; cero entrenamientos, cero inferencias nuevas y cero acceso a datasets/test. La propuesta siguiente está pendiente de ejecución. No modifica M06_FV_JOINT_RETURN_V1 ni autoriza repetirlo.

## Evidencia y alcance de la auditoría

Se ejecutó `python scripts/audit_m11_generalization.py`. Evidencia recuperable: `output/axxel/m11_audit_20261122T122015431003Z/`, con `audit.json`, hashes de entradas, respaldo del estado anterior y `checksums.json`.

Comprobaciones aprobadas:

- Resultado M10 SHA-256 `e78f0bbbf46ab0693ced209b1b2642bd6b23960e922b0774463a83a000b89321`.
- Nueve archivos del manifiesto de reconciliación, incluido notebook archivado y seis gráficas, íntegros.
- Protocolo, payload y fuentes locales de entrenamiento/evaluación coinciden con las fuentes fijadas; el payload se analizó como texto/AST, nunca se ejecutó.
- Una celda con el ID registrado contiene un resultado idéntico al JSON local. Es evidencia HISTORICAL del notebook descargado, no lectura nueva de sus salidas remotas.
- Seis combinaciones distintas brazo/semilla COMPLETED en paso 300; 78 registros históricos de checkpoints (0,25,…,300 por corrida). Estos mensajes no sustituyen verificar archivos y commits remotos.
- Medias, desviaciones entre semillas y nueve diferencias de MAE reconciliadas aritméticamente. Los nueve intervalos simultáneos contienen cero; gate=false.
- Scalers train-only locales íntegros respecto al hash del protocolo.

**Límite importante:** no se descargaron ni revalidaron en esta misión `predictions.npz`, `development_index.npz`, `completed.json` ni checkpoints de Fase V en Drive. Por ello no se recalcularon MAE/IC desde observaciones individuales, ni normas de pesos/adaptadores, ni distribuciones de retornos. La auditoría confirma consistencia de agregados, fuentes y archivos locales; no certifica de nuevo toda la cadena remota.

La lectura de interfaz del 22-11-2026 12:19:03 UTC encontró una única pestaña maestra, runtime conectado y backend GPU. GPU exacta, VRAM, montaje actual de Drive, modelos vivos y procesos de entrenamiento quedan UNKNOWN. Última restauración conocida: HISTORICAL, 01:05:38 UTC, T4 y tres cabezas. Registro actual: `output/axxel/20261122T1219024663942Z/`. Se preservó el primer intento fallido por permisos de Windows. No se reinició ni restauró el runtime.

Entorno local comprobado: Python 3.13.5, NumPy 2.2.6, pandas 2.3.3, Playwright 1.65.2. Había numerosos cambios pendientes de Git; se conservaron. No hubo commit ni push.

## Resultados y diagnóstico

MAE en unidades de precio XAUUSD; desarrollo 2020, 950 ventanas, cuatro horizontes y 243 sesiones NY. Persistencia: **3,560039287**.

| Semilla | Cabeza III original | Control final | Joint final | Control − original | Joint − control |
|---|---:|---:|---:|---:|---:|
|271828|3,555175813|3,577933976|3,578414298|+0,022758162|+0,000480322|
|271829|3,553271416|3,574401693|3,574432133|+0,021130277|+0,000030440|
|271830|3,552606561|3,560099197|3,560406587|+0,007492636|+0,000307390|

El incremento medio control–original es **0,017127025**, frente a **0,000272717** Joint–control (aproximadamente 63 veces menor). Son diferencias descriptivas; no se calculó un contraste nuevo de significación para control–original. Las cabezas originales fueron seleccionadas previamente sobre 2020, por lo que esta comparación está favorecida por selección.

Control medio: 3,570811622, un 0,3026% por encima de persistencia. Joint medio: 3,571084339, un 0,3103% por encima. Las desviaciones entre semillas (~0,00945) no son errores estándar sobre un holdout; tres semillas no son tres conjuntos temporales independientes.

| Semilla | IC simultáneo histórico de Joint − control |
|---|---:|
|271828|[−0,000402482; +0,001358124]|
|271829|[−0,000516980; +0,000579873]|
|271830|[−0,000288409; +0,000972339]|

Los intervalos proceden del bootstrap histórico pareado, bloques circulares de cinco sesiones, 10.000 réplicas y Bonferroni para nueve contrastes. No rechazar diferencia cero no demuestra equivalencia: no había margen de equivalencia registrado. El ajuste tampoco corrige las selecciones históricas sobre 2020. Dos IC95 sin ajuste frente a la mejor cabeza III excluyen cero hacia deterioro; los simultáneos no lo hacen. No seleccionar retrospectivamente esos contrastes para cambiar el dictamen.

Hallazgos que orientan el diagnóstico:

1. **Continuación:** ambos brazos empeoran frente a sus cabezas iniciales. El código crea AdamW nuevo, con momentos iniciales vacíos: es continuación de pesos, no reanudación del optimizador III. El efecto observado mezcla actualizaciones adicionales, reinicio del optimizador y selección favorable de las cabezas iniciales. No permite atribuir causalmente el deterioro a LoRA.
2. **Objetivo y métrica:** se optimiza Huber sobre retornos escalados, pero se decide por MAE de precios. Las escalas comprobadas son [16,284439; 21,376600; 25,753595; 29,261263] bps. Si `e` es error de precio, el residuo de la pérdida es `10000*e/(last_close*return_scale[h])`. Así, igual error en precio recibe distinta penalización según precio inicial y horizonte; el umbral de Huber en precio también varía. Es una desalineación matemática, no prueba de que explique el resultado. No se midió la proporción de residuos en cada rama de Huber.
3. **Horizontes y horas:** H1 y H2 empeoran frente a persistencia en las tres semillas Joint; H3/H4 son mixtos. A las 08:30 y 10:30 NY todas las diferencias Joint–persistencia son positivas; a las 09:30 y 11:30 son negativas. Es exploratorio y no justifica seleccionar horarios ni diseñar una estrategia.
4. **Adaptación:** diferencias pequeñas de MAE entre Joint y control no implican predicciones iguales, pesos inmóviles o LoRA inútil. El código exige gradientes finitos/no nulos; falta cuantificar cambios funcionales y normas efectivas `2*B@A` por capa. Comparar MAE agregados puede ocultar compensación entre errores.
5. **Curvas:** la gráfica histórica inspeccionada contiene pérdidas de minibatches variables y solo dos puntos de desarrollo (0/300). No hay una curva de validación intermedia que permita localizar sobreajuste; tampoco es válido comparar directamente un minibatch con todo desarrollo para estimar la brecha de generalización.
6. **Métricas auxiliares:** RMSE de los seis finales supera persistencia. La exactitud direccional Joint ronda 49,92–51,71%, sin inferencia estadística nueva. El 0% direccional de persistencia deriva de sign(0), no es una referencia binaria justa. El baseline Kronos greedy sigue UNAVAILABLE_UNVERIFIED.

2020 ya fue usado para selección; 2021–2022 tienen exposición histórica con L64 y otras ventanas, y no se deben presentar como holdout virgen ni mezclar con L56. En esta misión no se accedió a esos años. El test 2023–2024 permanece reservado. Con la evidencia actual no puede establecerse superioridad generalizable ni distinguir de forma concluyente cambio de distribución, ruido, sesgo de selección y sobreajuste.

## Propuesta: M11_ADAPTER_OFF_FIXED_HEAD_V1

**Experimento diagnóstico de inferencia, sin entrenamiento.** Pregunta: ¿qué efecto directo tiene el adaptador aprendido en el error, manteniendo exactamente la misma cabeza Joint final? Es distinto del contraste Joint–control, donde cambian tanto adaptador como cabeza. Esta propuesta es nueva y no altera los parámetros del protocolo V.

Diseño fijado:

- Tres semillas existentes: 271828, 271829, 271830. Únicamente checkpoints Joint comprometidos del paso 300; ninguna selección de checkpoint intermedio o nueva semilla.
- Pareja ON: cabeza Joint final y adaptador final. Pareja OFF: la misma cabeza y mismo backbone con contribución LoRA desactivada en una copia aislada. Tokenizador, scalers, precisión, entrada L56/H4 y orden de las 950 ventanas dev2020 idénticos; eval/inference_mode, dropout desactivado, todos los parámetros congelados.
- Reutilizar predicciones ON persistidas una vez verificadas. Reproducir ON solo como control de equivalencia numérica. Generar únicamente las tres predicciones OFF nuevas. Sin optimizador, backward, ajuste de escala, calibración ni búsqueda de coeficientes.
- Primaria por semilla: media pareada `abs(error_ON) − abs(error_OFF)` en precio. Negativo indica contribución favorable del adaptador condicionada a esa cabeza; positivo, desfavorable. No estima la calidad de volver a entrenar una cabeza sin adaptador.
- Inferencia propuesta: conservar unidad sesión NY, bloques circulares de 5, 10.000 réplicas, semilla bootstrap 271828. Familia nueva de tres contrastes: IC95 y Bonferroni con colas 0,05/(2×3), identificados como exploratorios sobre desarrollo reutilizado. No mezclar con los nueve contrastes históricos ni reutilizar sus intervalos.
- Secundarias descriptivas: media/RMS/máximo de `prediction_ON−prediction_OFF`, RMS de retorno predicho/real, MAE por horizonte y hora, sesgo, normas relativas de `2*B@A`. No convertir estos desgloses en nuevas hipótesis confirmatorias.
- Interpretación predefinida: tres límites superiores <0 indican contribución local consistente; tres límites inferiores >0 indican perjuicio local consistente; cualquier otro resultado se informa como mixto/inconcluso. Ningún resultado habilita una afirmación de generalización o entrenamiento automático. No afirmar equivalencia por un IC que cruce cero.

Antes de ejecutar, verificar commits y SHA-256 de los tres checkpoints, manifiestos de predicciones/índice y huella científica; identidad exacta de ventanas, sesiones, precios y orden; causalidad certificada y ausencia de actividad incompatible. La entrada debe permitir únicamente artefactos del experimento y datos autorizados hasta 2020, sin escanear ni abrir test. Deduplicar por hashes, estado de adaptador y especificación; reutilizar resultados existentes. No usar el ejecutor M10 para este diagnóstico.

Comprobar reproducción ON con las tolerancias históricas (máximo ≤0,001 y media ≤0,00001 de diferencia en precio), pero no asumir que bastan para efectos pequeños: guardar la discrepancia observada y compararla con el efecto ON–OFF. Si el efecto no puede distinguirse del error numérico, dictamen numéricamente inconcluso. Conservar predicciones por ventana, métricas, IC, configuración, versiones y hashes en carpeta nueva con manifiesto; verificar lectura posterior. Mantener un solo escritor del notebook.

No se propone ampliar pasos ni una nueva corrida de entrenamiento ahora. La siguiente decisión sobre una ablación de pérdida en espacio de precios debe esperar a este diagnóstico y a un protocolo separado con evaluación temporal no seleccionada, presupuesto y criterio de decisión registrados antes de ejecutarlo.
