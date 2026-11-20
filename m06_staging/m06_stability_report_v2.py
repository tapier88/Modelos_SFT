#@title MISIÓN 06 · ESTABILIDAD. Evidencias, incidencias y puerta de continuación
import json,os,hashlib
from pathlib import Path
from datetime import datetime,timezone
from google.colab import _message
assert 'M6_OPERATIONAL_TESTS' in globals() and all(M6_OPERATIONAL_TESTS.values()), 'Pruebas operativas pendientes'
M6_BROWSER_SAVE_EVIDENCE=json.loads("{\"historical40\":true,\"loader\":true,\"probe\":true,\"modified\":\"2026-11-15T13:57:58.756Z\",\"probe_id\":\"v5gIUkiygBio\",\"cell_count\":52}")
M6_EDITOR_EVIDENCE={'initial_principal_editors':2,'principal_editors_after_backup':1,
 'retired_target':'89ED27EAA8A371D85EB23F0FB40ABDD7','active_target':'849BB1163F4C36B903D74F4775670F3E',
 'backup_before_navigation':True,'main_notebook_API_writes_disabled':True,
 'scope':'CDP 9222 verificado; no garantiza ausencia de clientes externos no observables'}
backup_root=m6_stable.ROOT.parent
backup_files={'primary':backup_root/'RESPALDO_ESTABILIZACION_PRIMARY_20261115.ipynb',
 'secondary':backup_root/'RESPALDO_ESTABILIZACION_SECONDARY_20261115.ipynb'}
expected={'primary':'cd32cc8e9e83780ee382cb45944138c9202365bc99da61872a62c537fb5170ae',
 'secondary':'ac4e662f0d74d154bc2b2dbe82c6126d944b698783031b1233e45673d3319651'}
for name,p in backup_files.items(): assert m6_stable.sha(p)==expected[name]
versions={k:json.loads(p.read_text()) for k,p in backup_files.items()}
def cid(c): return c.get('id') or c.get('metadata',{}).get('id')
def h(v): return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
a={cid(c):c for c in versions['primary']['cells']}; b={cid(c):c for c in versions['secondary']['cells']}
comp=[]
for key in dict.fromkeys(list(a)+list(b)):
    x,y=a.get(key),b.get(key)
    row={'id':key,'primary':x is not None,'secondary':y is not None}
    for f in ['source','metadata','outputs','execution_count']:
        row[f+'_equal']=bool(x is not None and y is not None and x.get(f)==y.get(f))
        row[f+'_primary_sha']=h(x.get(f)) if x is not None else None
        row[f+'_secondary_sha']=h(y.get(f)) if y is not None else None
    comp.append(row)
assert all(all(row[f+'_equal'] for f in ['source','metadata','outputs','execution_count']) for row in comp[:40])
m6_stable.atomic_json(m6_stable.OPS/'notebook_versions_comparison.json',comp)
m6_stable.atomic_json(m6_stable.OPS/'backup_hashes.json',{k:{'path':str(p),'sha256':m6_stable.sha(p),'bytes':p.stat().st_size} for k,p in backup_files.items()})
M6_STABILITY_CHECKS={'A_save_and_remote_readback':all(M6_BROWSER_SAVE_EVIDENCE[k] for k in ['historical40','loader','probe']),
 **M6_OPERATIONAL_TESTS,'single_visible_principal_writer':M6_EDITOR_EVIDENCE['principal_editors_after_backup']==1,
 'backups_verified':all(m6_stable.sha(p)==expected[k] for k,p in backup_files.items()),
 'phase_I_recovered':M6_PHASE_I_DONE,'hashes_and_cache_evidence':M6_RECOVERY_EVIDENCE['original_weights_unchanged']}
M06_ENVIRONMENT_STABLE=all(M6_STABILITY_CHECKS.values())
assert M06_ENVIRONMENT_STABLE
state={'M06_ENVIRONMENT_STABLE':M06_ENVIRONMENT_STABLE,'checks':M6_STABILITY_CHECKS,'pid':os.getpid(),
 'utc':datetime.now(timezone.utc).isoformat(),'recovery':M6_RECOVERY_EVIDENCE,
 'browser_save':M6_BROWSER_SAVE_EVIDENCE,'editors':M6_EDITOR_EVIDENCE,
 'writer_protocol':'Colab UI only; independent artifacts with locks, hashes, fsync, replace and readback',
 'limits':['No garantia absoluta frente a editores externos ajenos a CDP; revisar pestañas y cambios remotos antes de escribir',
 'Atomicidad de os.replace depende de Drive FUSE; se verifican contenido, tamaño y hash tras cada escritura',
 'El manifiesto experimental es una línea base de integridad; nuevos experimentos requieren una nueva revisión autorizada y archivada']}
m6_stable.atomic_json(m6_stable.OPS/'environment_status.json',state)
incidents=[
 {'incident':'Concurrent notebook editors','evidence':'2 pestañas del mismo ID; secundaria con guardado bloqueado desde 22:30','fix':'respaldo completo de ambas; segunda sustituida por visor del respaldo; solo Colab modifica principal'},
 {'incident':'RAM volatile after disconnect','evidence':'assert M6_PHASE_I_DONE ausente; nuevo runtime observado','fix':'recuperación desde archivos en espacio de variables vacío; B y D ejecutadas'},
 {'incident':'Non-finite JSON','evidence':'ValueError Out of range float values nan; N/A de persistencia convertido por pandas a NaN','fix':'validar antes de escribir; rechazar por defecto; null solo con política explícita y registro; NaN/Inf/tensor/NumPy comprobados'},
 {'incident':'Success flag too early','evidence':'recuperación vieja activaba Phase I antes de escribir su informe','fix':'flag publicado solo después de validar datos, pesos, cachés, métricas y escribir evidencia'},
 {'incident':'MC pandas aggregate collision','evidence':'Series.aggregate se interpretaba como método','fix':'acceso explícito por clave; seis réplicas completadas y recuperadas sin recalcular'},
 {'incident':'Historical execution display counter','evidence':'frente al respaldo original, celda a401cc0e76 cambió execution_count 1 a null; código, metadatos y salidas iguales','fix':'registrado; no reejecutar ni alterar contenido histórico'}]
m6_stable.atomic_json(m6_stable.OPS/'incidents.json',incidents)
protocol='''# Protocolo de escritura M06
1. Mantener un único editor del ID principal. Antes de editar: enumerar pestañas CDP, leer versión remota, comparar IDs/código/metadatos/salidas con la versión visible.
2. Colab es el único escritor del notebook. Prohibido reemplazar sus bytes por Drive API mientras esté abierto. API solo lectura y creación de respaldos/artefactos independientes.
3. Ante conflicto: pausar cálculo conservando cachés, respaldar cada cliente mediante Guardar una copia, respaldar remoto, comparar por ID; nunca pulsar guardar mis cambios o descartar sin comparación.
4. Antes de continuar: Ctrl+S desde Colab, comprobar mensaje de éxito y leer nuevamente el archivo remoto. Verificar celda marcadora y salidas.
5. Recuperar con m06_recovery_v1. No ejecutar todas. La recuperación rechaza dependencias/cachés/hash faltantes; no recalcula experimentos para ocultar pérdidas.
6. Guardar artefactos independientes mediante m6_stable.atomic_json/atomic_bytes, con expected_sha cuando se actualice un archivo; registrar explícitamente valores faltantes. Checkpoints NPZ deben escribirse a temporal, validar y reemplazar, y acompañarse de hash/tamaño.
7. El manifiesto previo queda inmutable. Antes de una nueva fase, crear una revisión de manifiesto, conservar la previa y registrar los nuevos artefactos. No aceptar automáticamente cambios en archivos protegidos.
8. Repetir los controles de escritor y persistencia tras desconexiones o cambios de pestaña. La estabilidad es evidencia de las pruebas actuales, no garantía de que nunca reaparecerá un conflicto.
'''
m6_stable.atomic_bytes(m6_stable.OPS/'PROTOCOLO_ESCRITURA.md',protocol.encode())
rows='\n'.join('| '+k+' | '+('PASS' if v else 'FAIL')+' |' for k,v in M6_STABILITY_CHECKS.items())
report='# M06 — estabilización operativa verificada\n\nM06_ENVIRONMENT_STABLE = TRUE\n\n| Prueba | Resultado |\n|---|---|\n'+rows+'\n\n'
report+='Recuperados 8.428 contextos de entrenamiento, 950 de desarrollo, 243 sesiones NY y 60 cachés Fase I. Seis réplicas Fase II recuperadas desde archivos, sin Monte Carlo nuevo. Pesos originales y checkpoint M05 verificados.\n\n'
report+='Dos recuperaciones desde archivos devolvieron el mismo resumen, hashes, métricas y predicciones. El inventario de métricas, figuras y checkpoints no cambió. Se probaron NaN, Inf, tipos NumPy/PyTorch, un NPZ truncado aislado, escritura obsoleta y segundo escritor. Los rechazos conservaron el contenido previo.\n\n'
report+='Las versiones primaria (51 celdas) y secundaria (48) se archivaron completas. Sus primeras 40 celdas son idénticas, incluidos metadatos y salidas. Las diferencias comunes están en las salidas y ejecución de MC y recuperación; sus códigos coinciden. El principal conserva las réplicas terminadas; la secundaria archivada conserva sus errores.\n\n'
report+='El respaldo original anterior a M06 conserva un contador execution_count=1 para a401cc0e76; Colab lo muestra null en las versiones actuales. Código, metadatos y salidas históricos no cambiaron.\n\n'
report+='Autoría según registro de esta sesión: Drive API añadió previamente celdas M06 40–47 al mismo ID; Colab ejecutó fases y añadió celdas de respaldo, corrección y pruebas. Desde la orden de escritor único, ninguna actualización API ha reemplazado el principal.\n\n'
report+='Limitaciones: se observaron las pestañas expuestas por CDP; otros dispositivos no son bloqueados por este protocolo. La consulta de procesos Windows no tuvo acceso CIM. El reemplazo de archivos está sujeto a semántica Drive FUSE y siempre exige lectura/hash posterior.\n\n'
report+='RESEARCH ONLY — ORDERS DISABLED. Sin acceso a datos reservados, sin reinicio deliberado de runtime activo. No se inicia Fase III hasta guardar estas evidencias.\n'
m6_stable.atomic_bytes(m6_stable.OPS/'INFORME_ESTABILIZACION.md',report.encode())
m6_stable.atomic_json(m6_stable.OPS/'notebook_stabilized_snapshot.ipynb',_message.blocking_request('get_ipynb',timeout_sec=30)['ipynb'])
print('M06_ENVIRONMENT_STABLE =',str(M06_ENVIRONMENT_STABLE).upper())
print(json.dumps(M6_STABILITY_CHECKS,indent=2,ensure_ascii=False))
print('EVIDENCIAS',m6_stable.OPS)
display(Markdown(report))
