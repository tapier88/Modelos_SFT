[CmdletBinding()]
param([Parameter(Position=0)][ValidateSet('start','status','recover','close')][string]$Action='start')
$ErrorActionPreference='Stop'
$Root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$NotebookId='1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss'
$NotebookUrl="https://colab.research.google.com/drive/$NotebookId"
$DriveRoot='/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571'
$StatePath=Join-Path $Root 'ESTADO_ACTUAL_KRONOS.md'
$EvidencePath=Join-Path $Root 'm06_staging/m07_reconciliation_20261120T224952Z/final_result.json'
$Now=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
$SessionDir=Join-Path $Root "output/axxel/$Now"
New-Item -ItemType Directory -Force $SessionDir | Out-Null
$Result=[ordered]@{timestamp_utc=[DateTime]::UtcNow.ToString('o');action=$Action;research_only=$true;orders_enabled=$false;test_2023_2024='SEALED';notebook=$NotebookUrl;cdp='unknown';runtime='unknown';gpu='unknown';drive='unknown';models='unknown';checkpoint='unknown';mission='';training_active='unknown';remote_execution_enabled=$false;recovery_mode='diagnostic_backup_only';notes=@()}
$Result.remote_execution_enabled=($Action -eq 'start')
$Result['training_enabled']=$false
$Result['remote_scope']='pinned_health_and_restore_only_for_start'

function Write-Session { $Result | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 (Join-Path $SessionDir 'session.json') }
function Ensure-ChromeCDP {
  try { $null=Invoke-RestMethod -Uri 'http://127.0.0.1:9222/json/version' -TimeoutSec 3; return } catch { }
  $chrome='C:\Program Files\Google\Chrome\Application\chrome.exe'
  $profile='C:\ChromeAutomationProfile'
  if(-not (Test-Path -LiteralPath $chrome) -or -not (Test-Path -LiteralPath $profile)){
    throw 'Chrome o perfil C:\ChromeAutomationProfile ausente; revise scripts/README.md.'
  }
  Start-Process -FilePath $chrome -ArgumentList @('--remote-debugging-address=127.0.0.1','--remote-debugging-port=9222',"--user-data-dir=$profile",$NotebookUrl) -WindowStyle Hidden
  for($attempt=0;$attempt -lt 15;$attempt++){
    Start-Sleep -Seconds 1
    try { $null=Invoke-RestMethod -Uri 'http://127.0.0.1:9222/json/version' -TimeoutSec 2; return } catch { }
  }
  throw 'Chrome CDP no disponible despues del arranque; no se cerraron procesos existentes.'
}
function Get-NotebookPage {
  $url="http://127.0.0.1:9222/json/list"
  try { $targets=Invoke-RestMethod -Uri $url -TimeoutSec 3 } catch { $Result.cdp='unavailable'; return $null }
  $Result.cdp='connected'
  $matches=@($targets | Where-Object { $_.type -eq 'page' -and $_.url -match [regex]::Escape("/drive/$NotebookId") })
  if($matches.Count -gt 1){$Result.notes+=@('Notebook maestro aparece en más de una pestaña; se requiere resolver ambigüedad.');return $null}
  if($matches.Count -eq 1){return $matches[0]}
  return $null
}
function Get-NotebookSnapshot([switch]$Required) {
  $page=Get-NotebookPage
  if(-not $page){if($Result.cdp -eq 'connected'){$Result.runtime='notebook_not_open'};return $null}
  $Result.runtime='notebook_open; backend state requires UI/runtime probe'
  $Result.notes+=@("Notebook encontrado: $($page.url); title=$($page.title)")
  if($Required){$Result.notes+=@('Sin conector oficial local de Colab; recuperar por Playwright/CDP solo tras inspección y preservación del notebook.')}
  return $page
}
function Get-PersistentEvidence {
  if(Test-Path $EvidencePath){
    $e=Get-Content -Raw -Encoding UTF8 $EvidencePath | ConvertFrom-Json
    $dir=$e.output -replace '^/content/drive/MyDrive', ''
    $Result.checkpoint="M07 manifest sha256=$($e.manifest_sha256); $($e.files_verified) files; remote path $($e.output)"
    $Result.models="Recovered historically with hashes in persisted M07 evidence; current runtime must be checked"
    $Result.drive='historical mount verified; current runtime must be checked'
  } else {$Result.notes+=@('Falta evidencia local de reconciliación; no se infiere estado científico.')}
  $txt=Get-Content -Raw -Encoding UTF8 $StatePath
  $Result.mission=($txt -split "`n" | Where-Object {$_ -match 'Fase V:|Fase V NO|Pendiente|Próxima acción'} | Select-Object -First 2) -join ' '
  $logs=Get-ChildItem (Join-Path $Root 'output/axxel') -Directory | Sort-Object Name -Descending
  foreach($log in $logs){
    $record=Join-Path $log.FullName 'bootstrap_restore/result.json'
    if(Test-Path -LiteralPath $record){
      try {
        $past=Get-Content -Raw -Encoding UTF8 $record | ConvertFrom-Json
        if($past.result.status -eq 'RESTORED'){
          $Result['last_verified_restoration']=[ordered]@{provenance='HISTORICAL';date=$past.result.timestamp_utc;source=$record;gpu=$past.result.gpu;torch=$past.result.torch;heads=$past.result.heads}
          break
        }
      } catch { $Result.notes+=@("Registro ilegible: $record") }
    }
  }
}
function Read-CurrentStatus {
  $diagnosticPath=Join-Path $SessionDir 'readonly_status.json'
  try {
    python (Join-Path $Root 'automation/m08_readonly_status.py') --output $diagnosticPath
    if($LASTEXITCODE -ne 0){throw 'Read-only diagnostic process failed'}
    $diagnostic=Get-Content -Raw -Encoding UTF8 $diagnosticPath | ConvertFrom-Json
    if($Action -eq 'start' -and $Result.bootstrap){
      $health=$Result.bootstrap.result
      $live=@{gpu=$health.gpu.stdout.Trim();gpu_memory=$health.gpu.stdout.Trim();drive=($health.drive | ConvertTo-Json -Compress);checkpoint='Hashes individuales verificados; consultar bootstrap/result.json'}
      foreach($key in $live.Keys){
        $diagnostic.components.$key=[pscustomobject]@{status='CURRENT';value=$live[$key];evidence='Celda health fijada; resultado recuperado localmente';date=$health.timestamp_utc}
      }
    }
    if($Action -eq 'start' -and $Result.restoration){
      $restored=$Result.restoration.result
      $live=@{models="AXXEL_SESSION: $($restored.status), reused=$($restored.reused)";cuda=$restored.cuda;pytorch=$restored.torch}
      foreach($key in $live.Keys){
        $diagnostic.components.$key=[pscustomobject]@{status='CURRENT';value=$live[$key];evidence='bootstrap_restore/result.json';date=$restored.timestamp_utc}
      }
    }
    $Result['diagnostic']=$diagnostic
    if($diagnostic.snapshot.exact_tabs -eq 1){$Result.cdp='connected'}
    $Result.checkpoint='HISTORICAL: ' + $Result.checkpoint
    foreach($key in @('runtime','gpu','drive','models')){
      $entry=$diagnostic.components.$key
      $Result[$key]="$($entry.status): $($entry.value)"
    }
    if($diagnostic.snapshot.error){$Result.notes+=@("UNKNOWN: $($diagnostic.snapshot.error)")}
    foreach($property in $diagnostic.components.PSObject.Properties){
      $entry=$property.Value
      Write-Host "$($property.Name) | $($entry.status) | $($entry.value) | $($entry.evidence) | $($entry.date)"
    }
    foreach($property in $diagnostic.historical.PSObject.Properties){
      $entry=$property.Value
      Write-Host "$($property.Name) | HISTORICAL | $($entry.value | ConvertTo-Json -Compress -Depth 8) | $($entry.evidence) | $($entry.date)"
    }
  } catch {
    foreach($key in @('runtime','gpu','drive','models')){$Result[$key]='UNKNOWN: diagnostic unavailable'}
    $Result.notes+=@("UNKNOWN: read-only diagnostic failed: $($_.Exception.Message)")
  }
}
function Start-VerifiedSession {
  $Result['start_status']='BLOCKED'
  try { Ensure-ChromeCDP } catch { $Result.notes+=@($_.Exception.Message);return }
  $startDir=Join-Path $SessionDir 'bootstrap'
  python (Join-Path $Root 'automation/axxel_colab_start.py') --output $startDir
  if($LASTEXITCODE -ne 0){
    $Result.notes+=@("Inicio detenido; evidencia: $startDir/error.json. No se reintenta una ejecucion incierta.")
    return
  }
  $Result['bootstrap']=Get-Content -Raw -Encoding UTF8 (Join-Path $startDir 'result.json') | ConvertFrom-Json
  if($Result.bootstrap.status -eq 'PROBE_RECOVERED'){
    $startDir=Join-Path $SessionDir 'bootstrap_fresh'
    python (Join-Path $Root 'automation/axxel_colab_start.py') --output $startDir
    if($LASTEXITCODE -ne 0){$Result.notes+=@('Resultado previo recuperado; lectura actual pendiente.');return}
    $Result.bootstrap=Get-Content -Raw -Encoding UTF8 (Join-Path $startDir 'result.json') | ConvertFrom-Json
  }
  $health=$Result.bootstrap.result
  $valid=($health.drive.mountpoint -and $health.drive.AXXEL_CAPITAL_is_dir -and $health.drive.read_search_access)
  foreach($phase in @('fase_III','fase_IV')){
    $part=$health.phases.$phase
    if(-not $part.manifest_matches_M07 -or -not $part.checkpoints){$valid=$false}
    foreach($checkpoint in $part.checkpoints){if(-not $checkpoint.matches){$valid=$false}}
  }
  if(-not $valid){$Result.notes+=@('Drive o integridad pendientes. No se restauran modelos; revise bootstrap/result.json.');return}
  $restoreDir=Join-Path $SessionDir 'bootstrap_restore'
  python (Join-Path $Root 'automation/axxel_colab_start.py') --job restore --output $restoreDir
  if($LASTEXITCODE -ne 0){$Result.notes+=@("Restauracion detenida; evidencia: $restoreDir");return}
  $Result['restoration']=Get-Content -Raw -Encoding UTF8 (Join-Path $restoreDir 'result.json') | ConvertFrom-Json
  $Result['start_status']='RESTORED'
  $Result.notes+=@('Sesion Kronos recuperada en AXXEL_SESSION; entrenamiento deshabilitado.')
}
function Save-NotebookBackup($Page) {
  $dest=Join-Path $SessionDir 'notebook_before.ipynb'
  $env:AXXEL_NOTEBOOK_ID=$NotebookId; $env:AXXEL_NOTEBOOK_URL=$NotebookUrl; $env:AXXEL_BACKUP_PATH=$dest
  $py=@'
import os,json,pathlib
from playwright.sync_api import sync_playwright
nid=os.environ['AXXEL_NOTEBOOK_ID']; url=os.environ['AXXEL_NOTEBOOK_URL']; dest=pathlib.Path(os.environ['AXXEL_BACKUP_PATH'])
with sync_playwright() as p:
 b=p.chromium.connect_over_cdp('http://127.0.0.1:9222',timeout=10000)
 pages=[x for c in b.contexts for x in c.pages if x.url.split('?')[0].split('#')[0].rstrip('/')==url]
 assert len(pages)==1,'Notebook identity ambiguous or missing'
 page=pages[0]
 from controlar_kronos_colab import menu_item
 page.keyboard.press('Escape'); menu_item(page,[r'^Archivo$',r'^File$']).click(); menu_item(page,[r'Descargar',r'Download']).hover()
 with page.expect_download(timeout=30000) as d: menu_item(page,[r'^(Descargar\s+|Download\s+)?\.ipynb$']).click()
 d.value.save_as(str(dest))
 nb=json.loads(dest.read_text(encoding='utf-8'))
 assert len([c for c in nb['cells'] if c['cell_type']=='code'])>0
 print(json.dumps({'cells':len(nb['cells']),'sha256':__import__('hashlib').sha256(dest.read_bytes()).hexdigest()}))
'@
  $env:PYTHONPATH=(Join-Path $Root 'automation')
  $py | python -
  if($LASTEXITCODE -ne 0){throw 'Notebook backup or identity check failed'}
  Remove-Item Env:AXXEL_NOTEBOOK_ID,Env:AXXEL_NOTEBOOK_URL,Env:AXXEL_BACKUP_PATH,Env:PYTHONPATH -ErrorAction SilentlyContinue
}

Get-PersistentEvidence
$page=Get-NotebookSnapshot
switch($Action){
 'status' { Read-CurrentStatus }
 'start' { Start-VerifiedSession; Read-CurrentStatus
   if($Result.bootstrap){
     $health=$Result.bootstrap.result
     $Result.gpu="CURRENT $($health.timestamp_utc): $($health.gpu.stdout.Trim())"
     $Result.drive="CURRENT $($health.timestamp_utc): $($health.drive | ConvertTo-Json -Compress)"
     $Result.checkpoint="CURRENT $($health.timestamp_utc): ver bootstrap/result.json para hashes individuales"
   }
   if($Result.restoration){$Result.models="CURRENT: AXXEL_SESSION RESTORED; $($Result.restoration.result.timestamp_utc)"}
 }
 'recover' {
   if(-not $page){$Result.notes+=@('Recovery detenido: abre el notebook maestro en Chrome con el runtime deseado.');break}
   Save-NotebookBackup $page
   $Result.notes+=@('Respaldo de diagnostico guardado.')
   $Result.notes+=@('RECOVER_DIAGNOSTIC_ONLY: restauracion y ejecucion remota deshabilitadas.')
 }
 'close' {
   if($page){Save-NotebookBackup $page;$Result.notes+=@('Notebook respaldado; ejecución remota no interrumpida. Revisa visualmente salidas activas antes de cerrar Colab.')}
   else{$Result.notes+=@('Notebook no accesible por CDP; cierre remoto no verificado.')}
 }
}
Write-Session
if($Result.bootstrap){
  $health=$Result.bootstrap.result
  Write-Host "LECTURA REMOTA $($health.timestamp_utc): GPU=$($health.gpu.stdout.Trim()); Drive=$($health.drive | ConvertTo-Json -Compress)"
}
if($Result.restoration){
  $restored=$Result.restoration.result
  Write-Host "KRONOS: $($restored.status); reused=$($restored.reused); heads=$($restored.heads); CUDA=$($restored.cuda); entrenamiento=false"
}
Write-Host "AXXEL $Action | CDP=$($Result.cdp) | Runtime=$($Result.runtime) | Drive=$($Result.drive) | Modelos=$($Result.models)"
Write-Host "Checkpoint=$($Result.checkpoint)"
if($Result.last_verified_restoration){Write-Host "Ultima restauracion verificada (no lectura actual): $($Result.last_verified_restoration | ConvertTo-Json -Compress)"}
Write-Host "Misión pendiente=$($Result.mission)"
foreach($n in $Result.notes){Write-Host "- $n"}
Write-Host "Registro: $SessionDir"
if($Action -eq 'start' -and $Result.start_status -ne 'RESTORED'){
  $global:LASTEXITCODE=2
  if($MyInvocation.InvocationName -ne '.') { exit 2 }
}
