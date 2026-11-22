# Local integration harness: no network, browser or real Python process.
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$global:backupCalls=0
function Start-Process { throw 'SIMULATED_BROWSER_LAUNCH_UNAVAILABLE' }
function Invoke-RestMethod {
    param($Uri,$TimeoutSec)
    if($global:scenario -eq 'offline'){throw 'SIMULATED_CDP_OFFLINE'}
    if($global:scenario -eq 'missing'){return @()}
    $page=[pscustomobject]@{type='page';url='https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss';title='SIMULATED'}
    if($global:scenario -eq 'duplicate'){return @($page,$page)}
    return @($page)
}
function python {
    begin {
        $lines=@()
        $diagnosticOnly=($args.Count -ge 3 -and $args[0] -like '*m08_readonly_status.py')
        $bootstrapOnly=($args.Count -ge 3 -and $args[0] -like '*axxel_colab_start.py')
        if($bootstrapOnly){
            if($verb -ne 'start'){throw 'Bootstrap escaped start boundary'}
            $global:LASTEXITCODE=2
        }
        if($diagnosticOnly){
            '{"components":{},"historical":{},"snapshot":{"simulated":true}}' | Set-Content -Encoding UTF8 $args[2]
            $global:LASTEXITCODE=0
        }
    }
    process {$lines+=[string]$_}
    end {
        if($diagnosticOnly -or $bootstrapOnly){return}
        $code=$lines -join "`n"
        if($code -match 'Ejecutar celda|insert_text|Control\+Enter|Invoke-VerifiedCell|optimizer|\.evaluate\('){throw 'REMOTE EXECUTION DETECTED'}
        if($code -notmatch 'expect_download' -or $env:AXXEL_BACKUP_PATH -notlike '*notebook_before.ipynb'){throw 'Not a backup'}
        $global:backupCalls++
        '{"cells":[],"simulated":true}' | Set-Content -Encoding UTF8 $env:AXXEL_BACKUP_PATH
        $global:LASTEXITCODE=0
    }
}
$cases=0
foreach($scenario in @('connected','missing','offline','duplicate')){
    $global:scenario=$scenario
    foreach($verb in @('start','status','recover','close')){
        $before=$global:backupCalls
        . (Join-Path $root 'scripts/axxel.ps1') $verb | Out-Null
        $record=Get-Content -Raw -Encoding UTF8 (Join-Path $SessionDir 'session.json') | ConvertFrom-Json
        $expected=0
        if($scenario -eq 'connected' -and $verb -in @('recover','close')){$expected=1}
        if(($global:backupCalls-$before) -ne $expected){throw "Unexpected backup calls: $scenario/$verb"}
        if($record.remote_execution_enabled -ne ($verb -eq 'start') -or $record.recovery_mode -ne 'diagnostic_backup_only'){throw 'Unsafe mode'}
        if($record.training_enabled -ne $false){throw 'Training enabled'}
        if($record.action -ne $verb){throw 'Wrong recorded action'}
        if($record.training_active -ne 'unknown'){throw 'Invented live runtime state'}
        $cases++
    }
}
Write-Output "SIMULATED acceptance PASS: $cases cases, no remote cells executed"
