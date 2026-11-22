$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$target = Join-Path $root 'scripts/axxel.ps1'
$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($target, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -gt 0) {
    $parseErrors | ForEach-Object { Write-Error ("{0}:{1}: {2}" -f $_.Extent.StartLineNumber, $_.Extent.StartColumnNumber, $_.Message) }
    exit 1
}
Write-Output 'PowerShell syntax OK: scripts/axxel.ps1'
