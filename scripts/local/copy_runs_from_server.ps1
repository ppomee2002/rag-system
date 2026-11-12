# Copy run folders from the experiment server to this PC (Windows PowerShell 5.1). Run by the user.
# - Remote side: read only (ls, sha256sum).
# - Local side: aborts if any destination already exists. Files go to a new staging folder,
#   are verified by sha256 against the server, and only then moved (Move-Item never overwrites).
#
#   powershell -ExecutionPolicy Bypass -File scripts\local\copy_runs_from_server.ps1 `
#       -User <account> -Runs results/crag_system/pre/PRE-DIAG-GRADER-AWQ-R1/<ts>,results/crag_system/pre/PRE-DIAG-GRADER-AWQ-R2/<ts>
#
# Paths in -Runs are relative to the repo root (same layout on both sides).
# ASCII only on purpose: PowerShell 5.1 reads BOM-less UTF-8 scripts as ANSI.
param(
    [Parameter(Mandatory = $true)][string]$User,
    [Parameter(Mandatory = $true)][string[]]$Runs,
    [string]$HostName = "100.82.53.54",
    [int]$Port = 2221,
    [string]$RemoteRoot = "RAG",          # relative to the remote home (~/RAG)
    [string]$LocalRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
)
$ErrorActionPreference = "Stop"
$target = "$User@$HostName"
# with -File, "-Runs a,b" arrives as one string: split on commas
$Runs = @($Runs | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim().TrimEnd('/') } | Where-Object { $_ })
if ($Runs.Count -eq 0) { throw "no run paths given" }

foreach ($r in $Runs) {
    if ($r -match '^[/\\]|\.\.') { throw "Runs must be repo-relative without '..': $r" }
    $dst = Join-Path $LocalRoot $r
    if (Test-Path -LiteralPath $dst) { throw "Destination exists, aborting (no overwrite): $dst" }
}

Write-Host "== remote listing"
ssh -p $Port $target "cd $RemoteRoot && for d in $($Runs -join ' '); do echo [`$d]; ls -la `$d || exit 1; done"
if ($LASTEXITCODE -ne 0) { throw "remote run folder check failed" }

$stage = Join-Path $LocalRoot ("results\_incoming\copy_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $stage | Out-Null
foreach ($r in $Runs) {
    $parent = Join-Path $stage (Split-Path $r -Parent)
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Write-Host "== scp $r"
    scp -P $Port -r -p "${target}:$RemoteRoot/$r" $parent
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $r" }
}

Write-Host "== sha256 check"
$remote = ssh -p $Port $target "cd $RemoteRoot && find $($Runs -join ' ') -type f -exec sha256sum {} +"
if ($LASTEXITCODE -ne 0) { throw "remote sha256sum failed" }
$n = 0
foreach ($line in $remote) {
    $h, $p = $line -split '\s+', 2
    $p = $p.TrimStart('*').Trim()
    $lp = Join-Path $stage $p
    if (-not (Test-Path -LiteralPath $lp)) { throw "missing in staging: $p" }
    $lh = (Get-FileHash -LiteralPath $lp -Algorithm SHA256).Hash.ToLower()
    if ($lh -ne $h) { throw "hash mismatch: $p" }
    $n++
}
$local = (Get-ChildItem -LiteralPath $stage -Recurse -File).Count
if ($local -ne $n) { throw "file count mismatch: remote $n, staging $local" }
Write-Host "  $n files match"

foreach ($r in $Runs) {
    $dst = Join-Path $LocalRoot $r
    New-Item -ItemType Directory -Path (Split-Path $dst -Parent) -Force | Out-Null
    Move-Item -LiteralPath (Join-Path $stage $r) -Destination $dst
    Write-Host "  -> $dst"
}
Write-Host "done. empty staging folder left at $stage (delete after checking)"
