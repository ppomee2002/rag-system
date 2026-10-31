param([string]$Model)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..\..')
if (-not $Model) { $Model = Read-Host 'OpenAI model ID' }
if (-not $env:OPENAI_API_KEY) {
    $secret = Read-Host 'OpenAI API key (hidden input; not saved)' -AsSecureString
    $env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
}
# This script does not spend API credits. It checks local prerequisites only.
$env:EXPERIMENT_MODEL = $Model
& .\.venv\Scripts\python.exe experiments/grading_optimization/preflight.py
