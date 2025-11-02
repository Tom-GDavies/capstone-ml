Param(
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPath = Join-Path $ScriptDir ".venv"

if (Test-Path $VenvPath) {
    Write-Host "Virtual environment already exists at $VenvPath"
} else {
    & $PythonPath -m venv $VenvPath
    Write-Host "Created venv at $VenvPath"
}

$Activate = Join-Path $VenvPath "Scripts\Activate.ps1"
. $Activate

python -m pip install --upgrade pip
pip install -r requirements.txt

deactivate
Write-Host "Environment setup complete. Activate later with .\.venv\Scripts\Activate.ps1"
