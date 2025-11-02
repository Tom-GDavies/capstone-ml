$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$DestRoot = Join-Path $ScriptDir "deliverables\omtad_xiv"
New-Item -ItemType Directory -Force -Path $DestRoot | Out-Null

function Copy-WithPattern {
    param(
        [string]$SourceDir,
        [string]$TargetDir,
        [string]$Pattern
    )

    if (Test-Path $SourceDir) {
        $files = Get-ChildItem -Path $SourceDir -Filter $Pattern -File -ErrorAction SilentlyContinue
        if ($files.Count -gt 0) {
            New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
            foreach ($file in $files) {
                Copy-Item $file.FullName -Destination $TargetDir -Force
            }
        } else {
            Write-Warning "No files matching $Pattern in $SourceDir"
        }
    } else {
        Write-Warning "Missing directory $SourceDir"
    }
}

Copy-WithPattern "$DestRoot\models" "$DestRoot\models" "*.pkl"
Copy-WithPattern "$DestRoot\models" "$DestRoot\models" "*.meta.json"
Copy-WithPattern "$DestRoot\charts" "$DestRoot\charts" "*.png"
Copy-WithPattern "$DestRoot\samples" "$DestRoot\samples" "track_features_sample.csv"
Copy-WithPattern "$DestRoot\samples" "$DestRoot\samples" "demo_tracks.csv"

$MetricsPath = Join-Path $DestRoot "metrics\report.txt"
if (-not (Test-Path $MetricsPath)) {
    $SourceMetrics = Join-Path $ScriptDir "metrics\report.txt"
    if (Test-Path $SourceMetrics) {
        New-Item -ItemType Directory -Force -Path (Join-Path $DestRoot "metrics") | Out-Null
        Copy-Item $SourceMetrics -Destination (Join-Path $DestRoot "metrics") -Force
    } else {
        Write-Warning "metrics/report.txt not found"
    }
}

$DemoPath = Join-Path $ScriptDir "DEMO.md"
if (Test-Path $DemoPath) {
    Copy-Item $DemoPath -Destination $DestRoot -Force
} else {
    Write-Warning "DEMO.md not found"
}

Write-Host "Exported files:"
Get-ChildItem -Path $DestRoot -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($DestRoot.Length + 1)
    Write-Host ("  {0} ({1} bytes)" -f $rel, $_.Length)
}
