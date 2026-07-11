# Master Stage-1 pipeline for Windows remote hosts (A6000).
# Usage:
#   .\scripts\run_all_stage1.ps1
#   .\scripts\run_all_stage1.ps1 -Smoke
#   .\scripts\run_all_stage1.ps1 -SkipVlm
param(
    [switch]$Smoke,
    [switch]$SkipVlm
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONPATH = $Root

$extra = @()
if ($Smoke) {
    $extra = @("--max_train_samples", "2000", "--max_val_samples", "500")
    $epText = 1; $epHc = 2; $epVlm = 1
} else {
    $epText = 4; $epHc = 12; $epVlm = 2
}

Write-Host "=== MMHS Stage-1 | smoke=$Smoke skipVlm=$SkipVlm ==="

if (-not (Test-Path "dataset\MMHS150K_GT.json")) {
    Write-Host "Downloading Kaggle dataset..."
    python scripts/setup_kaggle_data.py
}

python scripts/run_s1_text.py --model hate-latest --text_mode all_text --epochs $epText @extra
python scripts/run_s1_text.py --model twitter-roberta --text_mode all_text --epochs $epText @extra
python scripts/run_s1_hateclipper.py --fusion align --epochs $epHc @extra

if (-not $SkipVlm) {
    python scripts/run_s1_vlm.py --model "Qwen/Qwen3-VL-8B-Instruct" --epochs $epVlm @extra
}

python scripts/run_s1_ensemble.py
Write-Host "Done → results/stage1/"
