# Full pipeline queue: training -> translation -> temperature sweep -> evaluation.
# Each step logs to results/logs/<step>.out.txt; a failed step is recorded and the queue continues.
# Usage (repo root):  powershell -ExecutionPolicy Bypass -File run_all.ps1 [-Skip step1,step2]
param([string[]]$Skip = @())

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = ".venv\Scripts\python.exe"
$summary = "results\logs\queue_summary.txt"
New-Item -ItemType Directory -Force results\logs | Out-Null
"queue started $(Get-Date -Format s)" | Out-File -Encoding utf8 $summary

$steps = @(
    @("train_transformer", "-m src.transformer_scratch.train"),
    @("train_mt5",         "-m src.train_mt5"),
    @("train_llama_lora",  "-m src.train_lora --model llama"),
    @("train_gemma_lora",  "-m src.train_lora --model gemma"),
    @("tr_transformer",    "-m src.translate --system transformer"),
    @("tr_mt5",            "-m src.translate --system mt5"),
    @("tr_llama_prompt",   "-m src.translate --system llama_prompt"),
    @("tr_llama_lora",     "-m src.translate --system llama_lora"),
    @("tr_gemma_prompt",   "-m src.translate --system gemma_prompt"),
    @("tr_gemma_lora",     "-m src.translate --system gemma_lora"),
    @("tr_transformer_b4", "-m src.translate --system transformer --beam 4"),
    @("tr_mt5_b4",         "-m src.translate --system mt5 --beam 4"),
    @("temperature_sweep", "-m src.temperature_sweep"),
    @("evaluate",          "-m src.evaluate")
)

foreach ($s in $steps) {
    $name, $args_ = $s
    if ($Skip -contains $name) { "$name SKIPPED" | Out-File -Append -Encoding utf8 $summary; continue }
    $t0 = Get-Date
    cmd /c "$py $args_ > results\logs\$name.out.txt 2>&1"
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    "$name exit=$code minutes=$mins" | Out-File -Append -Encoding utf8 $summary
}
"queue finished $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $summary
