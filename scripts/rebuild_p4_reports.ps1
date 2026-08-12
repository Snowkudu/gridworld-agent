$stages = @(
    "baseline",
    "architecture",
    "head",
    "optimization",
    "weight_decay",
    "weight_decay_finalists"
)

foreach ($stage in $stages) {
    Write-Host ""
    Write-Host "===== $stage ====="

    $tbDir = "artifacts/p4_cnn/$stage/tensorboard"
    $exportDir = "artifacts/p4_cnn/$stage/tensorboard_exports"

    python -m utils.export_tensorboard `
        --input $tbDir `
        --output-dir $exportDir

    if ($LASTEXITCODE -ne 0) {
        throw "TensorBoard export failed for $stage"
    }
}

Write-Host ""
Write-Host "P4 TensorBoard exports complete."