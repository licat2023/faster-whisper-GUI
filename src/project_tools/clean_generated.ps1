# 删除本项目可重建的产物；不遍历虚拟环境、模型、录音或日志。
param([switch]$ClearBytecodeCache)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$sourceRoot = Join-Path $taskRoot 'src'
function Assert-NoLinkedTree([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "清理目标超出项目：$absolute"
    }
    # Check each ancestor before entering it, including .cache and build.
    $relative = $absolute.Substring($taskRoot.Length + 1)
    $cursor = $taskRoot
    foreach ($part in $relative.Split([IO.Path]::DirectorySeparatorChar)) {
        $cursor = Join-Path $cursor $part
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "拒绝清理链接：$cursor" }
        }
    }
    if (-not (Test-Path -LiteralPath $absolute)) { return }
    $pending = [Collections.Generic.Queue[string]]::new()
    $pending.Enqueue($absolute)
    while ($pending.Count) {
        $current = Get-Item -LiteralPath $pending.Dequeue() -Force
        if ($current.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "拒绝清理链接：$($current.FullName)" }
        if ($current.PSIsContainer) {
            foreach ($child in Get-ChildItem -LiteralPath $current.FullName -Force) {
                if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "拒绝清理链接：$($child.FullName)" }
                if ($child.PSIsContainer) { $pending.Enqueue($child.FullName) }
            }
        }
    }
}
# Complete validation before deleting even one bytecode file.
Assert-NoLinkedTree $sourceRoot
foreach ($relative in @('__pycache__', 'build', 'src\faster_whisper_gui.egg-info')) {
    Assert-NoLinkedTree (Join-Path $taskRoot $relative)
}
if ($ClearBytecodeCache) { Assert-NoLinkedTree (Join-Path $taskRoot '.cache\pycache') }
function Remove-GeneratedDirectory([string]$RelativePath) {
    $target = [IO.Path]::GetFullPath((Join-Path $taskRoot $RelativePath))
    if (-not $target.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "清理目标超出项目：$target"
    }
    if (Test-Path -LiteralPath $target) {
        Assert-NoLinkedTree $target
        $item = Get-Item -LiteralPath $target
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "拒绝清理链接：$target" }
        Remove-Item -LiteralPath $target -Recurse -Force
        Write-Host "已清理 $RelativePath"
    }
}
$bytecode = @(Get-ChildItem -LiteralPath $taskRoot -File | Where-Object Extension -in '.pyc', '.pyo')
$bytecode += @(Get-ChildItem -LiteralPath (Join-Path $taskRoot 'src') -Recurse -File | Where-Object Extension -in '.pyc', '.pyo')
foreach ($item in $bytecode) { Remove-Item -LiteralPath $item.FullName -Force }
Get-ChildItem -LiteralPath (Join-Path $taskRoot 'src') -Recurse -Directory -Filter __pycache__ |
    Sort-Object { $_.FullName.Length } -Descending | ForEach-Object {
        if (-not (Get-ChildItem -LiteralPath $_.FullName -Force)) { Remove-Item -LiteralPath $_.FullName -Force }
    }
Remove-GeneratedDirectory '__pycache__'
Remove-GeneratedDirectory 'build'
Remove-GeneratedDirectory 'src\faster_whisper_gui.egg-info'
if ($ClearBytecodeCache) { Remove-GeneratedDirectory '.cache\pycache' }
Write-Host "已删除 $($bytecode.Count) 个分散字节码文件"
