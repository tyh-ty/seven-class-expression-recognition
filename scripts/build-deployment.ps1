param([switch]$Offline)

$ErrorActionPreference = 'Stop'
$env:PSModulePath = (Join-Path $PSHOME 'Modules') + [IO.Path]::PathSeparator + $env:PSModulePath
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$configuration = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $root 'deployment\runtime-lock.json') | ConvertFrom-Json
$arguments = @('-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'start-demo.ps1'), '-PrepareOnly')
if ($Offline) { $arguments += '-Offline' }
& powershell.exe @arguments
if ($LASTEXITCODE -ne 0) { throw '环境自检失败，没有生成部署包。' }

# Include the installer as well: a new machine may lack the Microsoft C++ runtime.
$assets = @($configuration.python, $configuration.vc_redist) + @($configuration.wheels)
foreach ($asset in $assets) {
    $path = Join-Path (Join-Path $root 'deployment\cache') $asset.file
    $valid = (Test-Path -LiteralPath $path) -and ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -eq $asset.sha256)
    if (-not $valid) {
        if ($Offline) { throw "离线资源缺失：$($asset.file)" }
        Write-Host "正在准备离线资源：$($asset.file)"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force | Out-Null
        Invoke-WebRequest -UseBasicParsing -Uri $asset.url -OutFile "$path.partial" -TimeoutSec 300
        if ((Get-FileHash -LiteralPath "$path.partial" -Algorithm SHA256).Hash -ne $asset.sha256) {
            throw "下载校验失败：$($asset.file)"
        }
        Move-Item -LiteralPath "$path.partial" -Destination $path -Force
    }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$dist = Join-Path $root 'dist'
$staging = Join-Path $dist "seven-class-expression-system-windows-x64-$stamp"
New-Item -ItemType Directory -Path $staging -Force | Out-Null
$files = @(
    'start-demo.cmd', '启动服务.cmd', 'server.py',
    'scripts\start-demo.ps1', 'scripts\launcher.py',
    'training\emotion_preprocessing.py',
    'deployment\runtime-lock.json', 'deployment\requirements-runtime.txt'
)
$files += @($configuration.models | ForEach-Object { $_.file })
$files += @($assets | ForEach-Object { Join-Path 'deployment\cache' $_.file })
$files += @(Get-ChildItem -LiteralPath (Join-Path $root 'web') -File | ForEach-Object { Join-Path 'web' $_.Name })
foreach ($file in $files) {
    $destination = Join-Path $staging $file
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $root $file) -Destination $destination
}
$zipPath = "$staging.zip"
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($staging, $zipPath, 'Optimal', $true)
$hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash
"$hash  $([IO.Path]::GetFileName($zipPath))" | Set-Content -LiteralPath "$zipPath.sha256" -Encoding ASCII
& (Join-Path $root '.runtime\python.exe') -I -X utf8 (Join-Path $PSScriptRoot 'verify_deployment.py') --source $root --zip $zipPath
if ($LASTEXITCODE -ne 0) { throw '新包完整性或源文件一致性检查失败；旧包未清理。' }
& (Join-Path $PSScriptRoot 'cleanup-deployment.ps1') -KeepZip $zipPath
Write-Host "部署包已生成：$zipPath"
Write-Host '复制 ZIP 到实验室电脑，完整解压后双击 启动服务.cmd。无需复制训练数据和训练环境。'
