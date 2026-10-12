param(
    [switch]$PrepareOnly,
    [switch]$Offline,
    [switch]$NoBrowser,
    [ValidateRange(1024, 65515)][int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Prefer this PowerShell's built-ins even when launched by Python or PowerShell 7.
$env:PSModulePath = (Join-Path $PSHOME 'Modules') + [IO.Path]::PathSeparator + $env:PSModulePath
[Console]::OutputEncoding = New-Object Text.UTF8Encoding
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = Join-Path $root '.runtime'
$python = Join-Path $runtime 'python.exe'
$cache = Join-Path $root 'deployment\cache'
$lockPath = Join-Path $root 'deployment\runtime-lock.json'
$outputDirectory = Join-Path $root 'output'
$setupLock = $null
$transcribing = $false
$exitCode = 0

function Get-CheckedAsset($Asset) {
    $destination = Join-Path $cache $Asset.file
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -eq $Asset.sha256) {
            return $destination
        }
        Write-Host "缓存校验未通过，需要重新获取：$($Asset.file)"
    }
    if ($Offline) {
        throw "离线安装文件缺失或损坏：$($Asset.file)。请重新复制完整部署包。"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Write-Host "正在下载：$($Asset.file)"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $partial = "$destination.partial"
    $downloadError = $null
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Asset.url -OutFile $partial -TimeoutSec 300
            if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $Asset.sha256) {
                throw 'SHA256 校验失败，未使用下载内容。'
            }
            Move-Item -LiteralPath $partial -Destination $destination -Force
            return $destination
        } catch {
            $downloadError = $_.Exception.Message
            Write-Host "下载未完成（$attempt/3）：$downloadError"
        }
    }
    throw "无法下载 $($Asset.file)。请检查网络，或使用完整离线部署包。$downloadError"
}

function Test-PythonCommand([string]$Code) {
    if (-not (Test-Path -LiteralPath $python)) { return $false }
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python -I -X utf8 -c $Code 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Install-VCRuntime($Configuration) {
    Write-Host '缺少 Microsoft C++ 运行库，正在自动准备安装（系统可能要求管理员确认）。'
    $installer = Get-CheckedAsset $Configuration.vc_redist
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw 'Microsoft C++ 安装包签名校验失败，请联系实验室管理员安装官方 x64 运行库。'
    }
    $process = Start-Process -FilePath $installer -ArgumentList '/install', '/quiet', '/norestart' -Verb RunAs -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -eq 3010) {
        Write-Host 'C++ 运行库已安装；如果本次自检失败，请重启电脑后再次启动。'
    } elseif ($process.ExitCode -notin @(0, 1638)) {
        throw "Microsoft C++ 运行库安装失败（代码 $($process.ExitCode)），请联系实验室管理员。"
    }
}

try {
    if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or [Environment]::OSVersion.Version.Major -lt 10) {
        throw '此部署包支持 Windows 10/11。其他系统需要单独打包。'
    }
    $architecture = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $architecture = $env:PROCESSOR_ARCHITEW6432 }
    if ($architecture -ne 'AMD64') { throw '此部署包需要 Intel/AMD 64 位 Windows，暂不支持 32 位或 ARM64 系统。' }

    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    Start-Transcript -LiteralPath (Join-Path $outputDirectory 'startup.log') -Append | Out-Null
    $transcribing = $true
    Write-Host '面部表情识别 · 正在检查运行环境'
    Write-Host '环境安装在项目的 .runtime 文件夹，不修改系统 Python。'

    $configuration = Get-Content -Raw -Encoding UTF8 -LiteralPath $lockPath | ConvertFrom-Json
    foreach ($model in $configuration.models) {
        $modelPath = Join-Path $root $model.file
        if (-not (Test-Path -LiteralPath $modelPath)) {
            throw "模型缺失：$($model.file)。请从原电脑复制完整部署包；自训练模型不能自动从网上下载。"
        }
        if ((Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash -ne $model.sha256) {
            throw "模型校验失败：$($model.file)。请重新复制部署包。"
        }
    }

    try {
        $setupLock = [IO.File]::Open((Join-Path $root '.setup.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch {
        throw '另一个启动窗口正在检查或安装环境，请等待那个窗口完成后再试。'
    }
    # Model updates must not trigger reinstallation of unchanged Python packages.
    $requirements = Join-Path $root 'deployment\requirements-runtime.txt'
    $dependencySpec = [ordered]@{
        python = $configuration.python
        wheels = $configuration.wheels
        pip_wheel = $configuration.pip_wheel
        requirements_sha256 = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
    } | ConvertTo-Json -Depth 8 -Compress
    $dependencyHasher = [Security.Cryptography.SHA256]::Create()
    try {
        $lockHash = ([BitConverter]::ToString($dependencyHasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($dependencySpec)))).Replace('-', '')
    } finally {
        $dependencyHasher.Dispose()
    }
    $readyMarker = Join-Path $runtime 'ready.sha256'
    $version = $configuration.python.version
    $pythonOK = Test-PythonCommand "import sys; assert sys.version.split()[0] == '$version'"
    if (-not $pythonOK) {
        $archive = Get-CheckedAsset $configuration.python
        New-Item -ItemType Directory -Path $runtime -Force | Out-Null
        Write-Host "正在配置独立 Python $version…"
        Expand-Archive -LiteralPath $archive -DestinationPath $runtime -Force
        # Explicit import paths make the embedded runtime relocatable and isolated.
        $pth = Join-Path $runtime $configuration.python.pth
        @($configuration.python.stdlib, '.', 'Lib\site-packages', '..', 'import site') | Set-Content -LiteralPath $pth -Encoding ASCII
        if (-not (Test-PythonCommand "import sys; assert sys.version.split()[0] == '$version'")) {
            Install-VCRuntime $configuration
            if (-not (Test-PythonCommand "import sys; assert sys.version.split()[0] == '$version'")) {
                throw '独立 Python 无法运行。请查看杀毒软件拦截记录，或重启后重试。'
            }
        }
    }

    $packagesOK = $false
    if ((Test-Path -LiteralPath $readyMarker) -and (Get-Content -Raw -LiteralPath $readyMarker).Trim() -eq $lockHash) {
        $packagesOK = Test-PythonCommand 'import cv2, numpy, onnxruntime; assert hasattr(cv2, ''FaceDetectorYN'')'
    }
    if (-not $packagesOK) {
        Write-Host '正在准备推理依赖（首次启动需要几分钟）…'
        foreach ($wheel in $configuration.wheels) { Get-CheckedAsset $wheel | Out-Null }
        $pipWheel = Join-Path $cache $configuration.pip_wheel
        $wheelDirectory = Join-Path $cache 'wheels'
        $requirements = Join-Path $root 'deployment\requirements-runtime.txt'
        $sitePackages = Join-Path $runtime 'Lib\site-packages'
        & $python -I -X utf8 -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module(''pip'', run_name=''__main__'')' $pipWheel install --disable-pip-version-check --no-warn-script-location --no-index --only-binary=:all: --require-hashes --find-links $wheelDirectory --target $sitePackages --upgrade -r $requirements
        if ($LASTEXITCODE -ne 0) { throw '依赖安装失败，请查看上面的提示与 output\startup.log。' }
        if (-not (Test-PythonCommand 'import ctypes; ctypes.CDLL(''msvcp140.dll''); ctypes.CDLL(''vcruntime140_1.dll'')')) {
            Install-VCRuntime $configuration
        }
        & $python -I -X utf8 -c 'import cv2, numpy, onnxruntime; assert hasattr(cv2, ''FaceDetectorYN''); print(''依赖导入检查通过'')'
        if ($LASTEXITCODE -ne 0) {
            throw '推理依赖无法加载。请检查 C++ 运行库或杀毒软件拦截，并将 startup.log 提供给维护人员。'
        }
    }

    & $python -I -X utf8 -u (Join-Path $PSScriptRoot 'launcher.py') --check
    if ($LASTEXITCODE -ne 0) { throw '模型推理自检失败，请查看日志。旧电脑上的训练环境无需复制。' }
    $lockHash | Set-Content -LiteralPath $readyMarker -Encoding ASCII
    $setupLock.Dispose()
    $setupLock = $null
    if ($PrepareOnly) {
        Write-Host '部署环境准备完成，可以离线启动。'
    } else {
        $launchArguments = @('-I', '-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'launcher.py'), '--port', "$Port")
        if ($NoBrowser) { $launchArguments += '--no-browser' }
        & $python @launchArguments
        if ($LASTEXITCODE -ne 0) { throw '服务已异常退出，请查看 output\startup.log。' }
    }
} catch {
    Write-Host "`n启动失败：$($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
} finally {
    if ($setupLock) { $setupLock.Dispose() }
    if ($transcribing) { Stop-Transcript | Out-Null }
}
exit $exitCode
