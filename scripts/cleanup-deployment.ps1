param(
    [Parameter(Mandatory = $true)][string]$KeepZip,
    [switch]$Preview
)

$ErrorActionPreference = 'Stop'
$dist = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\dist'))
$keep = [IO.Path]::GetFullPath($KeepZip)
$releaseName = '^(?:seven-class-expression-system|emotion-lab)-windows-x64-\d{8}-\d{6}-\d{3}$'
$verificationName = '^verify-ckplus-model-\d{8}-\d{6}-\d{3}$'
$keepBase = [IO.Path]::GetFileNameWithoutExtension($keep)
if (-not $keep.StartsWith($dist + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
    [IO.Path]::GetDirectoryName($keep) -ne $dist -or
    [IO.Path]::GetExtension($keep) -ne '.zip' -or
    $keepBase -cnotmatch $releaseName -or
    -not (Test-Path -LiteralPath $keep -PathType Leaf)) {
    throw "要保留的部署 ZIP 无效或不在 dist 内：$keep"
}
$checksum = "$keep.sha256"
if (-not (Test-Path -LiteralPath $checksum -PathType Leaf)) {
    throw "缺少新包校验文件：$checksum"
}
$line = (Get-Content -LiteralPath $checksum -Raw).Trim()
$match = [regex]::Match($line, '^([A-Fa-f0-9]{64})\s+(.+\.zip)$')
if (-not $match.Success -or
    $match.Groups[2].Value -ne [IO.Path]::GetFileName($keep) -or
    (Get-FileHash -LiteralPath $keep -Algorithm SHA256).Hash -ne $match.Groups[1].Value) {
    throw '新包 SHA256 校验失败；旧包未清理。'
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($keep)
try {
    $entries = @($archive.Entries | ForEach-Object { $_.FullName.Replace([char]92, [char]47) })
    $runtimeLock = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot '..\deployment\runtime-lock.json') | ConvertFrom-Json
    $requiredEntries = @('/启动服务.cmd', '/start-demo.cmd', '/server.py', '/deployment/runtime-lock.json')
    $requiredEntries += @($runtimeLock.models | ForEach-Object { '/' + $_.file.Replace([char]92, [char]47) })
    foreach ($required in $requiredEntries) {
        if (-not @($entries | Where-Object { $_.EndsWith($required, [StringComparison]::Ordinal) }).Count) {
            throw "新包缺少必需文件 $required；旧包未清理。"
        }
    }
} finally {
    $archive.Dispose()
}

function Remove-OldArtifact([IO.FileSystemInfo]$Item) {
    $absolute = [IO.Path]::GetFullPath($Item.FullName)
    if (-not $absolute.StartsWith($dist + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        $Item.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
        throw "拒绝清理非 dist 内的路径或链接：$absolute"
    }
    if ($Item.PSIsContainer -and (Get-ChildItem -LiteralPath $absolute -Recurse -Force -Attributes ReparsePoint | Select-Object -First 1)) {
        throw "旧包目录包含链接，拒绝递归清理：$absolute"
    }
    if ($Preview) {
        Write-Host "待清理旧包：$absolute"
        return
    }
    if ($Item.PSIsContainer) {
        Remove-Item -LiteralPath $absolute -Recurse -Force
    } else {
        Remove-Item -LiteralPath $absolute -Force
    }
    Write-Host "已清理旧包：$absolute"
}

foreach ($item in Get-ChildItem -LiteralPath $dist -Force) {
    if ($item.FullName -eq $keep -or $item.FullName -eq $checksum) { continue }
    $base = $item.Name -replace '\.zip(?:\.sha256)?$', ''
    if ($base -cmatch $releaseName -or $base -cmatch $verificationName) { Remove-OldArtifact $item }
}

$withdrawn = Join-Path $dist 'withdrawn'
if (Test-Path -LiteralPath $withdrawn -PathType Container) {
    foreach ($item in Get-ChildItem -LiteralPath $withdrawn -Force) {
        $base = $item.Name -replace '\.zip(?:\.sha256)?$', ''
        if ($base -cmatch $releaseName) { Remove-OldArtifact $item }
    }
    if (-not (Get-ChildItem -LiteralPath $withdrawn -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $withdrawn
    }
}
