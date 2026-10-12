$ErrorActionPreference = 'Stop'

$modelDirectory = Join-Path $PSScriptRoot '..\models'
$partDirectory = Join-Path $modelDirectory 'download-parts'
$modelUrl = 'https://media.githubusercontent.com/media/KenAKAFrosty/onnx-model-zoo/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-12-int8.onnx'
$modelSize = 19300656
$partCount = 10
$partSize = [Math]::Ceiling($modelSize / $partCount)
$modelPath = Join-Path $modelDirectory 'emotion-ferplus-12-int8.onnx'
$expectedHash = '3e47195d79e9593294df9e81a6d296a1e10969b68a717284081c29493a0ff5f1'
$faceDetectorPath = Join-Path $modelDirectory 'face_detection_yunet_2023mar.onnx'
$faceDetectorUrl = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
$faceDetectorHash = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
$transformerModelPath = Join-Path $modelDirectory 'mobilevit_va_mtl.onnx'
$transformerModelUrl = 'https://api.github.com/repos/sb-ai-lab/EmotiEffLib/git/blobs/bed5781a1af9924536638ac3786603e124ee900e'
$transformerModelHash = '93a2bee1f2e0b04e313c695078e9351c75e8fb4c457af4268a1716cb35601b0c'

function Install-TransformerEmotionModel {
  New-Item -ItemType Directory -Path $modelDirectory -Force | Out-Null
  if (Test-Path -LiteralPath $transformerModelPath) {
    $existingTransformerHash = (Get-FileHash -LiteralPath $transformerModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($existingTransformerHash -eq $transformerModelHash) {
      return
    }
  }
  & curl.exe -sS -L --fail --retry 5 --retry-all-errors --max-time 900 `
    -H 'Accept: application/vnd.github.raw+json' `
    -H 'User-Agent: emotion-recognition-demo' `
    -o $transformerModelPath $transformerModelUrl
  if ($LASTEXITCODE -ne 0) {
    throw "MobileViT download failed with exit code $LASTEXITCODE."
  }
  $downloadedTransformerHash = (Get-FileHash -LiteralPath $transformerModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($downloadedTransformerHash -ne $transformerModelHash) {
    throw "MobileViT checksum mismatch: $downloadedTransformerHash"
  }
}

function Install-FaceDetector {
  if (Test-Path -LiteralPath $faceDetectorPath) {
    $existingFaceHash = (Get-FileHash -LiteralPath $faceDetectorPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($existingFaceHash -eq $faceDetectorHash) {
      return
    }
  }
  & curl.exe -sS -L --fail --retry 5 --retry-all-errors --max-time 120 -o $faceDetectorPath $faceDetectorUrl
  if ($LASTEXITCODE -ne 0) {
    throw "YuNet download failed with exit code $LASTEXITCODE."
  }
  $downloadedFaceHash = (Get-FileHash -LiteralPath $faceDetectorPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($downloadedFaceHash -ne $faceDetectorHash) {
    throw "YuNet checksum mismatch: $downloadedFaceHash"
  }
}

function Remove-TemporaryModelFiles {
  $temporaryFiles = @(
    (Join-Path $modelDirectory 'chunk-test.bin'),
    (Join-Path $modelDirectory 'face_landmarker.task')
  )
  foreach ($temporaryFile in $temporaryFiles) {
    if (Test-Path -LiteralPath $temporaryFile) {
      Remove-Item -LiteralPath $temporaryFile -Force
    }
  }
  if (Test-Path -LiteralPath $partDirectory) {
    Get-ChildItem -LiteralPath $partDirectory -File | Remove-Item -Force
    Remove-Item -LiteralPath $partDirectory -Force
  }
}

Install-TransformerEmotionModel

if (Test-Path -LiteralPath $modelPath) {
  $existingHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($existingHash -eq $expectedHash) {
    Install-FaceDetector
    Remove-TemporaryModelFiles
    Write-Output "Model already ready: $modelPath"
    Write-Output "SHA256: $existingHash"
    Write-Output "Transformer model ready: $transformerModelPath"
    Write-Output "SHA256: $transformerModelHash"
    exit 0
  }
}

New-Item -ItemType Directory -Path $partDirectory -Force | Out-Null

$processes = for ($index = 0; $index -lt $partCount; $index++) {
  $rangeStart = $index * $partSize
  $rangeEnd = [Math]::Min($modelSize - 1, (($index + 1) * $partSize) - 1)
  $partPath = Join-Path $partDirectory ('part-{0:D2}.bin' -f $index)
  $arguments = @(
    '-sS', '-L', '--fail', '--retry', '5', '--retry-all-errors',
    '--max-time', '600', '-r', "$rangeStart-$rangeEnd",
    '-o', $partPath, $modelUrl
  )

  $process = Start-Process `
    -FilePath 'curl.exe' `
    -ArgumentList $arguments `
    -WindowStyle Hidden `
    -PassThru

  [PSCustomObject]@{
    Index = $index
    ExpectedBytes = $rangeEnd - $rangeStart + 1
    Path = $partPath
    Process = $process
  }
}

foreach ($item in $processes) {
  $item.Process.WaitForExit()
  if ($item.Process.ExitCode -ne 0) {
    throw "Model part $($item.Index) failed with exit code $($item.Process.ExitCode)."
  }

  $actualBytes = (Get-Item -LiteralPath $item.Path).Length
  if ($actualBytes -ne $item.ExpectedBytes) {
    throw "Model part $($item.Index) has $actualBytes bytes; expected $($item.ExpectedBytes)."
  }
}

$output = [System.IO.File]::Open($modelPath, [System.IO.FileMode]::Create)
try {
  foreach ($item in $processes | Sort-Object Index) {
    $input = [System.IO.File]::OpenRead($item.Path)
    try {
      $input.CopyTo($output)
    } finally {
      $input.Dispose()
    }
  }
} finally {
  $output.Dispose()
}

$actualModelSize = (Get-Item -LiteralPath $modelPath).Length
if ($actualModelSize -ne $modelSize) {
  throw "Combined model has $actualModelSize bytes; expected $modelSize."
}

$hash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $expectedHash) {
  throw "Model checksum mismatch: $hash"
}

Install-FaceDetector
Remove-TemporaryModelFiles

Write-Output "Model ready: $modelPath"
Write-Output "SHA256: $hash"
Write-Output "Transformer model ready: $transformerModelPath"
Write-Output "SHA256: $transformerModelHash"
