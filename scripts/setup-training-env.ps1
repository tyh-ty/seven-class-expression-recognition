$ErrorActionPreference = 'Stop'

$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$environment = Join-Path $root '.train-env'
$python = Join-Path $environment 'python.exe'

if (-not (Test-Path -LiteralPath $python)) {
  conda create --prefix $environment python=3.11 pip -y
  if ($LASTEXITCODE -ne 0) {
    throw "Conda environment creation failed with exit code $LASTEXITCODE."
  }
}

$torchUrl = 'https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp311-cp311-win_amd64.whl'
$torchvisionUrl = 'https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.26.0%2Bcu128-cp311-cp311-win_amd64.whl'
& $python -c "import torch, torchvision; assert torch.__version__.startswith('2.11.0+cu128'); assert torchvision.__version__.startswith('0.26.0+cu128')" 2>$null
$torchReady = $LASTEXITCODE -eq 0
if (-not $torchReady) {
  & $python -m pip install $torchUrl $torchvisionUrl
  if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Mirror installation failed; retrying the official CUDA 12.8 index.'
    & $python -m pip install 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
      throw "PyTorch installation failed with exit code $LASTEXITCODE."
    }
  }
}

& $python -m pip install -r (Join-Path $root 'training\requirements.txt')
if ($LASTEXITCODE -ne 0) {
  throw "Training dependency installation failed with exit code $LASTEXITCODE."
}

& $python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if ($LASTEXITCODE -ne 0) {
  throw "Training environment verification failed with exit code $LASTEXITCODE."
}

$env:HF_ENDPOINT = 'https://hf-mirror.com'
$env:HF_HUB_DISABLE_XET = '1'
& $python -c "import timm; timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=8); print('ViT-S/16 pretrained weights ready')"
if ($LASTEXITCODE -ne 0) {
  throw "Pretrained ViT download failed with exit code $LASTEXITCODE."
}
