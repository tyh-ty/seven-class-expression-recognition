#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_TEMP=""
on_exit() {
  local status=$?
  # This directory is created by mktemp inside this installation only.
  if [[ -n "$PYTHON_TEMP" && -n "${ROOT:-}" ]]; then
    case "$PYTHON_TEMP" in
      "$ROOT/.runtime-linux"/setup.*) rm -rf -- "$PYTHON_TEMP" ;;
    esac
  fi
  if (( status != 0 )); then
    printf '\n启动失败，请查看上方提示；修复后重新双击即可继续。\n' >&2
    if [[ -t 0 ]]; then
      read -r -p '按回车关闭窗口…' || true
    fi
  fi
  return "$status"
}
trap on_exit EXIT

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "错误：这个启动脚本只能在 Linux 中运行。" >&2
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" && "$ARCH" != "amd64" ]]; then
  echo "错误：当前包需要 Intel/AMD 64 位 Linux（x86_64），检测到：$ARCH" >&2
  exit 1
fi

for command in tar sha256sum flock; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "错误：系统缺少基础工具 $command（需要 tar、coreutils、util-linux）。" >&2
    exit 1
  fi
done

# Serialize setup for repeated clicks; release before running the server.
exec 9>"$ROOT/.linux-setup.lock"
if ! flock -n 9; then
  echo "另一个启动窗口正在配置环境，等待完成…"
  flock 9
fi

ENV_DIR="$ROOT/.venv"
VENV_PYTHON="$ENV_DIR/bin/python"
CACHE_DIR="$ROOT/deployment/linux-cache"
# The bundled binary wheels target CPython 3.12. Recreate older environments.
if [[ ! -x "$VENV_PYTHON" ]] || ! "$VENV_PYTHON" -I -c 'import sys, ssl, pip; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1; then
  source "$ROOT/deployment/linux-python.env"
  RUNTIME_DIR="$ROOT/.runtime-linux"
  PYTHON="$RUNTIME_DIR/python/bin/python3"
  mkdir -p "$RUNTIME_DIR"
  if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -I -c 'import ssl, venv, ensurepip' >/dev/null 2>&1; then
    ARCHIVE="$CACHE_DIR/$LINUX_PYTHON_ARCHIVE"
    valid_archive() {
      [[ -f "$1" ]] || return 1
      local digest
      digest=$(sha256sum < "$1")
      [[ "${digest%% *}" == "$LINUX_PYTHON_SHA256" ]]
    }
    if ! valid_archive "$ARCHIVE"; then
      echo "错误：包内 Python 离线资源缺失或校验失败。请重新完整解压部署包。" >&2
      exit 1
    fi
    echo "正在从包内安装独立 Python $LINUX_PYTHON_VERSION（无需联网）…"
    PYTHON_TEMP=$(mktemp -d "$RUNTIME_DIR/setup.XXXXXXXX")
    tar -xzf "$ARCHIVE" -C "$PYTHON_TEMP"
    if ! "$PYTHON_TEMP/python/bin/python3" -I -c 'import ssl, venv, ensurepip'; then
      echo "错误：独立 Python 无法运行。请使用 Ubuntu 22.04 / Debian 12 或更新的 x86_64 Linux。" >&2
      exit 1
    fi
    if [[ -e "$RUNTIME_DIR/python" ]]; then
      mv -- "$RUNTIME_DIR/python" "$RUNTIME_DIR/python.previous.$(date +%s).$$"
    fi
    mv -- "$PYTHON_TEMP/python" "$RUNTIME_DIR/python"
    rmdir -- "$PYTHON_TEMP"
    PYTHON_TEMP=""
  fi
  echo "正在创建项目内独立识别环境…"
  if [[ -e "$ENV_DIR" ]]; then
    mv -- "$ENV_DIR" "$ROOT/.venv.previous.$(date +%s).$$"
  fi
  "$PYTHON" -I -m venv "$ENV_DIR"
fi

REQUIREMENTS="$ROOT/requirements-runtime-linux.txt"
STAMP="$ENV_DIR/.runtime-requirements-installed"
REQUIREMENTS_HASH=$(sha256sum < "$REQUIREMENTS")
REQUIREMENTS_HASH=${REQUIREMENTS_HASH%% *}
INSTALLED_HASH=""
if [[ -f "$STAMP" ]]; then INSTALLED_HASH=$(cat "$STAMP"); fi
if [[ "$INSTALLED_HASH" != "$REQUIREMENTS_HASH" ]] || ! "$VENV_PYTHON" -I -c 'import cv2, numpy, onnxruntime' >/dev/null 2>&1; then
  if [[ ! -f "$CACHE_DIR/SHA256SUMS" ]] || ! (cd "$CACHE_DIR" && sha256sum --strict -c SHA256SUMS >/dev/null); then
    echo "错误：包内离线资源缺失或损坏。请重新完整解压部署包。" >&2
    exit 1
  fi
  echo "正在从包内安装识别依赖（无需联网）…"
  "$VENV_PYTHON" -I -m pip --isolated install --no-index --no-cache-dir \
    --find-links "$CACHE_DIR/wheels" --only-binary=:all: --disable-pip-version-check -r "$REQUIREMENTS"
  "$VENV_PYTHON" -I -c 'import cv2, numpy, onnxruntime'
  "$VENV_PYTHON" -I -X utf8 "$ROOT/scripts/launcher.py" --check
  printf '%s\n' "$REQUIREMENTS_HASH" > "$STAMP"
fi

flock -u 9
exec 9>&-
"$VENV_PYTHON" -I -X utf8 "$ROOT/scripts/launcher.py" "$@"
