# Install llama-cpp-python on Windows using a PyPI *wheel* only (no local C++ compiler).
# If this fails, you must install Visual Studio Build Tools (see messages at end).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Py = Join-Path $PSScriptRoot "benchmark_env\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "benchmark_env\Scripts\python.exe not found. From repo root run: python -m venv benchmark_env"
}

Write-Host "==> Python info"
& $Py -c "import struct,sys; print(sys.version); print('Pointer size:', 8*struct.calcsize('P'), 'bits (need 64 for typical wheels)')"

Write-Host "==> Upgrading pip / wheel / setuptools"
& $Py -m pip install --upgrade pip setuptools wheel

# Drop any cached sdist so pip does not keep choosing .tar.gz over a wheel
Write-Host "==> Clearing pip cache entry for llama-cpp-python (if any)"
& $Py -m pip cache remove llama-cpp-python 2>$null
Write-Host "    (ignore errors if nothing was cached)"

Write-Host "==> Installing llama-cpp-python (wheel ONLY for this package — no source build)"
& $Py -m pip install --no-cache-dir "llama-cpp-python>=0.2.0" --only-binary=llama-cpp-python
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "--------------------------------------------------------------------"  -ForegroundColor Yellow
    Write-Host "WHY THIS HAPPENS" -ForegroundColor Yellow
    Write-Host "  pip had no matching pre-built wheel (e.g. 32-bit Python, odd ABI," -ForegroundColor Yellow
    Write-Host "  or no wheel for this version on PyPI for your OS)." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "FIX (pick one)" -ForegroundColor Yellow
    Write-Host "  A) Use 64-bit Python 3.10/3.11 from python.org, recreate venv, re-run this script." -ForegroundColor Yellow
    Write-Host "  B) Install MSVC so pip can compile from source:" -ForegroundColor Yellow
    Write-Host "     https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
    Write-Host "     Select workload: 'Desktop development with C++'" -ForegroundColor Yellow
    Write-Host "     Then open 'x64 Native Tools Command Prompt for VS 2022' and run:" -ForegroundColor Yellow
    Write-Host "       pip install llama-cpp-python" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------------------------"  -ForegroundColor Yellow
    exit 1
}

Write-Host "==> Verifying import"
& $Py -c "from llama_cpp import Llama; print('llama-cpp-python OK')"

Write-Host "Done."
