@echo off
setlocal

:: Compila la DLL dei kernel K-quant (Q8_0, Q3_K, Q4_K, Q5_K, Q6_K).
:: Richiede Visual Studio 2022 Build Tools con il toolset C++ x64.

set SCRIPT_DIR=%~dp0
set ROOT=%SCRIPT_DIR%..
set OUT_DIR=%ROOT%\src\flash_moe
set OUT_DLL=%OUT_DIR%\fast_dequant_k_parallel.dll

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

:: Individua vcvars64.bat
set VCVARS=
for %%P in (
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do (
    if exist %%P (
        set "VCVARS=%%~P"
        goto :found
    )
)

echo [ERROR] vcvars64.bat non trovato. Installa Visual Studio 2022 con "C++ build tools".
exit /b 1

:found
echo [BUILD] usando %VCVARS%
call "%VCVARS%" >nul

echo [BUILD] compilando cpp\dequant_k.cpp (AVX2 + F16C)
pushd "%ROOT%"
:: /arch:AVX2 non basta da solo per gli intrinsic F16C (_mm256_cvtps_ph):
:: il path f16 e' quello che genera i pesi per OpenVINO, ed e' il piu' caldo.
cl /nologo /EHsc /O2 /arch:AVX2 /LD /W3 ^
   cpp\dequant_k.cpp ^
   /link /OUT:"%OUT_DLL%" /DLL /MACHINE:x64
set RC=%errorlevel%
popd

if not %RC% == 0 (
    echo [ERROR] compilazione fallita
    exit /b %RC%
)

echo [BUILD] OK: %OUT_DLL%
echo [BUILD] funzioni esportate:
echo     dequant_k(type, src, dst, n_elements, want_f16)
echo     dequant_k_supported(type)
echo     dequant_k_block_size(type)
echo [BUILD] verifica con:  python tools\test_dequant_k.py
exit /b 0
