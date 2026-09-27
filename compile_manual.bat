@echo off
:: Questo script imposta l'ambiente VS e compila

:: Imposta variabili d'ambiente per x64
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

:: Vai nella cartella del progetto
cd /d %~dp0

echo Compilazione in corso...
echo.

:: Compila con AVX2
cl /EHsc /O2 /GL /arch:AVX2 /openmp /LD cpp\fast_dequant_q8_0_avx2.cpp /link /OUT:src\flash_moe\fast_dequant_q8_0.dll /DLL /MACHINE:x64 /LTCG

if %errorlevel% neq 0 (
    echo.
    echo Compilazione AVX2 fallita, provo versione semplice...
    echo.
    cl /EHsc /O2 /openmp /LD cpp\fast_dequant_q8_0.cpp /link /OUT:src\flash_moe\fast_dequant_q8_0.dll /DLL
)

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo SUCCESSO! DLL creata in:
    echo src\flash_moe\fast_dequant_q8_0.dll
    echo ============================================
) else (
    echo.
    echo ============================================
    echo ERRore nella compilazione
    echo ============================================
)

pause
