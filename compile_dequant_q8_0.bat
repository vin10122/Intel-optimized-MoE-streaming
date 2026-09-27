@echo off
setlocal enabledelayedexpansion

:: Compilazione DLL per dequantizzazione Q8_0 -> BF16
:: Richiede Visual Studio 2022 con componenti C++ installati

:: Trova cl.exe
set CL_PATH=
for /f "delims=" %%i in ('where cl 2^>nul') do set CL_PATH=%%i

if not defined CL_PATH (
    echo Errore: cl.exe non trovato. Assicurarsi che Visual Studio 2022 sia installato.
    echo Per usare questo script da comandi Visual Studio, eseguire:
    echo   "Developer Command Prompt for VS 2022" e poi lanciare questo script
    pause
    exit /b 1
)

echo [COMPILE] Trovato cl.exe in: %CL_PATH%

:: Crea directory output
if not exist flash_moe\nul mkdir flash_moe

:: Compila con AVX2 e OpenMP
echo [COMPILE] Compilazione fast_dequant_q8_0.cpp...
cl /EHsc /O2 /arch:AVX2 /openmp /LD /LDd /Zi scratch\fast_dequant_q8_0.cpp /link /OUT:flash_moe\fast_dequant_q8_0.dll /DLL /MACHINE:x64

if %errorlevel% neq 0 (
    echo [ERROR] Compilazione fallita
    pause
    exit /b 1
)

echo [COMPILE] DLL creata: flash_moe\fast_dequant_q8_0.dll
pause
