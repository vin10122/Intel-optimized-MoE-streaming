@echo off
setlocal enabledelayedexpansion

:: Compilazione DLL per dequantizzazione Q8_0 -> BF16 (versione completa con AVX2)
:: Richiede Visual Studio 2022 con componenti C++ installati

:: Crea directory output
if not exist flash_moe\nul mkdir flash_moe

:: Compila con AVX2 e OpenMP - versione AVX2 ottimizzata
echo [COMPILE] Compilazione fast_dequant_q8_0_avx2.cpp (AVX2 + OpenMP)...
cl /EHsc /O2 /GL /arch:AVX2 /openmp /LD /Qpar /Qvec scratch\fast_dequant_q8_0_avx2.cpp /link /OUT:flash_moe\fast_dequant_q8_0.dll /DLL /MACHINE:x64 /LTCG

if %errorlevel% neq 0 (
    echo [ERROR] Compilazione AVX2 fallita
    echo Provo con versione semplice...
    goto :simple
)

echo [COMPILE] DLL AVX2 creata: flash_moe\fast_dequant_q8_0.dll
echo [COMPILE] Funzioni esportate:
    dequantize_q8_0_to_bf16 (fallback semplice)
    dequantize_q8_0_to_bf16_avx2 (AVX2 ottimizzato)

goto :end

:simple
:: Versione semplice senza AVX2
cl /EHsc /O2 /openmp /LD scratch\fast_dequant_q8_0.cpp /link /OUT:flash_moe\fast_dequant_q8_0.dll /DLL /MACHINE:x64

if %errorlevel% neq 0 (
    echo [ERROR] Anche la compilazione semplice e' fallita
    pause
    exit /b 1
)

echo [COMPILE] DLL semplice creata: flash_moe\fast_dequant_q8_0.dll

:end
pause
