@echo off
setlocal

:: Compila Q8_0 dequantizer con VS 2022 Build Tools
set CL_PATH="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"

%CL_PATH% cpp\fast_dequant_q8_0_avx2.cpp /EHsc /O2 /GL /arch:AVX2 /openmp /LD /Qpar /Qvec /link /OUT:src\flash_moe\fast_dequant_q8_0.dll /DLL /MACHINE:x64 /LTCG

if %errorlevel% neq 0 (
    echo Compilazione AVX2 fallita, provo versione semplice...
    %CL_PATH% cpp\fast_dequant_q8_0.cpp /EHsc /O2 /openmp /LD /link /OUT:src\flash_moe\fast_dequant_q8_0.dll /DLL /MACHINE:x64
)

pause
