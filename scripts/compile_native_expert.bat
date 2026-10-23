@echo off
setlocal
set "REPO_DIR=%~dp0.."
set "OUT_DIR=%REPO_DIR%\src\flash_moe"
set "BUILD_DIR=%REPO_DIR%\build\native"
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%P in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find VC\Auxiliary\Build\vcvars64.bat`) do set "VCVARS=%%P"
)
if not defined VCVARS (
    echo [ERROR] Visual Studio C++ x64 Build Tools not found.
    exit /b 1
)
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cl /nologo /EHsc /O2 /arch:AVX2 /LD /W3 ^
    /I"%REPO_DIR%\cpp" ^
    /Fo"%BUILD_DIR%\moe_quant_dot.obj" ^
    /Fd"%BUILD_DIR%\moe_quant_dot.pdb" ^
    "%REPO_DIR%\cpp\moe_quant_dot.cpp" ^
    /link /OUT:"%OUT_DIR%\moe_quant_dot.dll" ^
    /IMPLIB:"%BUILD_DIR%\moe_quant_dot.lib" /DLL /MACHINE:x64
if errorlevel 1 exit /b 1
echo [BUILD] Native decode kernel ready. The weight-conversion DLL is unchanged.
exit /b 0
