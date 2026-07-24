$Env:CONDA_DEFAULT_ENV = "311"
$Env:CONDA_PREFIX = "C:\Users\schulz0022\conda-envs\311"
$Env:PATH = @(
    "C:\Users\schulz0022\conda-envs\311",
    "C:\Users\schulz0022\conda-envs\311\Library\mingw-w64\bin",
    "C:\Users\schulz0022\conda-envs\311\Library\usr\bin",
    "C:\Users\schulz0022\conda-envs\311\Library\bin",
    "C:\Users\schulz0022\conda-envs\311\Scripts",
    "C:\Users\schulz0022\conda-envs\311\bin",
    "C:\windows\system32",
    "C:\windows",
    "C:\windows\System32\Wbem",
    "C:\windows\System32\WindowsPowerShell\v1.0\",
    "C:\windows\System32\OpenSSH\",
    "C:\ProgramData\chocolatey\bin",
    "C:\Program Files\Microsoft VS Code\bin",
    "C:\Users\schulz0022\AppData\Local\Microsoft\WindowsApps",
    "c:\Users\schulz0022\.vscode\extensions\openai.chatgpt-26.311.21342-win32-x64\bin\windows-x86_64"
) -join ";"

Write-Host "Codex PowerShell environment set to conda env 311."
Write-Host "Python:" (& "C:\Users\schulz0022\conda-envs\311\python.exe" --version)
