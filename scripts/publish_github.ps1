# Publicar salares_web en TeleAmb-upla/salares_web
# Ejecutar desde la raíz del proyecto:  .\scripts\publish_github.ps1

$ErrorActionPreference = "Stop"
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

Set-Location $PSScriptRoot\..

Write-Host "Comprobando sesión de GitHub..."
gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Inicia sesión (se abrirá el navegador o verás un código de un solo uso):"
    gh auth login -h github.com -p https -w
}

Write-Host ""
Write-Host "Creando repositorio y subiendo código..."
gh repo create TeleAmb-upla/salares_web --public --description "Visualizador web Anillo MESS - Salar de Huasco" --source=. --remote=origin --push

Write-Host ""
Write-Host "Listo: https://github.com/TeleAmb-upla/salares_web"
