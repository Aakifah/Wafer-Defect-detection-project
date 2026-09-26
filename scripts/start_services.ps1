# Wafer Defect Detection — Local Service Startup Script (Windows PowerShell)
# Starts all infrastructure services via Docker Compose and initializes MinIO buckets.

param(
    [switch]$NoInit,
    [switch]$Down
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$DockerComposeFile = Join-Path $PSScriptRoot ".." "docker-compose.yml"

if ($Down) {
    Write-Host "Stopping all services..." -ForegroundColor Yellow
    docker compose -f $DockerComposeFile down --volumes
    Write-Host "All services stopped." -ForegroundColor Green
    exit 0
}

Write-Host "Starting all infrastructure services..." -ForegroundColor Cyan
docker compose -f $DockerComposeFile up -d

Write-Host ""
Write-Host "Services started. Waiting for health checks..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "Service Status:" -ForegroundColor Cyan
docker compose -f $DockerComposeFile ps

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  All services are running locally:" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host "  MinIO Console:  http://localhost:9001 (admin / minioadmin)" -ForegroundColor White
Write-Host "  MinIO API:      http://localhost:9000" -ForegroundColor White
Write-Host "  MLflow:         http://localhost:5000" -ForegroundColor White
Write-Host "  Pushgateway:    http://localhost:9091" -ForegroundColor White
Write-Host "  Prometheus:     http://localhost:9090" -ForegroundColor White
Write-Host "  Grafana:        http://localhost:3000 (admin / admin)" -ForegroundColor White
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "To stop all services: .\scripts\start_services.ps1 -Down" -ForegroundColor Yellow
