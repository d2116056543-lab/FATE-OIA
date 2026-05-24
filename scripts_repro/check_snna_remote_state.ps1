$ErrorActionPreference = 'Stop'
Set-Location 'E:\sbw\SNNA_repro\SNNA'

Write-Host '---ROOT---'
Get-Location
Write-Host '---GIT---'
git rev-parse HEAD
Write-Host '---DATA---'
Get-ChildItem dataset | Select-Object Name,Mode,LinkType,Target
Write-Host '---MAIN_DINO_HEAD---'
Get-Content main_dino.py -TotalCount 80
