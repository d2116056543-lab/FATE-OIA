$ErrorActionPreference = 'SilentlyContinue'
Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match 'wsl|python|bash|cmd' -or
        $_.CommandLine -match 'SNNA_repro|main_dino|run_snna_full|multi_label_train'
    } |
    Select-Object ProcessId,Name,CommandLine |
    Format-List
