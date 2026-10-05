<#
.SYNOPSIS
    Create a desktop shortcut that opens the dashboard, with a global hotkey.

.DESCRIPTION
    Windows only honours a shortcut's hotkey when the .lnk lives on the Desktop
    or in the Start Menu, so the shortcut is written to the Desktop.

    Default hotkey: Ctrl + Alt + S.

.PARAMETER Hotkey
    The key combination to assign, in WScript.Shell form, e.g. "CTRL+ALT+S".

.PARAMETER Remove
    Delete the shortcut instead of creating it.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1 -Hotkey "CTRL+ALT+F"
    powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string]$Hotkey = "CTRL+ALT+S",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "Open Dashboard.bat"
$icon = Join-Path $root "dashboard\favicon.svg"
$desktop = [Environment]::GetFolderPath("Desktop")
$linkPath = Join-Path $desktop "Solar Flare Dashboard.lnk"

if ($Remove) {
    if (Test-Path -LiteralPath $linkPath) {
        Remove-Item -LiteralPath $linkPath -Force
        Write-Output "Removed $linkPath"
    } else {
        Write-Output "Nothing to remove; $linkPath does not exist."
    }
    return
}

if (-not (Test-Path -LiteralPath $target)) {
    throw "Launcher not found: $target"
}

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($linkPath)
$link.TargetPath       = $target
$link.WorkingDirectory = $root
$link.Description      = "Open the Solar Flare Prediction dashboard (academic prototype)"
$link.WindowStyle      = 7          # start minimised; the batch file opens the browser
$link.Hotkey           = $Hotkey
# A .lnk cannot use an SVG for its icon, so fall back to a stock one.
$link.IconLocation     = "$env:SystemRoot\System32\imageres.dll,109"
$link.Save()

Write-Output "Created $linkPath"
Write-Output "Hotkey  : $Hotkey"
Write-Output "Target  : $target"
Write-Output ""
Write-Output "Press $Hotkey from anywhere in Windows to open the dashboard."
Write-Output "The hotkey only works while the shortcut stays on the Desktop."
Write-Output "Remove it again with:  -Remove"
