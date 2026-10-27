# Source for the self-contained Windows launchers. Run build_launchers.py after editing.
[CmdletBinding()]
param(
    [string]$LauncherPath = $env:XIANGQI_LAUNCHER_PATH,
    [string]$ArchivePath = $env:XIANGQI_ARCHIVE_PATH,
    [string]$CacheRoot = $env:XIANGQI_LAUNCHER_HOME,
    [switch]$CheckOnly,
    [switch]$NoBrowser = ($env:XIANGQI_NO_BROWSER -eq '1'),
    [switch]$NoPrompt,
    [int]$Port = 8765,
    [switch]$LibraryOnly
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$script:AppFiles = @(
    'app.py', 'rules.py', 'engines.py', 'model_registry.py', 'model_worker.py',
    'policy_process.py', 'static/index.html', 'static/app.js', 'static/style.css',
    'engine/Pikafish-Windows-x86-64-universal.exe', 'engine/pikafish.nnue'
)
$script:RuntimeFiles = @('python.exe', 'python313.dll', 'Lib/encodings/__init__.py')

function Test-AppFolder([string]$Folder) {
    foreach ($name in $script:AppFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $Folder $name) -PathType Leaf)) { return $false }
    }
    return $true
}

function Test-Python([string]$Python) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { return $false }
    # Do not invoke Windows Store aliases. No system installation is performed.
    if ($Python -match '\\WindowsApps\\') { return $false }
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo.FileName = $Python
    $process.StartInfo.Arguments = '-I -c "import sys, encodings, json, http.server, ssl, ctypes; assert (3, 10) <= sys.version_info[:2] <= (3, 13); print(''XIANGQI_PYTHON_OK'')"'
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(10000)) {
            $process.Kill()
            Write-Host "Python check timed out: $Python"
            return $false
        }
        if ($process.ExitCode -eq 0 -and $stdout.Result.Trim() -eq 'XIANGQI_PYTHON_OK') { return $true }
        Write-Host "Python check failed: $Python"
        Write-Host $stderr.Result.Trim()
    } catch { Write-Host "Could not run Python: $Python ($($_.Exception.Message))" }
    finally { $process.Dispose() }
    return $false
}

function Find-Python([string]$AppFolder, [switch]$LocalOnly) {
    $parent = Split-Path -Parent $AppFolder
    $candidates = @(
        (Join-Path $AppFolder 'runtime/python/python.exe'),
        (Join-Path $parent '.runtime/python/python.exe'),
        (Join-Path $AppFolder '.venv/Scripts/python.exe'),
        (Join-Path $parent '.venv/Scripts/python.exe')
    )
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Python $candidate) { return $candidate }
    }
    $candidates = @()
    if (-not $LocalOnly) {
        foreach ($key in @('HKCU:\Software\Python\PythonCore\3.13\InstallPath', 'HKLM:\Software\Python\PythonCore\3.13\InstallPath', 'HKCU:\Software\Python\PythonCore\3.12\InstallPath', 'HKLM:\Software\Python\PythonCore\3.12\InstallPath', 'HKCU:\Software\Python\PythonCore\3.11\InstallPath', 'HKLM:\Software\Python\PythonCore\3.11\InstallPath', 'HKCU:\Software\Python\PythonCore\3.10\InstallPath', 'HKLM:\Software\Python\PythonCore\3.10\InstallPath')) {
            try {
                if (Test-Path -LiteralPath $key) {
                    $install = (Get-Item -LiteralPath $key).GetValue('')
                    if ($install) { $candidates += Join-Path $install 'python.exe' }
                }
            } catch { } # A restricted registry must not prevent using an offline ZIP.
        }
        $command = Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) { $candidates += $command.Source }
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Python $candidate) { return $candidate }
    }
    return $null
}

function Get-SafeEntryPath([string]$Base, [string]$Name) {
    $normalized = $Name.Replace('\', '/')
    if (-not $normalized.StartsWith('XiangqiLab/', [StringComparison]::Ordinal) -or
        $normalized.Contains(':') -or ($normalized.Split('/') -contains '..') -or
        ($normalized.Split('/') -contains '.')) { throw "Unsafe ZIP path: $Name" }
    $root = [IO.Path]::GetFullPath($Base).TrimEnd('\') + '\'
    $target = [IO.Path]::GetFullPath((Join-Path $Base $normalized))
    if (-not $target.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe ZIP path: $Name" }
    return $target
}

function Test-ReleaseArchive([IO.Compression.ZipArchive]$Zip, [string]$Target) {
    $names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    [long]$size = 0
    if ($Zip.Entries.Count -gt 20000) { throw 'Too many files in this ZIP.' }
    foreach ($entry in $Zip.Entries) {
        [void](Get-SafeEntryPath $Target $entry.FullName)
        if (-not $names.Add($entry.FullName.Replace('\', '/'))) { throw 'Duplicate paths in this ZIP.' }
        if ((($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000) { throw 'Links are not supported in a Windows ZIP.' }
        $size += $entry.Length
    }
    if ($size -gt 2GB) { throw 'This ZIP is larger than the supported offline package.' }
    foreach ($name in ($script:AppFiles + @('release.json') + @($script:RuntimeFiles | ForEach-Object { 'runtime/python/' + $_ }))) {
        if (-not $names.Contains('XiangqiLab/' + $name)) {
            throw "Incomplete Windows release ZIP: missing $name. A GitHub source ZIP does not include Python."
        }
    }
    $entry = $Zip.GetEntry('XiangqiLab/release.json')
    if (-not $entry -or $entry.Length -gt 16384) { throw 'Invalid release information.' }
    $reader = New-Object IO.StreamReader($entry.Open())
    try { $meta = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
    if ($meta.platform -ne 'Windows' -or -not $meta.offline) { throw 'Choose the complete Windows offline ZIP.' }
    return $meta
}

function Expand-Release([string]$Archive, [string]$PackageRoot) {
    $archiveFull = (Resolve-Path -LiteralPath $Archive).ProviderPath
    $packageFull = [IO.Path]::GetFullPath($PackageRoot)
    [void][IO.Directory]::CreateDirectory($packageFull)
    $hasher = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($archiveFull)
    try { $hash = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $hasher.Dispose() }
    $destination = Join-Path $packageFull ('windows-' + $hash.Substring(0, 16))
    # Read and validate before creating any extracted files. Never overwrite the source folder.
    $zip = [IO.Compression.ZipFile]::OpenRead($archiveFull)
    $stage = $null
    try {
        $meta = Test-ReleaseArchive $zip $destination
        $cachedApp = Join-Path $destination 'XiangqiLab'
        if ((Test-AppFolder $cachedApp) -and (Test-Python (Join-Path $cachedApp 'runtime/python/python.exe'))) {
            Write-Host "Using the unpacked Windows package ($($meta.version))."
            return $cachedApp
        }
        # Preserve a damaged previous copy, which may contain someone's models or saved files.
        if (Test-Path -LiteralPath $destination) { $destination += '-repair-' + [guid]::NewGuid().ToString('N') }
        $stage = Join-Path $packageFull ('.unpacking-' + [guid]::NewGuid().ToString('N'))
        [void][IO.Directory]::CreateDirectory($stage)
        Write-Host "Unpacking the full Windows package ($($meta.version)). Please wait..."
        foreach ($entry in $zip.Entries) {
            $path = Get-SafeEntryPath $stage $entry.FullName
            if ($entry.FullName.EndsWith('/')) { [void][IO.Directory]::CreateDirectory($path); continue }
            [void][IO.Directory]::CreateDirectory((Split-Path -Parent $path))
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $path, $false)
        }
        $app = Join-Path $stage 'XiangqiLab'
        if (-not (Test-AppFolder $app) -or -not (Test-Python (Join-Path $app 'runtime/python/python.exe'))) {
            throw 'The extracted files failed the Python check. Download or copy the complete Windows ZIP again.'
        }
        # If another launcher finished first, keep this complete copy under a unique name.
        if (Test-Path -LiteralPath $destination) { $destination += '-' + [guid]::NewGuid().ToString('N') }
        Move-Item -LiteralPath $stage -Destination $destination
        $stage = $null
        return (Join-Path $destination 'XiangqiLab')
    } finally {
        $zip.Dispose()
        if ($stage -and (Test-Path -LiteralPath $stage)) {
            $resolvedStage = (Resolve-Path -LiteralPath $stage).ProviderPath
            $safeRoot = $packageFull.TrimEnd('\') + '\'
            if ($resolvedStage.StartsWith($safeRoot, [StringComparison]::OrdinalIgnoreCase) -and
                (Split-Path -Leaf $resolvedStage).StartsWith('.unpacking-')) {
                Remove-Item -LiteralPath $resolvedStage -Recurse -Force
            }
        }
    }
}

function Find-ReleaseArchives([string[]]$Folders) {
    $found = @()
    foreach ($folder in ($Folders | Where-Object { $_ } | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $folder -PathType Container)) { continue }
        # Prefer a package beside the launcher over an unrelated download elsewhere.
        $found += @(Get-ChildItem -LiteralPath $folder -Filter 'XiangqiLab-Windows-*.zip' -File -ErrorAction SilentlyContinue |
            Sort-Object -Property @{Expression={
                if ($_.Name -match '-v(\d+\.\d+(?:\.\d+)?)\.zip$') { [version]$Matches[1] } else { [version]'0.0' }
            };Descending=$true}, @{Expression='LastWriteTime';Descending=$true})
    }
    return @($found | Select-Object -ExpandProperty FullName -Unique)
}

function Select-ReleaseArchive([string]$InitialFolder) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = 'Choose the complete Xiangqi Lab Windows ZIP (not the GitHub source ZIP)'
    $dialog.Filter = 'ZIP packages (*.zip)|*.zip'
    $dialog.InitialDirectory = $InitialFolder
    try {
        if ($dialog.ShowDialog() -eq [Windows.Forms.DialogResult]::OK) { return $dialog.FileName }
    } finally { $dialog.Dispose() }
    return $null
}

function Start-Xiangqi {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
        throw 'This package supports Windows x64 (Intel or AMD). Windows ARM is not covered.'
    }
    if (-not $LauncherPath) { throw 'Launcher path is missing. Use Start-Xiangqi-Windows.bat.' }
    $folder = Split-Path -Parent ([IO.Path]::GetFullPath($LauncherPath))
    $parent = Split-Path -Parent $folder
    Write-Host 'Xiangqi Lab - checking files and Python...'
    $app = $null
    $python = $null
    foreach ($candidate in @($folder, (Join-Path $folder 'xiangqi_lab'), (Join-Path $folder 'XiangqiLab'))) {
        if (Test-AppFolder $candidate) { $app = $candidate; break }
    }
    if ($app -and -not $ArchivePath) { $python = Find-Python $app }
    if (-not $python) {
        Write-Host 'Looking for a complete Windows ZIP. No Python installation is needed with that package.'
        if ($ArchivePath) {
            $archives = @($ArchivePath)
        } else {
            $downloads = Join-Path $env:USERPROFILE 'Downloads'
            $key = $null
            try { $key = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders' -ErrorAction SilentlyContinue } catch { }
            if ($key -and $key.'{374DE290-123F-4565-9164-39C4925E467B}') {
                $downloads = [Environment]::ExpandEnvironmentVariables($key.'{374DE290-123F-4565-9164-39C4925E467B}')
            }
            $folders = @($folder, (Join-Path $folder 'dist'), $parent)
            if ($parent) { $folders += Join-Path $parent 'dist' }
            $folders += @($downloads, [Environment]::GetFolderPath('DesktopDirectory'))
            $archives = @(Find-ReleaseArchives $folders)
        }
        $unpacked = $null
        foreach ($archive in $archives) {
            Write-Host "Package: $archive"
            try { $unpacked = Expand-Release $archive (Join-Path $CacheRoot 'packages'); break }
            catch {
                Write-Host $_.Exception.Message
                if ($ArchivePath) { throw }
            }
        }
        if (-not $unpacked -and -not $NoPrompt) {
            Write-Host 'If you opened this launcher inside a ZIP, Windows may have hidden its original location.'
            Write-Host 'Choose the full Windows release ZIP in the next window.'
            $selected = Select-ReleaseArchive $folder
            if ($selected) { $unpacked = Expand-Release $selected (Join-Path $CacheRoot 'packages') }
        }
        if (-not $unpacked) {
            throw 'No complete Windows package found. Put XiangqiLab-Windows-v*.zip beside this launcher and try again. A GitHub source ZIP has no Python runtime; source users can also install Python 3.13.'
        }
        # Use a package's runtime with the local source code if the source is complete.
        if (-not $app) { $app = $unpacked }
        $python = Join-Path $unpacked 'runtime/python/python.exe'
    }
    Write-Host "App folder: $app"
    Write-Host "Python: $python"
    if ($CheckOnly) { Write-Host 'Launch check passed.'; return }
    Write-Host 'Opening the website. Keep this window open; press Ctrl+C to stop.'
    # -E protects the main Python, but model workers also need a clean runtime location.
    # These changes affect this launcher and its children only, never system settings.
    $env:PYTHONHOME = $null
    $env:PYTHONPATH = $null
    $env:PYTHONUTF8 = '1'
    $entryPoint = Join-Path $app 'start.py'
    if (-not (Test-Path -LiteralPath $entryPoint -PathType Leaf)) { $entryPoint = Join-Path $app 'app.py' }
    $arguments = @('-E', '-s', '-X', 'utf8', $entryPoint, '--port', "$Port")
    if ($NoBrowser) { $arguments += '--no-browser' }
    Push-Location -LiteralPath $app
    try {
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "The app stopped with exit code $LASTEXITCODE. See the log above." }
    } finally { Pop-Location }
}

if (-not $LibraryOnly) {
    $transcript = $false
    try {
        if (-not $CacheRoot) { $CacheRoot = Join-Path $env:LOCALAPPDATA 'XiangqiLab' }
        $logFolder = Join-Path $CacheRoot 'logs'
        [void][IO.Directory]::CreateDirectory($logFolder)
        $log = Join-Path $logFolder ('launch-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '.log')
        Start-Transcript -LiteralPath $log -Force | Out-Null
        $transcript = $true
        Write-Host "Launch log: $log"
        Start-Xiangqi
    } catch {
        Write-Host ''
        Write-Host ('Could not start Xiangqi Lab: ' + $_.Exception.Message) -ForegroundColor Red
        if ($log) { Write-Host "Send this log to your teammate: $log" }
        exit 1
    } finally {
        if ($transcript) { Stop-Transcript | Out-Null }
    }
}
