$outFile = "pinggy_out.txt"
$errFile = "pinggy_err.txt"

# Remove old files if they exist
if (Test-Path $outFile) { Remove-Item $outFile -Force }
if (Test-Path $errFile) { Remove-Item $errFile -Force }

Write-Output "Starting SSH tunnel to Pinggy..."
# Start the SSH process in the background, redirecting output
$proc = Start-Process ssh -ArgumentList "-o StrictHostKeyChecking=no -p 443 -R0:localhost:5173 free@a.pinggy.io" -RedirectStandardOutput $outFile -RedirectStandardError $errFile -PassThru -NoNewWindow

# Wait up to 10 seconds for the URL to be generated in the output file
$url = $null
for ($i = 0; $i -lt 10; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $outFile) {
        $content = Get-Content $outFile -Raw
        if ($content -match "(https://[a-zA-Z0-9.-]+\.pinggy\.(link|xyz|io))") {
            $url = $matches[1]
            break
        }
    }
}

if ($url) {
    Write-Output "==========================================="
    Write-Output "SUCCESS: Public HTTPS Tunnel established!"
    Write-Output "URL: $url"
    Write-Output "==========================================="
} else {
    Write-Output "Failed to establish tunnel."
    if (Test-Path $errFile) {
        Write-Output "--- SSH Error Log ---"
        Get-Content $errFile
    }
    if (Test-Path $outFile) {
        Write-Output "--- SSH Output Log ---"
        Get-Content $outFile
    }
}
