

Write-Host "========================================="
Write-Host "  OTA Update Build and Publish Script "
Write-Host "========================================="
Write-Host ""
Write-Host "Select the app to update:"
Write-Host "1. AIQ_Fleet app"
Write-Host "2. AIQ_authority app"
$appChoice = Read-Host "Enter your choice (1 or 2)"

$appName = ""
$appDir = ""
$xmlName = ""

if ($appChoice -eq "1") {
    $appName = "street_aiq_tipper_app"
    $appDir = "AIQ_Fleet app"
    $xmlName = "appcast_fleet.xml"
} elseif ($appChoice -eq "2") {
    $appName = "street_aiq_app"
    $appDir = "AIQ_authority app"
    $xmlName = "appcast_authority.xml"
} else {
    Write-Error "Invalid choice."
    exit
}

Write-Host ""
Write-Host "Selected App: $appName"
$versionStr = Read-Host "Enter new version (e.g., 1.0.1+2)"
$versionNumber = $versionStr.Split('+')[0]

if (-not $versionStr.Contains("+")) {
    Write-Error "Version must include build number, e.g., 1.0.1+2"
    exit
}

$releaseNotes = Read-Host "Enter release notes (optional, press Enter to skip)"
if ([string]::IsNullOrWhiteSpace($releaseNotes)) {
    $releaseNotes = "<li>Bug fixes and performance improvements.</li>"
} else {
    $releaseNotes = "<li>$releaseNotes</li>"
}

# Update pubspec.yaml
Write-Host "Updating pubspec.yaml version to $versionStr..."
$pubspecPath = "$appDir\pubspec.yaml"
$pubspecContent = Get-Content $pubspecPath
$pubspecContent = $pubspecContent -replace "^version:\s*.*", "version: $versionStr"
Set-Content -Path $pubspecPath -Value $pubspecContent

# Build the APK
Write-Host "Building APK..."
Push-Location $appDir
flutter clean
flutter build apk --release
Pop-Location

$apkSource = "$appDir\build\app\outputs\flutter-apk\app-release.apk"
$apkDestName = "${appName}_v${versionNumber}.apk"
$apkDest = "updates\$apkDestName"

if (Test-Path $apkSource) {
    Write-Host "Copying APK to updates folder..."
    Copy-Item $apkSource -Destination $apkDest -Force
} else {
    Write-Error "APK build failed, source file not found."
    exit
}

# Generate Appcast XML
$xmlPath = "updates\$xmlName"
$downloadUrl = "https://takemytrash.web.app/${apkDestName}"
Write-Host "Generating Appcast XML at $xmlPath..."

$xmlContent = @"
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
    <channel>
        <title>$appName Appcast</title>
        <description>Most recent updates to $appName</description>
        <language>en</language>
        <item>
            <title>Version $versionNumber</title>
            <description>
                <![CDATA[
                    <ul>
                        $releaseNotes
                    </ul>
                ]]>
            </description>
            <enclosure url="$downloadUrl" sparkle:version="$versionStr" sparkle:os="android" />
        </item>
    </channel>
</rss>
"@

Set-Content -Path $xmlPath -Value $xmlContent

Write-Host ""
Write-Host "Deploying to Firebase Hosting..."
firebase deploy --only hosting

Write-Host ""
Write-Host "Update published successfully to Firebase!"
Write-Host "Users anywhere in the world will receive this update."
