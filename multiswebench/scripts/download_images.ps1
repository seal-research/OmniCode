# Check if filename is provided as argument
if ($args.Count -ne 1) {
    Write-Host "Usage: .\pull_images.ps1 <image list file>"
    exit 1
}

# Read filename
$imageFile = $args[0]

# Check if file exists
if (-Not (Test-Path $imageFile)) {
    Write-Host "File $imageFile does not exist."
    exit 1
}

# Read file line by line and download images
Get-Content $imageFile | ForEach-Object {
    $image = $_.Trim()
    
    # Skip empty lines
    if (-Not [string]::IsNullOrWhiteSpace($image)) {
        Write-Host "Downloading image: $image"
        docker pull $image
    }
}

Write-Host "All images downloaded successfully."