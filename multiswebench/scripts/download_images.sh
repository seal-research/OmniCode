#!/bin/bash

# Check if a filename is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <image list file>"
    exit 1
fi

# Read the filename
image_file="$1"

# Check if the file exists
if [ ! -f "$image_file" ]; then
    echo "File $image_file does not exist."
    exit 1
fi

# Read file line by line and download images
while IFS= read -r image; do
    # Skip empty lines
    if [ -z "$image" ]; then
        continue
    fi
    
    echo "Downloading image: $image"
    docker pull "$image"
done < "$image_file"

echo "All images have been downloaded."