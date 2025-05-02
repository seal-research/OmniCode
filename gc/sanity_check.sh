#!/bin/bash
# This job is responsible for instances {start_index} through {end_index}

# Loop through the assigned instance range
for INSTANCE_ID in $(seq {start_index} {end_index}); do
    echo "Processing instance $INSTANCE_ID"
    
    # Download your data from bucket
    gsutil cp gs://{data_bucket}/instances/$INSTANCE_ID.data /tmp/input.data
    
    # Run your processing script
    python {script_path} --input /tmp/input.data --output /tmp/output.data
    
    # Upload results back to bucket
    gsutil cp /tmp/output.data gs://{data_bucket}/results/$INSTANCE_ID.result
    
    echo "Completed instance $INSTANCE_ID"
done