#!/bin/bash
# Script to run all the celery instances in the codearena environment

# Check if conda is activated
if [[ -z "${CONDA_DEFAULT_ENV}" || "${CONDA_DEFAULT_ENV}" != "codearena" ]]; then
  echo "Please activate the codearena conda environment first:"
  echo "conda activate codearena"
  exit 1
fi

# Create a results file
echo "instance_id,status" > celery_results.csv

# Read the instance_ids from the celery_instances.txt file
while read instance_id; do
  echo "Processing instance: $instance_id"
  
  # Export the instance_id as an environment variable
  export INSTANCE_ID="$instance_id"
  
  # Run the validate_instance.sh script
  ./validate_instance.sh
  
  # Check if the instance was resolved
  if grep -q "\"instances_resolved\": 1" gold.test.json; then
    status="RESOLVED"
    echo "✅ $instance_id: RESOLVED"
  else
    status="UNRESOLVED"
    echo "❌ $instance_id: UNRESOLVED"
  fi
  
  # Save the result to the CSV file
  echo "$instance_id,$status" >> celery_results.csv
  
  # Clean up logs to save space
  rm -rf logs
done < celery_instances.txt

# Print summary
echo ""
echo "=== SUMMARY ==="
echo "Resolved instances:"
grep "RESOLVED" celery_results.csv | cut -d',' -f1
echo ""
echo "Results saved to celery_results.csv"

