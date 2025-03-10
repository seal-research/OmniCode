#!/usr/bin/env python
import json

# Read the JSON file
with open('data/codearena_instances.json', 'r') as f:
    data = json.load(f)

# Extract Celery instance IDs
celery_instances = [
    item['instance_id'] 
    for item in data 
    if item.get('repo', '').startswith('celery/celery') or 
       item.get('instance_id', '').startswith('celery__')
]

# Print summary
print(f'Found {len(celery_instances)} Celery instances')

# Write to file
with open('celery_instances.txt', 'w') as f:
    f.write('\n'.join(celery_instances)) 