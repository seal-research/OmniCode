# Filtered Code Instances Dataset

This dataset contains 300 filtered code instances extracted from a larger collection.

## Dataset Description

- **Instance count:** 300
- **Format:** JSON
- **License:** Apache-2.0

## Contents

This dataset includes the following instance IDs:

```
astropy__astropy-13453, astropy__astropy-14508, camel-ai__camel-1363, camel-ai__camel-1627, django__django-11141, django__django-11163, django__django-11179, django__django-11206, django__django-11211, django__django-11265...
```

## Dataset Structure

Each instance in the dataset contains the following key fields:

- `repo`: Repository name
- `instance_id`: Unique identifier for the instance
- `base_commit`: Base commit hash
- `patch`: Code patch representing the change
- `test_patch`: Test patch for verification
- `problem_statement`: Description of the issue
- Other metadata related to the code change

## Usage

```python
# Example code to load the dataset
import json
from datasets import load_dataset

# Using the Hugging Face datasets library
dataset = load_dataset("your-username/your-dataset-name")
instances = dataset["train"]

# Or load directly from the JSON file
with open('instances.json', 'r') as f:
    instances = json.load(f)
```

## Citations

Please cite this dataset appropriately if you use it in your work.
