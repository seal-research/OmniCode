from pathlib import Path
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

REPOS_TO_INCLUDE = [
    "ytdl-org__youtube-dl",
    "scrapy__scrapy",
    "keras-team__keras",
    "fastapi__fastapi",
    "celery__celery",
    "camel-ai__camel",
]

def subsample(
    instances_path: str | Path,
    output_path: str | Path,
    total: int = 250,
):
    logger.info(f"Starting subsampling process")
    logger.info(f"Parameters: instances_path={instances_path}, output_path={output_path}, total={total}")
    
    random.seed(42)
    logger.debug("Random seed set to 42")
    
    instances_path, output_path = Path(instances_path), Path(output_path)
    
    logger.info(f"Reading instances from {instances_path}")
    all_instances = set([i.strip() for i in instances_path.read_text().strip().splitlines()])
    logger.info(f"Found {len(all_instances)} total instances")
    
    included_instances = []
    other_instances = []
    
    logger.info(f"Filtering instances by priority repositories")
    for instance in all_instances:
        included = False
        for repo in REPOS_TO_INCLUDE:
            if instance.startswith(repo):
                included_instances.append(instance)
                included = True
                break
        if not included:
            other_instances.append(instance)
    
    logger.info(f"Found {len(included_instances)} instances from priority repositories")
    logger.info(f"Found {len(other_instances)} instances from other repositories")
    
    if len(included_instances) > total:
        logger.info(f"Too many priority instances ({len(included_instances)}), randomly sampling {total}")
        final_instances = random.sample(included_instances, total)
    elif len(included_instances) == total:
        logger.info(f"Exact match: {total} priority instances available")
        final_instances = included_instances
    else:
        remaining = total - len(included_instances)
        logger.info(f"Using all {len(included_instances)} priority instances and sampling {remaining} additional instances")
        final_instances = included_instances + random.sample(other_instances, remaining)
    
    final_instances = sorted(final_instances)
    logger.info(f"Selected {len(final_instances)} instances")
    
    logger.info(f"Writing results to {output_path}")
    output_path.write_text('\n'.join(final_instances))
    logger.info("Subsampling complete")
        
if __name__ == '__main__':
    import fire
    fire.Fire(subsample)
