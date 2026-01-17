#!/usr/bin/env python3
"""
Test script to verify batch processing setup before running on full dataset.
This script tests a small subset of instances to ensure everything works correctly.
"""

import json
import sys
from pathlib import Path

# Add the auto-code-rover directory to the path
sys.path.insert(0, str(Path(__file__).parent / "auto-code-rover"))

from acr_batch_runner import load_config, run_batch_processing


def test_batch_setup():
    """Test the batch processing setup with a small subset."""
    
    print("=" * 60)
    print("AutoCodeRover Batch Processing - Setup Test")
    print("=" * 60)
    
    # Load configuration
    config_file = Path(__file__).parent / "batch_config.json"
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        return False
    
    try:
        config = load_config(config_file)
        print(f"✅ Configuration loaded from {config_file}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Check input file
    input_tasks = Path(config["input_tasks"])
    if not input_tasks.exists():
        print(f"❌ Input tasks file not found: {input_tasks}")
        return False
    print(f"✅ Input tasks file found: {input_tasks}")
    
    # Load a small subset of tasks for testing
    try:
        from acr_runner import load_tasks
        all_tasks = load_tasks(input_tasks)
        test_tasks = all_tasks[:2]  # Test with first 2 instances
        print(f"✅ Loaded {len(all_tasks)} total tasks, testing with {len(test_tasks)}")
    except Exception as e:
        print(f"❌ Failed to load tasks: {e}")
        return False
    
    # Check environment variables
    import os
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY environment variable not set")
        return False
    print("✅ OPENROUTER_API_KEY environment variable is set")
    
    # Create test output directory
    test_output_dir = Path(__file__).parent / "test_batch_output"
    test_output_dir.mkdir(exist_ok=True)
    print(f"✅ Test output directory created: {test_output_dir}")
    
    # Test with just one mode to save time
    test_modes = ["bugfixing"]
    
    print("\n" + "=" * 60)
    print("Starting test run...")
    print("=" * 60)
    
    try:
        # Run test batch processing
        run_batch_processing(
            input_tasks=input_tasks,
            output_dir=test_output_dir,
            model_name=config["model_name"],
            acr_root=Path(config["acr_root"]),
            modes=test_modes,
            instance_ids=",".join([task["instance_id"] for task in test_tasks]),
            save_intermediate=True,
            intermediate_save_interval=1,
            max_retries=1,
            timeout_seconds=1800  # 30 minutes for test
        )
        
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Check results
        for mode in test_modes:
            mode_dir = test_output_dir / mode
            if mode_dir.exists():
                patch_files = list(mode_dir.glob("*.patch"))
                results_file = mode_dir / f"{mode}_results.jsonl"
                
                print(f"Mode {mode}:")
                print(f"  - Patch files: {len(patch_files)}")
                print(f"  - Results file: {'✅' if results_file.exists() else '❌'}")
                
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = [json.loads(line) for line in f]
                    successful = sum(1 for r in results if r.get("success", False))
                    print(f"  - Successful: {successful}/{len(results)}")
        
        print(f"\nTest results saved to: {test_output_dir}")
        print("You can now run the full batch processing on your cluster!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_batch_setup()
    sys.exit(0 if success else 1) 