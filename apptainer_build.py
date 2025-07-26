from __future__ import annotations

import shutil
import subprocess

from swebench.harness.constants import DEF_IMAGE_BUILD_DIR, APPTAINER_BASH

from swebench.harness.docker_build import (
    close_logger,
    setup_logger,
)

from CodeArena_test_spec import get_test_specs_from_dataset

from swebench.harness.docker_build import BuildImageError

def build_sandbox(dataset):
    test_specs = get_test_specs_from_dataset(dataset)
    for test_spec in test_specs:
        # build setup file and put in logs/instance_id folder
        setup_scripts = {
                    "setup_env.sh": test_spec.setup_env_script,
                    "setup_repo.sh": test_spec.install_repo_script,
                }
        
        build_dir = DEF_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
        logger = setup_logger("def", build_dir / "build_image.log")
        logger.info("Building image def\n")

        try:
            # Pull the Apptainer base image
            logger.info("Pulling Apptainer image...")
            result = subprocess.run(
                [APPTAINER_BASH, "pull", "apptainer_base.sif", "docker://wellslu/apptainer_base:latest"],
                cwd=str(build_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                logger.info(f"Failed to pull Apptainer image:\n{result.stderr}")
                raise BuildImageError(
                    test_spec.instance_id,
                    f"Failed to pull Apptainer base image: {result.stderr}",
                    logger,
                )

            # Build the Apptainer base sandbox
            logger.info("Building Apptainer sandbox...")
            result = subprocess.run(
                [APPTAINER_BASH, "build", "--sandbox", "apptainer_sandbox", "apptainer_base.sif"],
                cwd=str(build_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                logger.info(f"Failed to build Apptainer sandbox image:\n{result.stderr}")
                raise BuildImageError(
                    test_spec.instance_id,
                    f"Failed to build Apptainer base sandbox: {result.stderr}",
                    logger,
                )

            # Write the setup scripts to the build directory and run setup scripts in the sandbox
            for setup_script_name, setup_script in setup_scripts.items():
                logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
                setup_script_path = build_dir / setup_script_name
                with open(setup_script_path, "w") as f:
                    f.write(setup_script)
                shutil.copy(setup_script_path, build_dir / f"apptainer_sandbox/root/{setup_script_name}")
            
                logger.info(f"Running {setup_script_name} in the Apptainer sandbox...")
            
                result = subprocess.run(
                    [APPTAINER_BASH, "exec", "--writable", "apptainer_sandbox", "bash", "-c", 
                    f"cd apptainer_sandbox && bash /root/{setup_script_name}"],
                    cwd=str(build_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    logger.info(f"Failed to run {setup_script_name} in Apptainer sandbox:\n{result.stderr}")
                    raise BuildImageError(
                        test_spec.instance_id,
                        f"Failed to run {setup_script_name} in Apptainer sandbox: {result.stderr}",
                        logger,
                    )

            logger.info(f"Apptainer sandbox image built successfully: {result.stdout}")

        except Exception as e:
            logger.error(f"Error building image: {e}")
        finally:
            logger.info("Finished building Apptainer sandbox.")
            close_logger(logger)  # functions that create loggers should close them
        