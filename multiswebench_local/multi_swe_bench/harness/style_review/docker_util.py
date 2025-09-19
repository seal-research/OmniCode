import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union, Dict, List

# Try to import docker but tolerate missing package/daemon.
try:
    import docker  # type: ignore

    _docker_imported = True
except Exception:
    docker = None  # type: ignore
    _docker_imported = False

# Try to instantiate docker client and ping daemon (keep original `docker_client` name)
docker_client = None
if _docker_imported:
    try:
        _tmp_client = docker.from_env()  # type: ignore
        _tmp_client.ping()
        docker_client = _tmp_client
    except Exception:
        docker_client = None

# Locate apptainer/singularity binary if present
def _find_apptainer_bin() -> Optional[str]:
    for name in ("apptainer", "singularity"):
        p = shutil.which(name)
        if p:
            return p
    return None


APPTAINER_BIN = _find_apptainer_bin()


def _sif_name_from_image(image_full_name: str) -> str:
    """
    Deterministic mapping from docker-style image name to a safe SIF filename.
    This mapping is stable and reversible-ish (keeps behaviour consistent with prior changes).
    Example: 'ghcr.io/org/name:tag' -> 'ghcr.io_org_name_tag.sif'
    """
    safe = image_full_name
    return f"{safe}.sif"


def exists(image_name: str) -> bool:
    """
    Return True if the image exists locally for the active runtime.
    - Docker: checks local Docker images (same as original)
    - Apptainer: checks for a corresponding .sif file derived from image_name
    """
    # Docker preferred
    if docker_client:
        try:
            docker_client.images.get(image_name)  # type: ignore
            return True
        except Exception:
            return False

    # Fall back to checking for SIF file when using Apptainer
    if APPTAINER_BIN:
        sif = Path(_sif_name_from_image(image_name))
        return sif.exists()

    raise RuntimeError("No container runtime available. Install Docker or Apptainer/Singularity.")


def _stream_subprocess(proc: subprocess.Popen, logger: logging.Logger) -> str:
    """Stream subprocess stdout/stderr to logger and collect output."""
    output = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.rstrip("\n")
        logger.info(line)
        output += line + "\n"
    proc.wait()
    return output


def build(
    workdir: Path, dockerfile_name: str, image_full_name: str, logger: logging.Logger
):
    """
    Build an image/tag.

    Behavior:
      - If Docker (daemon + SDK) is available: build with Docker (exactly like before).
      - Otherwise if Apptainer is available:
          1) If a Singularity definition file ('Singularity' or 'Singularity.def') exists
             in workdir, build SIF from that.
          2) Else try to pull a prebuilt Docker image from a registry and build a SIF from it
             using `apptainer build <sif> docker://<image_full_name>`. This keeps the same
             image naming format for callers (they still pass the same docker-style name).
    """
    workdir = Path(workdir).resolve()
    logger.info(
        f"Start building image `{image_full_name}`, working directory is `{workdir}`"
    )

    # Primary: Docker (preserve original behavior)
    if docker_client:
        try:
            build_logs = docker_client.api.build(  # type: ignore
                path=str(workdir),
                dockerfile=dockerfile_name,
                tag=image_full_name,
                rm=True,
                forcerm=True,
                decode=True,
                encoding="utf-8",
            )

            for log in build_logs:
                if "stream" in log:
                    logger.info(log["stream"].strip())
                elif "error" in log:
                    error_message = log["error"].strip()
                    logger.error(f"Docker build error: {error_message}")
                    raise RuntimeError(f"Docker build failed: {error_message}")
                elif "status" in log:
                    logger.info(log["status"].strip())
                elif "aux" in log:
                    logger.info(log["aux"].get("ID", "").strip())

            logger.info(f"image({workdir}) build success: {image_full_name}")
            return
        except Exception as e:
            # If Docker exist but build fails, surface error (consistent with original behavior).
            logger.error(f"Docker build attempt failed: {e}")
            raise

    # Fallback: Apptainer
    if APPTAINER_BIN:
        # 1) Prefer Singularity definition file if present
        candidates = [workdir / "Singularity", workdir / "Singularity.def"]
        def_path: Optional[Path] = None
        for c in candidates:
            if c.exists():
                def_path = c
                break

        sif_name = _sif_name_from_image(image_full_name)
        sif_path = workdir / sif_name

        if def_path:
            # Build SIF from Singularity def
            cmd = [APPTAINER_BIN, "build", "--force", str(sif_path), str(def_path)]
            logger.info("Running Apptainer build (from Singularity def): " + " ".join(cmd))
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                _stream_subprocess(proc, logger)
                if proc.returncode != 0:
                    raise RuntimeError(f"Apptainer build failed (exit {proc.returncode})")
                logger.info(f"Apptainer build success: {sif_path}")
                return
            except Exception as e:
                logger.error(f"Apptainer build error: {e}")
                raise

        # 2) If no def found, attempt to pull the Docker image from registry and convert it to SIF.
        #    This preserves the exact image id callers use (docker-style name).
        cmd = [APPTAINER_BIN, "build", "--force", str(sif_path), f"docker://{image_full_name}"]
        logger.info("Running Apptainer build (from docker://): " + " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            _stream_subprocess(proc, logger)
            if proc.returncode != 0:
                raise RuntimeError(f"Apptainer build from docker:// failed (exit {proc.returncode})")
            logger.info(f"Apptainer build success: {sif_path}")
            return
        except Exception as e:
            logger.error(f"Apptainer build (docker://) error: {e}")
            raise RuntimeError(
                f"Apptainer build failed. Provide a Singularity definition in `{workdir}` "
                "or ensure network access to pull the docker image, or install Docker."
            )

    # If neither runtime is available
    raise RuntimeError("Unable to build image: no Docker daemon and no Apptainer available.")


def run(
    image_full_name: str,
    run_command: str,
    output_path: Optional[Path] = None,
    global_env: Optional[list[str]] = None,
    volumes: Optional[Union[dict[str, str], list[str]]] = None,
) -> str:
    """
    Run a command inside the specified image.

    - Docker: same behavior as original.
    - Apptainer: expects a .sif file whose name is derived from image_full_name via _sif_name_from_image.
      If the .sif does not exist, the function will attempt to build it on-the-fly using:
        apptainer build <sif> docker://<image_full_name>
      (this preserves the external naming format — callers still pass the same docker-style name).
    """
    # Docker path (unchanged)
    if docker_client:
        container = docker_client.containers.run(
            image=image_full_name,
            command=run_command,
            remove=False,
            detach=True,
            stdout=True,
            stderr=True,
            environment=global_env,
            volumes=volumes,
        )

        output = ""
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                for line in container.logs(stream=True, follow=True):
                    line_decoded = line.decode("utf-8")
                    f.write(line_decoded)
                    output += line_decoded
        else:
            container.wait()
            output = container.logs().decode("utf-8")

        container.remove()
        return output

    # Apptainer path
    if APPTAINER_BIN:
        sif_path = Path(_sif_name_from_image(image_full_name))
        # If SIF missing, attempt to build from docker://<image_full_name> (keeps image naming same)
        if not sif_path.exists():
            # Attempt to build in current working directory
            logger = logging.getLogger(__name__)
            logger.info(f"SIF `{sif_path}` not found — attempting to build from docker://{image_full_name}")
            build(Path("."), "", image_full_name, logger)  # build will raise if it fails

            if not sif_path.exists():
                raise RuntimeError(
                    f"After attempted apptainer build, SIF `{sif_path}` still not found."
                )

        # Prepare binds
        binds: List[str] = []
        if volumes:
            if isinstance(volumes, dict):
                for host, cont in volumes.items():
                    binds.append(f"{host}:{cont}")
            else:
                binds.extend(volumes)

        # Prepare environment prefix (we prefix the inner command with env VAR=VAL to avoid depending on CLI flags)
        env_prefix = ""
        if global_env:
            env_parts = []
            for kv in global_env:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    env_parts.append(f"{k}='{v}'")
            if env_parts:
                env_prefix = " ".join(env_parts) + " "

        # Compose apptainer exec command
        cmd = [APPTAINER_BIN, "exec"]
        for b in binds:
            cmd += ["--bind", b]
        cmd += [str(sif_path), "/bin/bash", "-lc", env_prefix + run_command]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        output = ""
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                assert proc.stdout is not None
                for line in proc.stdout:
                    f.write(line)
                    output += line
        else:
            assert proc.stdout is not None
            for line in proc.stdout:
                output += line

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Apptainer exec failed (exit {proc.returncode})")

        return output

    raise RuntimeError("No container runtime available: please install Docker or Apptainer/Singularity.")
