#!/usr/bin/env python3
"""
fetch_all_preds_paramiko.py

Downloads remote /home/<user>/OmniCode/GPT_alibaba*/all_preds.jsonl files via SSH/SFTP
and stores each in a local folder named exactly like the remote GPT_* directory.

Example:
  python fetch_all_preds_paramiko.py \
    --user dd732 \
    --host unicorn-login-01.coecis.cornell.edu \
    --remote-base /home/dd732/OmniCode \
    --pattern 'GPT_alibaba*' \
    --dest .

Security:
  - Script prompts for password via getpass (not stored).
  - Prefer SSH keys (leave password blank and it will try key-based auth).
"""

import argparse
import getpass
import os
import sys
import stat
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("This script requires paramiko. Install with: pip install paramiko", file=sys.stderr)
    sys.exit(1)


def list_remote_all_preds(ssh_client: paramiko.SSHClient, remote_base: str, pattern: str, verbose: bool = False):
    """
    Uses a safe shell loop on remote to find existing all_preds.jsonl paths.
    Returns a list of absolute remote paths (strings).
    """
    # The loop ensures we only print actual files; it avoids printing the literal glob if no match.
    remote_glob = os.path.join(remote_base, f"{pattern}", "all_preds.jsonl")
    cmd = ("bash -lc " +
           "'" +
           "for f in " + remote_glob + "; do [ -f \"$f\" ] && printf \"%s\\n\" \"$f\"; done" +
           "'")
    if verbose:
        print("REMOTE CMD:", cmd)
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    out = stdout.read().decode().splitlines()
    err = stderr.read().decode().strip()
    if err and verbose:
        print("Remote stderr:", err)
    return out


def download_file_sftp(sftp: paramiko.SFTPClient, remote_path: str, local_path: str, verbose: bool = False):
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)
    if verbose:
        print(f"Downloading {remote_path} -> {local_path}")
    sftp.get(remote_path, local_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", default="dd732", help="Remote username")
    p.add_argument("--host", default="unicorn-login-01.coecis.cornell.edu", help="Remote host")
    p.add_argument("--port", default=22, type=int, help="SSH port")
    p.add_argument("--remote-base", default="/home/dd732/OmniCode", help="Remote base path (OmniCode folder)")
    p.add_argument("--pattern", default="GPT_google*", help="Pattern for GPT directories (shell glob pattern)")
    p.add_argument("--dest", default=".", help="Local destination base directory")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    password = getpass.getpass(f"Password for {args.user}@{args.host} (leave blank to try SSH keys): ")

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        if args.verbose:
            print(f"Connecting to {args.host}:{args.port} as {args.user} ...")
        # Try password auth if provided, otherwise try keys (paramiko will try keys automatically if password is None or empty)
        connect_kwargs = dict(hostname=args.host, port=args.port, username=args.user, timeout=30)
        if password:
            connect_kwargs["password"] = password
        ssh.connect(**connect_kwargs)
    except Exception as e:
        print(f"SSH connection failed: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        remote_paths = list_remote_all_preds(ssh, args.remote_base, args.pattern, verbose=args.verbose)
        if not remote_paths:
            print("No all_preds.jsonl files found on remote matching the pattern.")
            ssh.close()
            return

        # Open SFTP
        sftp = ssh.open_sftp()

        downloaded = []
        for remote_path in remote_paths:
            # Extract GPT directory name: basename of parent directory of remote_path
            parent_dir = os.path.basename(os.path.dirname(remote_path))
            if not parent_dir:
                if args.verbose:
                    print(f"Skipping weird path (no parent dir): {remote_path}")
                continue
            local_dir = os.path.join(args.dest, parent_dir)
            local_path = os.path.join(local_dir, "all_preds.jsonl")
            try:
                download_file_sftp(sftp, remote_path, local_path, verbose=args.verbose)
                downloaded.append((remote_path, local_path))
            except Exception as e:
                print(f"Failed to download {remote_path}: {e}", file=sys.stderr)

        sftp.close()
        ssh.close()

        print(f"Done. Downloaded {len(downloaded)} file(s).")
        for r, l in downloaded:
            print(f"  {r}  ->  {l}")

    except Exception as e:
        print("Error during operation:", e, file=sys.stderr)
        ssh.close()
        sys.exit(3)


if __name__ == "__main__":
    main()
