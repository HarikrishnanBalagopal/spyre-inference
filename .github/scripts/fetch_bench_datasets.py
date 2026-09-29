#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Put the vLLM benchmark trace files on this host, then print their env vars.

The serve benchmarks replay recorded traces. Until now every host was expected
to have them pre-mounted under /models, which only the x86_64 benchmark hosts
do, so the s390x and ppc64le perf lanes could not run a serve config at all.
This fetches them from the artifact store into a cache dir instead, so any
arch can run.

Cache-first by design: a file whose SHA256 already matches DATASETS is left
alone and never re-downloaded, so a warm host does no network I/O. Only a
missing or corrupt file is fetched. The digests below are the integrity gate,
so a truncated transfer or a silently-replaced artifact fails here rather than
skewing a benchmark.

The cache dir is usually shared and writable by many jobs at once (on x86 it is
a ReadWriteMany PVC mounted into every runner pod), so a fetch takes an
exclusive lock per file and re-checks the cache after acquiring it. Concurrent
jobs therefore download a given trace once, and readers only ever see a
complete file because the download lands on a `.part` sibling and is renamed
into place after its digest is verified.

Eval the output to set the vars the configs reference:

    eval "$(python3 .github/scripts/fetch_bench_datasets.py)"

Subcommands:
  fetch (default)   Ensure each dataset is cached, print `export VAR=path`.
  env               Print `export VAR=path` only, fetching nothing.
  verify            Check the cache and exit non-zero if anything is missing.

Env:
  SPYRE_BENCH_DATA_DIR   cache dir (default ~/.cache/spyre/vllm-bench-data). CI
                         points this at the shared PVC the runners mount.
  ARTIFACTORY_BASE_URL   artifact store base, e.g. https://<host>
  ARTIFACTORY_BENCH_DATA_PATH  repo-relative prefix holding the .jsonl files
  ARTIFACTORY_TOKEN      bearer token (only needed when something must be fetched)
"""

import fcntl
import hashlib
import os
import subprocess
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

# The traces each serve config needs, keyed by the env var that names it.
# `sha256` is the integrity gate; see the module docstring. Only the two
# datasets the benchmark configs actually reference are listed: the store
# holds further truncations that no config selects, and pulling those too
# would cost every runner a multi-hundred-MB transfer for nothing. Add an
# entry here when a config starts using one.
DATASETS = {
    "SPYRE_AIOPS_DATASET": {
        "filename": "aiops_results_2025.11.03_e2ee1b0_correct_order.jsonl",
        "sha256": "468cf059f2ec3b14108efc09baffabf5c5a440172a25660673d5a8fa029d0637",
    },
    "SPYRE_CICS_DATASET": {
        "filename": "cics_results_2025.11.03_e2ee1b0_correct_order.jsonl",
        "sha256": "e0efa895b4a22b601748c7dda9e8a2db146773fb8735fc45ee2920c042b40fb7",
    },
}

DEFAULT_CACHE_DIR = "~/.cache/spyre/vllm-bench-data"


def cache_dir() -> Path:
    return Path(os.environ.get("SPYRE_BENCH_DATA_DIR") or DEFAULT_CACHE_DIR).expanduser()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # Traces run to hundreds of MB, so stream rather than read() them.
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_cached(path: Path, expected_sha: str) -> bool:
    """True when `path` is present AND its digest matches, so a fetch can be skipped."""
    if not path.is_file():
        return False
    actual = sha256_of(path)
    if actual == expected_sha:
        return True
    print(
        f"{path.name}: cached copy has sha256 {actual}, expected {expected_sha}; re-fetching",
        file=sys.stderr,
    )
    return False


def _require(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        sys.exit(
            f"{var} is not set, and a dataset must be fetched. Set it or pre-populate the cache."
        )
    return value


@contextmanager
def file_lock(path: Path):
    """Hold an exclusive lock for one dataset, so parallel jobs fetch it once.

    The lock file is a separate `.lock` sibling, never the dataset itself: locking
    the dataset would mean opening it for write and truncating a good cached copy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def download(filename: str, dest: Path, expected_sha: str) -> None:
    """Fetch one trace into `dest`, verifying the digest before it is published.

    Downloads to a `.part` sibling and renames only after the digest matches, so
    an interrupted run can never leave a short file that later looks cached.
    """
    base = _require("ARTIFACTORY_BASE_URL").rstrip("/")
    prefix = _require("ARTIFACTORY_BENCH_DATA_PATH").strip("/")
    token = _require("ARTIFACTORY_TOKEN")
    url = f"{base}/artifactory/{prefix}/{filename}"

    # PID-suffixed so two processes can never write the same temp file.
    partial = dest.with_suffix(f"{dest.suffix}.{os.getpid()}.part")
    print(f"Fetching {filename} ...", file=sys.stderr)
    subprocess.run(
        [
            "curl",
            "-fSL",
            "--no-progress-meter",
            "--retry",
            "3",
            "--retry-delay",
            "5",
            "-H",
            f"Authorization: Bearer {token}",
            "-o",
            str(partial),
            url,
        ],
        check=True,
    )

    actual = sha256_of(partial)
    if actual != expected_sha:
        partial.unlink(missing_ok=True)
        sys.exit(f"{filename}: sha256 {actual} does not match expected {expected_sha}")
    partial.replace(dest)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="fetch", choices=["fetch", "env", "verify"])
    args = parser.parse_args()

    destination = cache_dir()
    if args.command == "fetch":
        destination.mkdir(parents=True, exist_ok=True)

    missing = []
    for var, spec in DATASETS.items():
        path = destination / spec["filename"]
        # `env` only reports where the files belong, so it never hashes or fetches.
        if args.command != "env" and not is_cached(path, spec["sha256"]):
            if args.command == "fetch":
                with file_lock(path):
                    # Another job may have finished the fetch while we waited.
                    if not is_cached(path, spec["sha256"]):
                        download(spec["filename"], path, spec["sha256"])
            else:
                missing.append(path)
        print(f"export {var}={path}")

    if missing:
        for path in missing:
            print(f"missing from cache: {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
