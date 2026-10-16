"""Download the official ALOHA checkpoint, pin object generations and verify MD5."""

import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "gcs-objects.json"
    prefix = "checkpoints/pi0_aloha_sim/"
    if manifest.exists():
        listing = json.loads(manifest.read_text())
    else:
        with urlopen(
            "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix="
            + quote(prefix)
        ) as response:
            listing = json.load(response)
        if listing.get("nextPageToken"):
            raise RuntimeError(
                "Unexpected paginated checkpoint; do not silently omit objects"
            )
        manifest.write_text(json.dumps(listing, indent=2) + "\n")

    def download(item):
        name = item["name"]
        if name.endswith("/"):
            return None
        path = args.output / name.removeprefix(prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size != int(item["size"]):
            temporary = path.with_name(path.name + ".part")
            url = (
                "https://storage.googleapis.com/openpi-assets/"
                + quote(name)
                + "?generation="
                + item["generation"]
            )
            urlretrieve(url, temporary)
            temporary.replace(path)
        md5, sha = hashlib.md5(), hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024**2):
                md5.update(chunk)
                sha.update(chunk)
        if base64.b64encode(md5.digest()).decode() != item["md5Hash"]:
            raise RuntimeError(f"Checksum mismatch: {name}")
        print(name, flush=True)
        return {
            "path": str(path.relative_to(args.output)),
            "generation": item["generation"],
            "sha256": sha.hexdigest(),
            "bytes": int(item["size"]),
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [row for row in pool.map(download, listing["items"]) if row]
    (args.output / "checksums.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
