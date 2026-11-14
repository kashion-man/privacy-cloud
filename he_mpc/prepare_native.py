"""Install the public MP-SPDZ source/compiler tree from a pinned release."""
import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


RELEASE = "mp-spdz-0.4.4"
URL = f"https://github.com/data61/MP-SPDZ/releases/download/v0.4.4/{RELEASE}.tar.xz"
SHA256 = "73d6473526101b4433999fc4f7521fdfdc53ddf018e5f64480518d6293f62760"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the same release archive")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "third_party"
    root.mkdir(exist_ok=True)
    destination = root / RELEASE
    if destination.exists():
        parser.error(f"Runtime already exists at {destination}")
    with tempfile.TemporaryDirectory(dir=root) as directory:
        staging = Path(directory)
        archive = args.archive
        if archive is None:
            archive = staging / "release.tar.xz"
            with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with archive.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != SHA256:
                raise ValueError("MP-SPDZ release checksum mismatch")
        with tarfile.open(archive) as source:
            members = [member for member in source.getmembers()
                       if Path(member.name).parts[:1] == (RELEASE,)]
            source.extractall(staging, members=members, filter="data")
        compiled = staging / RELEASE
        if not (compiled / "compile.py").is_file():
            raise ValueError("MP-SPDZ compiler missing from release")
        compiled.rename(destination)
    print(destination)


if __name__ == "__main__":
    main()
