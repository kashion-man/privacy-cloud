"""Install only hash-verified public executable/library artifacts on WSL."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',default=str(Path(__file__).resolve().parent/'third_party/native-runtime-compact.tgz'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent/'third_party/mp-spdz-0.4.4'
    root=root.resolve()
    with tarfile.open(args.bundle) as archive:
        manifest=json.loads(archive.extractfile('native-runtime-manifest.json').read())
        if manifest['build'].get('insecure') is not False or manifest['build'].get('ot')!='SoftSpokenOT':
            raise ValueError('Expected secure default OT runtime')
        entries={}
        for member in archive.getmembers():
            if (not member.isfile() or root not in (root/member.name).resolve().parents or
                member.name not in set(manifest)|{'native-runtime-manifest.json'}):
                raise ValueError('Unexpected native artifact')
            data=archive.extractfile(member).read()
            if member.name!='native-runtime-manifest.json' and hashlib.sha256(data).hexdigest()!=manifest[member.name]:
                raise ValueError('Native artifact hash mismatch')
            entries[member.name]=data
        if set(entries)!=(set(manifest)-{'build'})|{'native-runtime-manifest.json'}:
            raise ValueError('Incomplete native runtime')
        backup=root/'semi-party.official-adx.x'
        if (root/'semi-party.x').is_file() and not backup.exists():
            shutil.copy2(str(root/'semi-party.x'),str(backup))
        for name,data in entries.items():
            destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_name(destination.name+'.hempc-install')
            temporary.write_bytes(data);temporary.chmod(0o755 if name!='native-runtime-manifest.json' else 0o644)
            temporary.replace(destination)
    print(json.dumps(dict(installed=True,build=manifest['build'],files=len(entries))))


if __name__=='__main__':main()
