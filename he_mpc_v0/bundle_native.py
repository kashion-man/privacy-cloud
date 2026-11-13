"""Bundle only executable code/shared libraries for the WSL MPC party.

Never includes Player-Data, TLS keys, HE keys, shares, model weights or logs.
Run on WHU216 after a successful default SoftSpokenOT build.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parent
    source = root/'third_party/mp-spdz-0.4.4'
    config = (source/'CONFIG.mine').read_text()
    if 'USE_KOS = 0' not in config or 'USE_LIBOTE2 = 1' not in config or 'INSECURE' in config:
        raise ValueError('Only the secure default OT build can be bundled')
    artifacts = [(source/'semi-party.x', 'semi-party.x'), (source/'libSPDZ.so', 'libSPDZ.so')]
    text = '\n'.join(subprocess.check_output(['ldd',str(p)],text=True) for p,_ in artifacts)
    if 'not found' in text:
        raise ValueError('Unresolved native dependency: '+text)
    libs = {}
    excluded = ('libc.so.', 'libm.so.', 'libpthread.so.', 'libdl.so.', 'librt.so.', 'ld-linux')
    for name, path in re.findall(r'^\s*(\S+) => (/\S+)',text,re.M):
        if not name.startswith(excluded) and name != 'libSPDZ.so':
            libs[name] = Path(path).resolve()
    artifacts += [(p,'native-libs/'+name) for name,p in sorted(libs.items())]
    manifest = {name: hashlib.sha256(p.read_bytes()).hexdigest() for p,name in artifacts}
    manifest['build'] = dict(mp_spdz='v0.4.4', architecture='haswell',
                             ot='SoftSpokenOT', insecure=False,
                             libote='fe6b2ff82a976a5332b3a2efe325831588b1324a')
    path = root/'native-runtime-manifest.json'
    path.write_text(json.dumps(manifest,indent=2))
    with tarfile.open(str(root/'native-runtime.tgz'),'w:gz') as archive:
        for p,name in artifacts:
            archive.add(str(p),arcname=name,recursive=False)
        archive.add(str(path),arcname='native-runtime-manifest.json',recursive=False)
    print(json.dumps(dict(artifact=str(root/'native-runtime.tgz'),
                          bytes=(root/'native-runtime.tgz').stat().st_size,
                          files=len(artifacts))))


if __name__=='__main__':main()
