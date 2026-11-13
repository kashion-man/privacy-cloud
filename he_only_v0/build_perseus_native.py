"""Isolated Perseus build for WHU216: Rust 1.99.1, CUDA 12.9, SM86."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile,time,tomllib,urllib.request,socket
r=Path(__file__).resolve().parent
downloads=r/'native-build-downloads'
prefix=r/'.rust-native'
downloads.mkdir(exist_ok=True)
socket.setdefaulttimeout(30)
def fetch(url,path,digest=None):
    if not path.exists() or (digest and hashlib.sha256(path.read_bytes()).hexdigest()!=digest):
        part=path.with_suffix(path.suffix+'.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=30) as incoming, part.open('wb') as out:
                    shutil.copyfileobj(incoming,out,1024*1024)
                if digest:
                    assert hashlib.sha256(part.read_bytes()).hexdigest()==digest,'Checksum mismatch'
                part.replace(path)
                break
            except Exception:
                if attempt==2: raise
                time.sleep(2)
    return path
manifest_url='https://rsproxy.cn/dist/channel-rust-1.99.1.toml'
manifest=tomllib.loads(urllib.request.urlopen(manifest_url,timeout=30).read().decode())
for component in ['rustc','rust-std','cargo']:
    item=manifest['pkg'][component]['target']['x86_64-unknown-linux-gnu']
    original=item['xz_url']
    assert original.startswith('https://static.rust-lang.org/dist/')
    url=original.replace('https://static.rust-lang.org','https://rsproxy.cn',1)
    archive=downloads/Path(original).name
    print(json.dumps({'stage':'download_toolchain','component':component,'url':url}),flush=True)
    fetch(url,archive,item['xz_hash'])
    with tarfile.open(archive) as package:
        members=package.getmembers()
        for entry in members:
            target=(downloads/entry.name).resolve()
            assert downloads.resolve() in target.parents
        top=members[0].name.split('/')[0]
        package.extractall(downloads,filter='data')
    subprocess.run(['sh',str(downloads/top/'install.sh'),'--prefix='+str(prefix),'--disable-ldconfig'],check=True)
print(subprocess.check_output([str(prefix/'bin/rustc'),'--version'],text=True),flush=True)
cargo_home=r/'.cargo-native'
cargo_home.mkdir(exist_ok=True)
(cargo_home/'config.toml').write_text('[source.crates-io]\nreplace-with = "rsproxy-sparse"\n[source.rsproxy-sparse]\nregistry = "sparse+https://rsproxy.cn/index/"\n[net]\nretry = 2\n[http]\ntimeout = 30\n')
env=os.environ.copy()
env.update(PATH=str(prefix/'bin')+':'+env.get('PATH',''),RUSTC=str(prefix/'bin/rustc'),CARGO_HOME=str(cargo_home),CARGO_TARGET_DIR=str(r/'perseus-target'),CUDA_HOME=env.get('CUDA_HOME','/usr/local/cuda'),PERSEUS_CORE_ARCH='86',PC_NVCC_JOBS='2',PYO3_PYTHON=str(r/'.venv/bin/python'))
cuda_lib=Path(env['CUDA_HOME'])/'lib64'
env['RUSTFLAGS']=' '.join('-L native='+str(lib)+' -C link-arg=-Wl,-rpath,'+str(lib)
                          for lib in (r/'.cuda/lib64',cuda_lib))
env['LD_LIBRARY_PATH']=str(r/'.cuda/lib64')+':'+str(cuda_lib)+':'+str(prefix/'lib')+':'+env.get('LD_LIBRARY_PATH','')
source=r/'perseus-core'
session_source=source/'crates/pc-py/src/session.rs'
text=session_source.read_text()
old='    let k_max = 8;'
new='    let k_max = knob(cfg.bts.k_max, 8, "bts.k_max")? as usize;\n    if k_max == 0 || k_max > 8 {\n        return Err(PyValueError::new_err("bts.k_max must be in 1..=8"));\n    }'
if old in text:
    assert text.count(old)==1
    session_source.write_text(text.replace(old,new))
else:
    assert new in text
fhe_source=source/'crates/pc-py/src/fhe.rs'
text=fhe_source.read_text()
anchor='    /// In place: pay a pending rescale (`core.realize_pending_rescale`).'
scale_method='''    /// Value-preserving public-scale alignment after a modulus drop.
    fn canonicalize_scale(&self, a: PyRef<'_, PyCt>) -> PyResult<PyCt> {
        self.same_session(&a)?;
        let m = a.meta();
        if m.deg != 1 || !m.nf.is_finite() || m.nf <= 0.0 {
            return Err(PyValueError::new_err("canonicalize_scale requires realized positive scale"));
        }
        let top = self.dev.ctx.top_of(m.level).map_err(gerr)?;
        let sf = self.dev.ctx.scale.sf(top as i32).map_err(|e| gerr(e.into()))?;
        let mut out = self.mult_scalar_ct(&a, Complex::new(sf / m.nf, 0.0))?;
        out.ct_mut()?.meta.nf = sf * sf;
        Ok(out)
    }

'''
if 'fn canonicalize_scale(' not in text:
    assert text.count(anchor)==1
    fhe_source.write_text(text.replace(anchor,scale_method+anchor))
lock_before=hashlib.sha256((source/'Cargo.lock').read_bytes()).hexdigest()
print(json.dumps({'stage':'cargo_build','arch':86,'lock_sha256':lock_before}),flush=True)
subprocess.run([str(prefix/'bin/cargo'),'build','--release','--locked','--jobs','2','-p','pc-py'],cwd=source,env=env,check=True)
assert hashlib.sha256((source/'Cargo.lock').read_bytes()).hexdigest()==lock_before
library=r/'perseus-target/release/lib_pc.so'
shutil.copy2(library,source/'python/perseus/_pc.abi3.so')
env['PYTHONPATH']=str(source/'python')
subprocess.run([str(r/'.venv/bin/python'),'-c','import perseus._pc as pc; print("PERSEUS_NATIVE_IMPORT_OK", pc.Config)'],env=env,check=True)
(r/'perseus_native_build_report.json').write_text(json.dumps({'build':'passed','arch':86,'cuda':'12.9','rust':'1.99.1','lock_sha256':lock_before},indent=2))
print('PERSEUS_NATIVE_BUILD_PASSED',flush=True)
