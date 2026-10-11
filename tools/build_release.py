"""Build relocatable Windows and macOS offline ZIPs with no external packages."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import stat
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'xiangqi_lab'
OUT=ROOT/'dist'
VERSION='1.1.1'

def add_bytes(z,name,data,mode=0o644,kind=stat.S_IFREG):
    info=zipfile.ZipInfo('XiangqiLab/'+name)
    info.create_system=3
    info.external_attr=(kind|mode)<<16
    info.compress_type=zipfile.ZIP_DEFLATED
    z.writestr(info,data)

def add_app(z,platform):
    for file in sorted(APP.rglob('*')):
        if not file.is_file() or '__pycache__' in file.parts or 'runtime' in file.parts:continue
        relative=file.relative_to(APP).as_posix()
        if platform=='Windows' and (file.name=='Pikafish-MacOS-universal' or file.suffix=='.command'):continue
        if platform=='macOS' and (file.suffix=='.exe' or file.suffix=='.bat'):continue
        mode=0o755 if file.suffix=='.command' or file.name=='Pikafish-MacOS-universal' else 0o644
        data=file.read_bytes()
        if file.suffix=='.bat':data=data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
        add_bytes(z,relative,data,mode)
    add_bytes(z,'release.json',json.dumps({'version':VERSION,'platform':platform,'python':'3.13.16','engine':'Pikafish 2026-09-06','offline':True},indent=2).encode())

def mac_runtime(z,arch,folder):
    archive=next((ROOT/'.setup/mac-runtimes').glob(f'*-{arch}-apple-darwin-install_only_stripped.tar.gz'))
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe tar path')
            name=f'runtime/{folder}/{member.name}'
            if member.isfile() or member.islnk():
                add_bytes(z,name,tar.extractfile(member).read(),member.mode)
            elif member.issym():
                add_bytes(z,name,member.linkname.encode(),member.mode,stat.S_IFLNK)

def main():
    OUT.mkdir(exist_ok=True)
    for platform in ('Windows','macOS'):
        file=OUT/f'XiangqiLab-{platform}-v{VERSION}.zip'
        print('Building',file.name,flush=True)
        with zipfile.ZipFile(file,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            add_app(z,platform)
            if platform=='Windows':
                runtime=ROOT/'.runtime/python'
                for p in sorted(runtime.rglob('*')):
                    if p.is_file() and '__pycache__' not in p.parts:
                        add_bytes(z,'runtime/python/'+p.relative_to(runtime).as_posix(),p.read_bytes())
            else:
                mac_runtime(z,'aarch64','macos-arm64')
                mac_runtime(z,'x86_64','macos-x64')
        with zipfile.ZipFile(file) as z:
            bad=z.testzip()
            if bad:raise ValueError('Bad archive member: '+bad)
            assert 'XiangqiLab/engine/pikafish.nnue' in z.namelist()
            assert any('source.tar.gz' in n for n in z.namelist())
            if platform=='macOS':
                assert z.getinfo('XiangqiLab/Start-macOS.command').external_attr>>16 & 0o111
                for arch in ('macos-arm64','macos-x64'):
                    assert f'XiangqiLab/runtime/{arch}/python/bin/python3' in z.namelist()
        print(f'Verified {file.stat().st_size/1024**2:.1f} MiB',flush=True)
    hashes=[]
    for p in sorted(OUT.glob(f'XiangqiLab-*-v{VERSION}.zip')):
        hashes.append(hashlib.file_digest(p.open('rb'),'sha256').hexdigest()+'  '+p.name)
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n',encoding='utf-8')

if __name__=='__main__':main()
