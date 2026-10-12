"""Build relocatable Windows and macOS offline ZIPs with no external packages."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import stat
import tarfile
import zipfile
from build_launchers import write_launchers

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'xiangqi_lab'
OUT=ROOT/'dist'
VERSION='1.2.0'

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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('Windows','macOS','all'), default='all')
    args=parser.parse_args()
    platforms=('Windows','macOS') if args.platform=='all' else (args.platform,)
    # Fail before publishing an incomplete package when local runtime inputs are missing.
    required=[APP/'engine/pikafish.nnue']
    if 'Windows' in platforms:
        required += [ROOT/'.runtime/python'/p for p in ('python.exe','python313.dll','Lib/encodings/__init__.py')]
        required += [APP/'engine/Pikafish-Windows-x86-64-universal.exe']
    if 'macOS' in platforms:
        required += [APP/'engine/Pikafish-MacOS-universal']
        for arch in ('aarch64','x86_64'):
            matches=list((ROOT/'.setup/mac-runtimes').glob(f'*-{arch}-apple-darwin-install_only_stripped.tar.gz'))
            if len(matches)!=1: raise ValueError(f'Expected exactly one prepared macOS runtime for {arch}')
    for p in required:
        if not p.is_file(): raise FileNotFoundError(f'Release input is missing: {p}')
    write_launchers()
    OUT.mkdir(exist_ok=True)
    for platform in platforms:
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
            if platform=='Windows':
                for p in ('python.exe','python313.dll','Lib/encodings/__init__.py'):
                    assert 'XiangqiLab/runtime/python/'+p in z.namelist()
                assert b'#==XIANGQI_POWERSHELL==' in z.read('XiangqiLab/Start-Windows.bat')
            if platform=='macOS':
                assert z.getinfo('XiangqiLab/Start-macOS.command').external_attr>>16 & 0o111
                for arch in ('macos-arm64','macos-x64'):
                    assert f'XiangqiLab/runtime/{arch}/python/bin/python3' in z.namelist()
        print(f'Verified {file.stat().st_size/1024**2:.1f} MiB',flush=True)
    if 'Windows' in platforms:
        (OUT/'Start-Xiangqi-Windows.bat').write_bytes((ROOT/'Start-Xiangqi-Windows.bat').read_bytes())
        (OUT/'WINDOWS-START.txt').write_text(
            'Xiangqi Lab - Windows quick start\n\n'
            f'Keep Start-Xiangqi-Windows.bat and XiangqiLab-Windows-v{VERSION}.zip together.\n'
            'Double-click Start-Xiangqi-Windows.bat. It checks files, unpacks the ZIP and opens the site.\n'
            'You can also extract the full ZIP yourself and run XiangqiLab/Start-Windows.bat.\n'
            'If opened inside a ZIP, the launcher searches nearby, Downloads and Desktop.\n'
            'If it cannot find the full Windows ZIP, select it in the file window.\n'
            'A GitHub source ZIP does not contain Python. Share the Windows release ZIP.\n'
            'No administrator access, Python installation or internet is needed for the offline package.\n'
            'Keep the console open while playing. Ctrl+C stops the site.\n'
            'Auto-extracted files and launch logs: %LOCALAPPDATA%/XiangqiLab/\n'
            'The console prints the app folder; add model folders to models/ there.\n'
            'macOS: use the macOS ZIP and Start-macOS.command after full extraction.\n',encoding='utf-8')
    hashes=[]
    files=sorted(OUT.glob(f'XiangqiLab-*-v{VERSION}.zip'))
    files += [p for p in (OUT/'Start-Xiangqi-Windows.bat',OUT/'WINDOWS-START.txt') if p.is_file()]
    for p in files:
        with p.open('rb') as stream: hashes.append(hashlib.file_digest(stream,'sha256').hexdigest()+'  '+p.name)
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n',encoding='utf-8')

if __name__=='__main__':main()
