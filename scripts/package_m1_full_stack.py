"""Build/verify the compact M1 replay kit; model weights stay outside Git."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from package_m1_ane_reference import relative_path, sha256


def manifest_for(root):
    files = {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('manifest.json', 'README.md', 'verification.json')}
    if any(name.endswith('.bin') or name.endswith('.hwx') or name.endswith('.pack') or '/weights/' in name for name in files):
        raise ValueError('Bulk weights/inputs/full HWX are not portable reference material')
    return {'schema': 2, 'architecture': 'H13G', 'cpu_subtype': 4, 'isa_version': 7,
            'files': files, 'stored_bytes': sum(info['bytes'] for info in files.values()),
            'learned_weight_banks_included': False, 'captured_tensor_binaries_included': False,
            'offline_hwx_evaluated': False, 'linux_replay_verified': False}


def verify_or_extract(root, output=None):
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest['schema'], manifest['architecture'], manifest['cpu_subtype'], manifest['isa_version']) != (2, 'H13G', 4, 7):
        raise ValueError('Expected compact H13G reference kit')
    if output is not None and output.exists():
        raise ValueError('Use a fresh extraction directory')
    if manifest_for(root) != manifest:
        raise ValueError('Changed or unexpected compact reference file')
    if output is not None:
        for name in manifest['files']:
            relative = relative_path(name)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        shutil.copyfile(root / 'manifest.json', output / 'manifest.json')
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    build = sub.add_parser('pack')
    for name in ('capture', 'target', 'draft', 'out'):
        build.add_argument('--' + name, type=Path, required=True)
    for name in ('verify', 'extract'):
        parser = sub.add_parser(name)
        parser.add_argument('bundle', type=Path)
        if name == 'extract':
            parser.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'pack':
        from derive_m1_lut6_templates import derive
        derive(args.capture, args.target, args.draft, args.out)
        manifest = manifest_for(args.out)
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        manifest = verify_or_extract(args.bundle, args.out if args.command == 'extract' else None)
    print(json.dumps({key: manifest[key] for key in ('stored_bytes', 'learned_weight_banks_included',
          'captured_tensor_binaries_included', 'offline_hwx_evaluated', 'linux_replay_verified')}, indent=2))


if __name__ == '__main__':
    main()
