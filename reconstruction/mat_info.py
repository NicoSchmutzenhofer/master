"""
Diagnostic script — run this FIRST to identify your .mat file format and structure.
No reconstruction, just prints everything about the files.
"""

import struct
from pathlib import Path

SINO_PATH = Path(r"/data/Data2/4_BIN_PCCT/4 bin Phantom raw data/Descriptor/sample_sinogram_4-bin_Phantom-Scan..CT.Thx.-_Abdomen_Staging.Körper.601.RAW.20260326.074425.20260401.175811.0815bcd5-7de9-4029-a9e4-0bce7ab6071b.raw.mat")
DESC_PATH = Path(r"/data/Data2/4_BIN_PCCT/4 bin Phantom raw data/Descriptor/descriptor_4-bin_Phantom-Scan..CT.Thx.-_Abdomen_Staging.Körper.601.RAW.20260326.074425.20260401.175811.0815bcd5-7de9-4029-a9e4-0bce7ab6071b.raw.mat")


def detect_mat_format(path):
    """Read the first 128 bytes and identify the MATLAB format."""
    with open(str(path), 'rb') as f:
        header = f.read(128)

    # HDF5 signature starts at byte 0: \x89HDF\r\n\x1a\n
    if header[:8] == b'\x89HDF\r\n\x1a\n':
        return 'HDF5 (MATLAB v7.3)'

    # MATLAB v5 header: first 116 bytes are a text description, bytes 124-127 are version
    try:
        text = header[:116].decode('latin-1').strip('\x00').strip()
        version_bytes = header[124:126]
        version = struct.unpack('<H', version_bytes)[0]
        if 'MATLAB' in text or version in (0x0100, 0x0200):
            return f'MATLAB v5/v7  (header: "{text[:60]}")'
    except Exception:
        pass

    return f'Unknown — first 8 bytes: {header[:8].hex()}'


def inspect_scipy(path):
    """Load with scipy.io and print all variables."""
    import scipy.io as sio
    import numpy as np

    data = sio.loadmat(str(path), struct_as_record=False, squeeze_me=False)

    print(f"\n  Variables in file:")
    for key, val in data.items():
        if key.startswith('_'):   # skip __header__, __version__, etc.
            continue
        if hasattr(val, 'shape'):
            print(f"    '{key}'  shape={val.shape}  dtype={val.dtype}")
        elif hasattr(val, '__class__'):
            print(f"    '{key}'  type={type(val).__name__}")

        # If it's a structured array (MATLAB struct), recurse one level
        if hasattr(val, 'dtype') and val.dtype.names:
            for field in val.dtype.names:
                try:
                    sub = val[field]
                    if hasattr(sub, 'shape'):
                        print(f"      .{field}  shape={sub.shape}  dtype={sub.dtype}")
                    # Two levels deep
                    if hasattr(sub, 'dtype') and sub.dtype.names:
                        for f2 in sub.dtype.names:
                            try:
                                sub2 = sub[f2]
                                if hasattr(sub2, 'shape'):
                                    print(f"        .{f2}  shape={sub2.shape}  dtype={sub2.dtype}")
                            except Exception:
                                pass
                except Exception:
                    pass

    return data


def inspect_hdf5(path):
    """Load with h5py and print all datasets."""
    import h5py

    def _recurse(name, obj, depth=0):
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}[Dataset] {name}  shape={obj.shape}  dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}[Group]   {name}/")
            if depth < 5:
                for key in obj.keys():
                    _recurse(key, obj[key], depth + 1)

    print(f"\n  HDF5 structure:")
    with h5py.File(str(path), 'r') as f:
        for key in f.keys():
            _recurse(key, f[key])


for label, path in [("SINOGRAM", SINO_PATH), ("DESCRIPTOR", DESC_PATH)]:
    print("=" * 60)
    print(f"{label}: {path.name}")
    print(f"  File size: {path.stat().st_size / 1e6:.1f} MB")

    fmt = detect_mat_format(path)
    print(f"  Format: {fmt}")

    if 'HDF5' in fmt:
        try:
            inspect_hdf5(path)
        except Exception as e:
            print(f"  h5py failed: {e}")
    else:
        try:
            inspect_scipy(path)
        except Exception as e:
            print(f"  scipy.io failed: {e}")
            # Still try h5py as last resort
            try:
                inspect_hdf5(path)
            except Exception as e2:
                print(f"  h5py also failed: {e2}")
                print("  >>> File may be corrupted or a non-standard format.")

print("\n=== Diagnosis complete ===")
print("Paste the output above so the loading code can be adapted.")