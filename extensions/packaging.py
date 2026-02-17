"""
Extension Packaging — .vortexext file format utilities.

A .vortexext file is a standard ZIP archive containing:
    manifest.json     (at the root level)
    <python package>/ (the extension code)

The extension ID from manifest.json is used as the subdirectory name
when installed into extensions/installed/.
"""

import json
import os
import shutil
import tempfile
import zipfile
from typing import Optional

from extensions.manifest import ExtensionManifest, load_manifest, validate_manifest_dict


def pack_extension(source_dir: str, output_path: str) -> str:
    """
    Pack an extension directory into a .vortexext archive.

    The source_dir must contain a manifest.json at its root.
    All files in source_dir are included in the archive (except __pycache__).

    Args:
        source_dir: Path to the extension directory
        output_path: Path for the output .vortexext file

    Returns:
        Absolute path to the created .vortexext file

    Raises:
        FileNotFoundError: If manifest.json is missing
        ValueError: If manifest is invalid
    """
    # Validate manifest first
    manifest_path = os.path.join(source_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest.json in {source_dir}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    errors = validate_manifest_dict(data)
    if errors:
        raise ValueError(f"Invalid manifest: {'; '.join(errors)}")

    # Create the zip
    output_path = os.path.abspath(output_path)
    if not output_path.endswith('.vortexext'):
        output_path += '.vortexext'

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']

            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zf.write(file_path, arcname)

    return output_path


def unpack_extension(vortexext_path: str, target_dir: str) -> str:
    """
    Unpack a .vortexext archive into the target directory.

    The archive is extracted into target_dir/<extension_id>/.

    Args:
        vortexext_path: Path to the .vortexext file
        target_dir: Parent directory to extract into

    Returns:
        Absolute path to the extracted extension directory

    Raises:
        FileNotFoundError: If the .vortexext file doesn't exist
        ValueError: If the archive is invalid or manifest is bad
        zipfile.BadZipFile: If the file is not a valid ZIP
    """
    if not os.path.exists(vortexext_path):
        raise FileNotFoundError(f"File not found: {vortexext_path}")

    # Extract to temp dir first for validation
    tmp_dir = tempfile.mkdtemp(prefix='vortex_ext_')

    try:
        with zipfile.ZipFile(vortexext_path, 'r') as zf:
            # Security: check for path traversal
            for info in zf.infolist():
                if info.filename.startswith('/') or '..' in info.filename:
                    raise ValueError(f"Unsafe path in archive: {info.filename}")
            zf.extractall(tmp_dir)

        # Validate manifest
        manifest_path = os.path.join(tmp_dir, 'manifest.json')
        if not os.path.exists(manifest_path):
            raise ValueError("Archive does not contain manifest.json at root level")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        errors = validate_manifest_dict(data)
        if errors:
            raise ValueError(f"Invalid manifest in archive: {'; '.join(errors)}")

        ext_id = data['id']
        final_dir = os.path.join(target_dir, ext_id)

        # Remove existing installation
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)

        # Move from temp to final location
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(tmp_dir, final_dir)

        return os.path.abspath(final_dir)

    except Exception:
        # Cleanup temp dir on failure
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        raise


def inspect_extension(vortexext_path: str) -> dict:
    """
    Read the manifest from a .vortexext file without extracting.

    Args:
        vortexext_path: Path to the .vortexext file

    Returns:
        Parsed manifest dict

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If manifest is missing or invalid
    """
    if not os.path.exists(vortexext_path):
        raise FileNotFoundError(f"File not found: {vortexext_path}")

    with zipfile.ZipFile(vortexext_path, 'r') as zf:
        try:
            with zf.open('manifest.json') as mf:
                data = json.loads(mf.read().decode('utf-8'))
        except KeyError:
            raise ValueError("Archive does not contain manifest.json")

    errors = validate_manifest_dict(data)
    if errors:
        raise ValueError(f"Invalid manifest: {'; '.join(errors)}")

    return data


def list_archive_contents(vortexext_path: str) -> list:
    """List all files in a .vortexext archive."""
    with zipfile.ZipFile(vortexext_path, 'r') as zf:
        return [info.filename for info in zf.infolist()]
