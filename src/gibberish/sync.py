"""
File synchronization logic for Gibberish.

This module handles the core synchronization logic including diff computation,
change detection, and coordination of file transfers between machines.
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum
import difflib
import hashlib
import os
import json
import fnmatch
from datetime import datetime
import struct
import zlib


class ChangeType(Enum):
    """Types of file changes"""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"


class FileChange:
    """Represents a file change"""

    def __init__(self, path: Path, change_type: ChangeType, size: int = 0,
                 diff_size: int = 0, is_binary: bool = False):
        self.path = path
        self.change_type = change_type
        self.size = size  # Original file size
        self.diff_size = diff_size  # Diff/patch size
        self.is_binary = is_binary

    def get_efficiency(self) -> Dict[str, Any]:
        """Calculate efficiency metrics for this change"""
        if self.change_type == ChangeType.DELETE:
            return {
                'original_size': self.size,
                'transfer_size': 0,
                'reduction': 100.0 if self.size > 0 else 0.0,
                'reduction_bytes': self.size
            }
        elif self.change_type == ChangeType.ADD:
            return {
                'original_size': self.size,
                'transfer_size': self.size,
                'reduction': 0.0,
                'reduction_bytes': 0
            }
        elif self.change_type == ChangeType.MODIFY:
            reduction_bytes = self.size - self.diff_size
            reduction_pct = (reduction_bytes / self.size * 100.0) if self.size > 0 else 0.0
            return {
                'original_size': self.size,
                'transfer_size': self.diff_size,
                'reduction': reduction_pct,
                'reduction_bytes': reduction_bytes
            }
        return {}


class SyncManager:
    """Manages file synchronization operations"""

    def __init__(self, root_dir: Path, ignore_patterns: List[str] = None,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB default
                 load_gitignore: bool = True):
        """
        Initialize the SyncManager

        Args:
            root_dir: Root directory to synchronize
            ignore_patterns: List of patterns to ignore (gitignore-style)
            max_file_size: Maximum file size to sync (bytes)
            load_gitignore: Whether to load .gitignore patterns
        """
        self.root_dir = Path(root_dir)
        self.ignore_patterns = ignore_patterns or ['.gibberish', '.git', '__pycache__', '.DS_Store']
        self.max_file_size = max_file_size
        self.cache_dir = self.root_dir / '.gibberish' / 'cache' / 'diffs'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load .gitignore patterns if requested
        if load_gitignore:
            self._load_gitignore_patterns()

    def _load_gitignore_patterns(self) -> None:
        """
        Load patterns from .gitignore file if it exists
        """
        gitignore_file = self.root_dir / '.gitignore'

        if gitignore_file.exists():
            try:
                with open(gitignore_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith('#'):
                            # Remove leading slash if present
                            if line.startswith('/'):
                                line = line[1:]
                            # Add to ignore patterns if not already present
                            if line not in self.ignore_patterns:
                                self.ignore_patterns.append(line)
            except Exception as e:
                print(f"Warning: Could not load .gitignore: {e}")

    def _matches_pattern(self, path_str: str, pattern: str) -> bool:
        """
        Check if a path matches a gitignore-style pattern

        Args:
            path_str: Path as string
            pattern: Pattern to match

        Returns:
            True if path matches pattern
        """
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            pattern = pattern.rstrip('/')
            # Match directory names
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str, f"*/{pattern}"):
                return True
            # Match anything under the directory
            if fnmatch.fnmatch(path_str, f"{pattern}/*") or fnmatch.fnmatch(path_str, f"*/{pattern}/*"):
                return True

        # Handle ** patterns (match any number of directories)
        if '**' in pattern:
            # Replace ** with glob pattern
            pattern = pattern.replace('**', '*')

        # Try direct match
        if fnmatch.fnmatch(path_str, pattern):
            return True

        # Try matching with wildcard prefix (for patterns that should match anywhere)
        if not pattern.startswith('*') and '/' not in pattern:
            if fnmatch.fnmatch(path_str, f"*/{pattern}"):
                return True
            # Also try matching just the filename
            if fnmatch.fnmatch(Path(path_str).name, pattern):
                return True

        return False

    def _should_ignore(self, path: Path) -> bool:
        """
        Check if a path should be ignored based on ignore patterns

        Args:
            path: Path to check (relative to root)

        Returns:
            True if path should be ignored
        """
        # Convert to forward-slash path for consistency
        path_str = path.as_posix()

        for pattern in self.ignore_patterns:
            if self._matches_pattern(path_str, pattern):
                return True

        return False

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of a file

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except (IOError, OSError) as e:
            raise ValueError(f"Failed to hash file {file_path}: {e}")

    def _is_binary_file(self, file_path: Path) -> bool:
        """
        Determine if a file is binary

        Args:
            file_path: Path to file

        Returns:
            True if file appears to be binary
        """
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                # Check for null bytes which typically indicate binary content
                if b'\x00' in chunk:
                    return True
                # Try to decode as UTF-8
                try:
                    chunk.decode('utf-8')
                    return False
                except UnicodeDecodeError:
                    return True
        except (IOError, OSError):
            return True

    def _write_file_atomic(self, file_path: Path, content, is_binary: bool = False) -> bool:
        """
        Write file atomically using temp file + rename pattern

        This ensures that the file is either fully written or not written at all,
        preventing partial writes due to crashes or errors.

        Args:
            file_path: Target file path
            content: Content to write (str for text, bytes for binary)
            is_binary: Whether content is binary

        Returns:
            True if write succeeded, False otherwise
        """
        import tempfile
        import stat

        # Preserve original file permissions and timestamps if file exists
        original_stat = None
        if file_path.exists():
            try:
                original_stat = file_path.stat()
            except Exception:
                pass

        # Create temporary file in the same directory to ensure atomic rename
        temp_fd = None
        temp_path = None

        try:
            # Create temp file in same directory as target
            temp_fd, temp_path_str = tempfile.mkstemp(
                dir=file_path.parent,
                prefix=f".{file_path.name}.",
                suffix=".tmp"
            )
            temp_path = Path(temp_path_str)

            # Write content to temp file
            if is_binary:
                os.write(temp_fd, content)
            else:
                os.write(temp_fd, content.encode('utf-8'))

            # Close the file descriptor
            os.close(temp_fd)
            temp_fd = None

            # Preserve permissions if original file existed
            if original_stat:
                try:
                    os.chmod(temp_path, original_stat.st_mode)
                    # Try to preserve timestamps (may not work on all systems)
                    os.utime(temp_path, (original_stat.st_atime, original_stat.st_mtime))
                except Exception:
                    pass  # Not critical if we can't preserve these

            # Atomic rename (os.replace is atomic on both Unix and Windows)
            os.replace(temp_path, file_path)

            return True

        except Exception as e:
            print(f"Error in atomic write to {file_path}: {e}")

            # Cleanup temp file if it exists
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except Exception:
                    pass

            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

            return False

    def scan_directory(self, since_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Scan directory and compute file hashes

        Args:
            since_time: Optional Unix timestamp - only scan files modified after this time

        Returns:
            Dictionary with:
            - 'files': mapping file paths (as strings) to file info dicts
            - 'scan_time': timestamp of scan completion
        """
        file_info = {}
        scan_start = datetime.utcnow().timestamp()

        for root, dirs, files in os.walk(self.root_dir):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(Path(d))]

            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                rel_path = file_path.relative_to(self.root_dir)

                # Skip ignored files
                if self._should_ignore(rel_path):
                    continue

                try:
                    stat_info = file_path.stat()
                    file_size = stat_info.st_size
                    mtime = stat_info.st_mtime

                    # Skip if not modified since last scan
                    if since_time and mtime < since_time:
                        continue

                    # Skip files that are too large
                    if file_size > self.max_file_size:
                        print(f"Warning: Skipping {rel_path} (size {file_size} exceeds limit)")
                        continue

                    # Compute hash
                    file_hash = self._compute_file_hash(file_path)

                    # Store file info
                    file_info[rel_path.as_posix()] = {
                        'hash': file_hash,
                        'size': file_size,
                        'mtime': mtime,
                        'is_binary': self._is_binary_file(file_path)
                    }

                except (OSError, ValueError) as e:
                    print(f"Warning: Error scanning {rel_path}: {e}")
                    continue

        return {
            'files': file_info,
            'scan_time': scan_start
        }

    def compute_diff(self, baseline: Dict[str, Any],
                    since_time: Optional[float] = None) -> List[FileChange]:
        """
        Compute diff between current state and baseline

        Args:
            baseline: Baseline data (can be simple dict of hashes or full baseline from BaselineManager)
            since_time: Optional Unix timestamp for incremental scanning

        Returns:
            List of FileChange objects
        """
        # Scan current directory
        scan_result = self.scan_directory(since_time=since_time)
        current = scan_result['files']

        # Extract baseline file hashes (handle both formats)
        if 'files' in baseline:
            # Full baseline format from BaselineManager
            baseline_files_dict = baseline['files']
        else:
            # Simple hash dict
            baseline_files_dict = baseline

        changes = []

        current_files = set(current.keys())
        baseline_files = set(baseline_files_dict.keys())

        # Find added files
        added = current_files - baseline_files
        for file_path in sorted(added):
            file_info = current[file_path]
            change = FileChange(
                path=Path(file_path),
                change_type=ChangeType.ADD,
                size=file_info['size'],
                diff_size=file_info['size'],  # For ADD, diff size = file size
                is_binary=file_info['is_binary']
            )
            changes.append(change)

        # Find deleted files
        deleted = baseline_files - current_files
        for file_path in sorted(deleted):
            change = FileChange(
                path=Path(file_path),
                change_type=ChangeType.DELETE,
                size=0,
                diff_size=0
            )
            changes.append(change)

        # Find modified files
        common_files = current_files & baseline_files
        for file_path in sorted(common_files):
            current_hash = current[file_path]['hash']

            # Get baseline hash (handle both formats)
            if isinstance(baseline_files_dict[file_path], str):
                baseline_hash = baseline_files_dict[file_path]
            else:
                baseline_hash = baseline_files_dict[file_path].get('hash', baseline_files_dict[file_path])

            if current_hash != baseline_hash:
                file_info = current[file_path]
                change = FileChange(
                    path=Path(file_path),
                    change_type=ChangeType.MODIFY,
                    size=file_info['size'],
                    diff_size=0,  # Will be set when patch is generated
                    is_binary=file_info['is_binary']
                )
                changes.append(change)

        return changes

    def _generate_text_diff(self, old_content: str, new_content: str) -> str:
        """
        Generate unified diff for text content

        Args:
            old_content: Original content
            new_content: New content

        Returns:
            Unified diff as string
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            lineterm='',
            n=3  # Context lines
        )

        return ''.join(diff)

    def _generate_binary_diff(self, old_data: bytes, new_data: bytes) -> bytes:
        """
        Generate binary diff using simple block-based delta algorithm.

        Since python-librsync is not available, this implements a basic
        block-based diff that identifies matching and changed blocks.

        Format:
        - Header: b'GBINDIFF' (8 bytes)
        - Version: uint32 (4 bytes) - version 1
        - Block size: uint32 (4 bytes)
        - Old file size: uint64 (8 bytes)
        - New file size: uint64 (8 bytes)
        - Blocks: [operation (1 byte) + data]*
          - Operation 0: COPY from old file (offset: uint64, length: uint32)
          - Operation 1: INSERT new data (length: uint32, data: bytes)

        Args:
            old_data: Original binary data
            new_data: New binary data

        Returns:
            Binary diff data
        """
        BLOCK_SIZE = 4096  # 4KB blocks

        # Simple case: if files are identical
        if old_data == new_data:
            # Return minimal diff indicating no changes
            header = b'GBINDIFF'
            header += struct.pack('<I', 1)  # Version
            header += struct.pack('<I', BLOCK_SIZE)
            header += struct.pack('<Q', len(old_data))
            header += struct.pack('<Q', len(new_data))
            # Single COPY operation covering entire file
            header += struct.pack('<BQI', 0, 0, len(old_data))
            return zlib.compress(header)

        # Build hash table of old file blocks
        old_blocks = {}
        for i in range(0, len(old_data), BLOCK_SIZE):
            block = old_data[i:i + BLOCK_SIZE]
            block_hash = hashlib.sha256(block).digest()
            if block_hash not in old_blocks:
                old_blocks[block_hash] = []
            old_blocks[block_hash].append(i)

        # Generate diff operations
        diff_ops = []
        new_pos = 0

        while new_pos < len(new_data):
            # Try to find matching block in old file
            new_block = new_data[new_pos:new_pos + BLOCK_SIZE]
            block_hash = hashlib.sha256(new_block).digest()

            if block_hash in old_blocks:
                # Found matching block - use COPY operation
                old_offset = old_blocks[block_hash][0]
                block_len = min(BLOCK_SIZE, len(new_data) - new_pos)

                # Verify the match
                if old_data[old_offset:old_offset + block_len] == new_block:
                    diff_ops.append((0, old_offset, block_len))
                    new_pos += block_len
                else:
                    # Hash collision, use INSERT
                    diff_ops.append((1, new_block))
                    new_pos += len(new_block)
            else:
                # No match - use INSERT operation
                diff_ops.append((1, new_block))
                new_pos += len(new_block)

        # Build diff data
        diff_data = b'GBINDIFF'
        diff_data += struct.pack('<I', 1)  # Version
        diff_data += struct.pack('<I', BLOCK_SIZE)
        diff_data += struct.pack('<Q', len(old_data))
        diff_data += struct.pack('<Q', len(new_data))

        for op in diff_ops:
            if op[0] == 0:  # COPY
                _, offset, length = op
                diff_data += struct.pack('<BQI', 0, offset, length)
            else:  # INSERT
                _, data = op
                diff_data += struct.pack('<BI', 1, len(data))
                diff_data += data

        # Compress the diff data
        return zlib.compress(diff_data)

    def _apply_binary_diff(self, old_data: bytes, diff_data: bytes) -> bytes:
        """
        Apply binary diff to reconstruct new file

        Args:
            old_data: Original binary data
            diff_data: Binary diff data from _generate_binary_diff

        Returns:
            Reconstructed binary data
        """
        # Decompress diff data
        try:
            diff_data = zlib.decompress(diff_data)
        except zlib.error:
            raise ValueError("Invalid compressed diff data")

        # Verify header
        if not diff_data.startswith(b'GBINDIFF'):
            raise ValueError("Invalid binary diff format")

        pos = 8  # Skip header

        # Read metadata
        version = struct.unpack('<I', diff_data[pos:pos + 4])[0]
        pos += 4

        if version != 1:
            raise ValueError(f"Unsupported binary diff version: {version}")

        block_size = struct.unpack('<I', diff_data[pos:pos + 4])[0]
        pos += 4

        old_size = struct.unpack('<Q', diff_data[pos:pos + 8])[0]
        pos += 8

        new_size = struct.unpack('<Q', diff_data[pos:pos + 8])[0]
        pos += 8

        # Verify old file size
        if len(old_data) != old_size:
            raise ValueError(f"Old file size mismatch: expected {old_size}, got {len(old_data)}")

        # Reconstruct new file
        new_data = bytearray()

        while pos < len(diff_data):
            operation = struct.unpack('<B', diff_data[pos:pos + 1])[0]
            pos += 1

            if operation == 0:  # COPY
                offset = struct.unpack('<Q', diff_data[pos:pos + 8])[0]
                pos += 8
                length = struct.unpack('<I', diff_data[pos:pos + 4])[0]
                pos += 4

                # Copy from old file
                new_data.extend(old_data[offset:offset + length])

            elif operation == 1:  # INSERT
                length = struct.unpack('<I', diff_data[pos:pos + 4])[0]
                pos += 4
                data = diff_data[pos:pos + length]
                pos += length

                # Insert new data
                new_data.extend(data)
            else:
                raise ValueError(f"Unknown operation: {operation}")

        return bytes(new_data)

    def generate_patch(self, file_path: Path, baseline_file: Path = None,
                      is_binary: bool = False) -> bytes:
        """
        Generate patch for a file by comparing current version with baseline

        Args:
            file_path: Path to current file (relative to root_dir)
            baseline_file: Path to baseline file (absolute path)
            is_binary: Whether file is binary

        Returns:
            Patch data as bytes
        """
        current_file = self.root_dir / file_path

        if not current_file.exists():
            raise ValueError(f"Current file does not exist: {current_file}")

        # For text files, use unified diff
        if not is_binary:
            try:
                # Read baseline file
                if baseline_file and baseline_file.exists():
                    with open(baseline_file, 'r', encoding='utf-8') as f:
                        old_content = f.read()
                else:
                    old_content = ""

                # Read current file
                with open(current_file, 'r', encoding='utf-8') as f:
                    new_content = f.read()

                # Generate diff
                diff = self._generate_text_diff(old_content, new_content)
                return diff.encode('utf-8')

            except UnicodeDecodeError:
                # If we fail to decode as text, treat as binary
                is_binary = True

        # For binary files, use binary diff
        if is_binary:
            # Read baseline file
            if baseline_file and baseline_file.exists():
                with open(baseline_file, 'rb') as f:
                    old_data = f.read()
            else:
                old_data = b""

            # Read current file
            with open(current_file, 'rb') as f:
                new_data = f.read()

            # Generate binary diff
            return self._generate_binary_diff(old_data, new_data)

    def _apply_text_patch(self, original_lines: List[str], patch_text: str) -> str:
        """
        Apply a unified diff patch to text content

        Args:
            original_lines: Original file lines
            patch_text: Unified diff patch

        Returns:
            Patched content as string
        """
        # Parse the unified diff
        patch_lines = patch_text.splitlines()

        # Simple patch application (handles unified diff format)
        result_lines = list(original_lines)

        i = 0
        line_offset = 0

        while i < len(patch_lines):
            line = patch_lines[i]

            # Look for hunk headers (@@ -start,count +start,count @@)
            if line.startswith('@@'):
                # Parse hunk header
                parts = line.split()
                if len(parts) >= 3:
                    old_info = parts[1].lstrip('-').split(',')
                    new_info = parts[2].lstrip('+').split(',')

                    old_start = int(old_info[0]) - 1  # Convert to 0-indexed
                    new_start = int(new_info[0]) - 1

                    i += 1

                    # Process hunk
                    old_pos = old_start
                    new_pos = new_start + line_offset

                    while i < len(patch_lines):
                        hunk_line = patch_lines[i]

                        if hunk_line.startswith('@@'):
                            # Next hunk, don't increment i
                            break
                        elif hunk_line.startswith('-'):
                            # Delete line
                            if new_pos < len(result_lines):
                                result_lines.pop(new_pos)
                                line_offset -= 1
                        elif hunk_line.startswith('+'):
                            # Add line
                            content = hunk_line[1:]
                            result_lines.insert(new_pos, content + '\n' if not content.endswith('\n') else content)
                            new_pos += 1
                            line_offset += 1
                        elif hunk_line.startswith(' '):
                            # Context line (unchanged)
                            new_pos += 1

                        i += 1
                        old_pos += 1
                else:
                    i += 1
            else:
                i += 1

        return ''.join(result_lines)

    def apply_patch(self, file_path: Path, patch_data: bytes, is_binary: bool = False,
                   baseline_file: Path = None, expected_hash: str = None) -> Dict[str, any]:
        """
        Apply patch to a file with comprehensive error handling and verification

        Args:
            file_path: Path to file (relative to root_dir)
            patch_data: Patch data
            is_binary: Whether file is binary
            baseline_file: Path to baseline file for patches
            expected_hash: Expected SHA256 hash of patched file (for verification)

        Returns:
            Dictionary with results:
            {
                'success': bool,
                'patched_file': Path to patched file,
                'verified': bool (if expected_hash provided),
                'actual_hash': str (computed hash),
                'error': str or None
            }
        """
        result = {
            'success': False,
            'patched_file': None,
            'verified': False,
            'actual_hash': None,
            'error': None
        }

        target_file = self.root_dir / file_path

        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Handle empty files edge case
            if len(patch_data) == 0:
                result['error'] = "Empty patch data provided"
                return result

            if not is_binary:
                # Apply text patch
                try:
                    patch_text = patch_data.decode('utf-8')
                except UnicodeDecodeError as e:
                    result['error'] = f"Failed to decode text patch: {e}"
                    return result

                # Read original file if exists, otherwise start with empty
                if baseline_file and baseline_file.exists():
                    try:
                        with open(baseline_file, 'r', encoding='utf-8') as f:
                            original_content = f.read()
                    except Exception as e:
                        result['error'] = f"Failed to read baseline file: {e}"
                        return result
                elif target_file.exists():
                    try:
                        with open(target_file, 'r', encoding='utf-8') as f:
                            original_content = f.read()
                    except Exception as e:
                        result['error'] = f"Failed to read existing file: {e}"
                        return result
                else:
                    original_content = ""

                original_lines = original_content.splitlines(keepends=True)

                # Apply patch
                try:
                    patched_content = self._apply_text_patch(original_lines, patch_text)
                except Exception as e:
                    result['error'] = f"Failed to apply text patch: {e}"
                    return result

                # Use atomic write (temp file + rename)
                success = self._write_file_atomic(target_file, patched_content, is_binary=False)
                if not success:
                    result['error'] = "Atomic write failed for text file"
                    return result

            else:
                # For binary files, check if this is a binary diff or full file
                if patch_data.startswith(b'x\x9c') or patch_data.startswith(b'\x78\x01'):
                    # Compressed binary diff format
                    # Read baseline file
                    if baseline_file and baseline_file.exists():
                        try:
                            with open(baseline_file, 'rb') as f:
                                old_data = f.read()
                        except Exception as e:
                            result['error'] = f"Failed to read baseline binary file: {e}"
                            return result
                    elif target_file.exists():
                        try:
                            with open(target_file, 'rb') as f:
                                old_data = f.read()
                        except Exception as e:
                            result['error'] = f"Failed to read existing binary file: {e}"
                            return result
                    else:
                        old_data = b""

                    # Apply binary diff
                    try:
                        new_data = self._apply_binary_diff(old_data, patch_data)
                    except Exception as e:
                        result['error'] = f"Failed to apply binary diff: {e}"
                        return result
                else:
                    # Full file content (legacy or small files)
                    new_data = patch_data

                # Use atomic write (temp file + rename)
                success = self._write_file_atomic(target_file, new_data, is_binary=True)
                if not success:
                    result['error'] = "Atomic write failed for binary file"
                    return result

            # Compute hash of patched file for verification
            try:
                actual_hash = self._compute_file_hash(target_file)
                result['actual_hash'] = actual_hash
            except Exception as e:
                result['error'] = f"Failed to compute hash of patched file: {e}"
                return result

            # Verify hash if expected hash provided
            if expected_hash:
                if actual_hash == expected_hash:
                    result['verified'] = True
                else:
                    result['error'] = f"Hash mismatch: expected {expected_hash}, got {actual_hash}"
                    # Clean up the incorrectly patched file
                    try:
                        if target_file.exists():
                            target_file.unlink()
                    except:
                        pass
                    return result

            result['success'] = True
            result['patched_file'] = target_file

        except Exception as e:
            result['error'] = f"Unexpected error applying patch to {file_path}: {e}"

        return result

    def apply_patches_atomic(self, patches: List[Dict[str, any]],
                            baseline_manager=None) -> Dict[str, any]:
        """
        Apply multiple patches atomically with rollback on failure

        Args:
            patches: List of patch dictionaries, each containing:
                {
                    'file_path': str (relative path),
                    'patch_data': bytes,
                    'is_binary': bool,
                    'expected_hash': str,
                    'baseline_file': Path (optional)
                }
            baseline_manager: Optional BaselineManager for accessing baseline files

        Returns:
            Dictionary with results:
            {
                'success': bool,
                'applied': list of successfully applied file paths,
                'failed': list of failed file paths,
                'errors': dict mapping file paths to error messages,
                'rollback_performed': bool
            }
        """
        result = {
            'success': True,
            'applied': [],
            'failed': [],
            'errors': {},
            'rollback_performed': False
        }

        # Track original states for rollback
        backup_dir = self.root_dir / '.gibberish' / 'cache' / 'patch_backup'
        backup_dir.mkdir(parents=True, exist_ok=True)

        backed_up_files = []

        try:
            # Phase 1: Backup existing files
            for patch in patches:
                file_path = Path(patch['file_path'])
                target_file = self.root_dir / file_path

                if target_file.exists():
                    # Create backup
                    backup_file = backup_dir / file_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target_file, backup_file)
                    backed_up_files.append(file_path)

            # Phase 2: Apply patches
            for patch in patches:
                file_path = Path(patch['file_path'])
                patch_data = patch['patch_data']
                is_binary = patch.get('is_binary', False)
                expected_hash = patch.get('expected_hash')
                baseline_file = patch.get('baseline_file')

                # Get baseline file from manager if not provided
                if baseline_manager and not baseline_file:
                    baseline_file = baseline_manager.get_baseline_file('current', str(file_path))

                # Apply patch
                patch_result = self.apply_patch(
                    file_path=file_path,
                    patch_data=patch_data,
                    is_binary=is_binary,
                    baseline_file=baseline_file,
                    expected_hash=expected_hash
                )

                if patch_result['success']:
                    result['applied'].append(str(file_path))
                else:
                    result['failed'].append(str(file_path))
                    result['errors'][str(file_path)] = patch_result['error']
                    result['success'] = False

                    # Rollback on first failure
                    print(f"Patch application failed for {file_path}: {patch_result['error']}")
                    print("Rolling back all changes...")
                    self._rollback_patches(backup_dir, backed_up_files)
                    result['rollback_performed'] = True
                    break

        except Exception as e:
            result['success'] = False
            result['errors']['_general'] = str(e)
            print(f"Unexpected error during patch application: {e}")
            print("Rolling back all changes...")
            self._rollback_patches(backup_dir, backed_up_files)
            result['rollback_performed'] = True

        finally:
            # Cleanup backup directory
            if backup_dir.exists():
                try:
                    shutil.rmtree(backup_dir)
                except Exception as e:
                    print(f"Warning: Failed to cleanup backup directory: {e}")

        return result

    def _rollback_patches(self, backup_dir: Path, backed_up_files: List[Path]):
        """
        Rollback patched files from backup

        Args:
            backup_dir: Directory containing backed up files
            backed_up_files: List of file paths that were backed up
        """
        for file_path in backed_up_files:
            backup_file = backup_dir / file_path
            target_file = self.root_dir / file_path

            if backup_file.exists():
                try:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, target_file)
                    print(f"Restored {file_path} from backup")
                except Exception as e:
                    print(f"Warning: Failed to restore {file_path}: {e}")

    def request_full_file(self, file_path: Path, protocol_handler, audio_manager,
                         expected_hash: str = None, chunk_size: int = 1024 * 64) -> Dict[str, any]:
        """
        Request and receive full file when patch application fails

        This implements fallback to full file transmission when patches cannot be applied.

        Args:
            file_path: Relative path to file
            protocol_handler: ProtocolHandler instance for communication
            audio_manager: AudioManager instance for data transfer
            expected_hash: Expected SHA256 hash of file
            chunk_size: Size of chunks for transfer (default 64KB)

        Returns:
            Dictionary with results:
            {
                'success': bool,
                'file_path': Path,
                'size': int (bytes received),
                'hash': str (actual hash),
                'verified': bool,
                'error': str or None
            }
        """
        result = {
            'success': False,
            'file_path': file_path,
            'size': 0,
            'hash': None,
            'verified': False,
            'error': None
        }

        try:
            # Send request for full file
            # This would integrate with protocol.py to send a request message
            print(f"Requesting full file transfer for {file_path}...")

            # For now, this is a placeholder - the actual implementation would
            # coordinate with the sender via the acoustic protocol
            result['error'] = "Full file transfer not yet implemented in protocol"

            # The full implementation would:
            # 1. Send FILE_REQUEST frame via protocol_handler
            # 2. Receive file data in chunks via protocol_handler.receive_and_reassemble()
            # 3. Write chunks to temp file
            # 4. Verify hash
            # 5. Atomic rename to final location

        except Exception as e:
            result['error'] = str(e)

        return result

    def verify_integrity(self, file_path: Path, expected_hash: str) -> bool:
        """
        Verify file integrity using hash

        Args:
            file_path: Path to file (relative to root_dir)
            expected_hash: Expected SHA256 hash

        Returns:
            True if hash matches
        """
        full_path = self.root_dir / file_path if not file_path.is_absolute() else file_path

        if not full_path.exists():
            return False

        try:
            actual_hash = self._compute_file_hash(full_path)
            return actual_hash == expected_hash
        except ValueError:
            return False

    def _get_cache_key(self, file_path: str, old_hash: str, new_hash: str) -> str:
        """
        Generate cache key for a diff

        Args:
            file_path: Relative file path
            old_hash: Hash of old file
            new_hash: Hash of new file

        Returns:
            Cache key string
        """
        # Use combination of file path and hashes to create unique cache key
        key_data = f"{file_path}:{old_hash}:{new_hash}"
        return hashlib.sha256(key_data.encode('utf-8')).hexdigest()

    def cache_diff(self, file_path: str, old_hash: str, new_hash: str,
                  diff_data: bytes, is_binary: bool = False) -> bool:
        """
        Cache a diff for later retrieval

        Args:
            file_path: Relative file path
            old_hash: Hash of old file
            new_hash: Hash of new file
            diff_data: Diff/patch data
            is_binary: Whether this is a binary diff

        Returns:
            True if cached successfully
        """
        try:
            cache_key = self._get_cache_key(file_path, old_hash, new_hash)
            cache_file = self.cache_dir / f"{cache_key}.diff"

            # Create metadata
            metadata = {
                'file_path': file_path,
                'old_hash': old_hash,
                'new_hash': new_hash,
                'is_binary': is_binary,
                'cached_at': datetime.utcnow().isoformat() + 'Z',
                'diff_size': len(diff_data)
            }

            # Write metadata
            metadata_file = self.cache_dir / f"{cache_key}.meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Write diff data
            with open(cache_file, 'wb') as f:
                f.write(diff_data)

            return True

        except Exception as e:
            print(f"Error caching diff for {file_path}: {e}")
            return False

    def get_cached_diff(self, file_path: str, old_hash: str, new_hash: str) -> Optional[bytes]:
        """
        Retrieve a cached diff

        Args:
            file_path: Relative file path
            old_hash: Hash of old file
            new_hash: Hash of new file

        Returns:
            Diff data if found, None otherwise
        """
        try:
            cache_key = self._get_cache_key(file_path, old_hash, new_hash)
            cache_file = self.cache_dir / f"{cache_key}.diff"

            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return f.read()

            return None

        except Exception as e:
            print(f"Error retrieving cached diff for {file_path}: {e}")
            return None

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear diff cache

        Args:
            older_than_days: Only clear diffs older than this many days (None = clear all)

        Returns:
            Number of cache entries cleared
        """
        cleared = 0

        try:
            if not self.cache_dir.exists():
                return 0

            current_time = datetime.utcnow().timestamp()
            cutoff_time = current_time - (older_than_days * 86400) if older_than_days else None

            for cache_file in self.cache_dir.glob("*.diff"):
                try:
                    # Check age if specified
                    if cutoff_time:
                        file_mtime = cache_file.stat().st_mtime
                        if file_mtime > cutoff_time:
                            continue

                    # Remove diff and metadata
                    cache_file.unlink()
                    metadata_file = cache_file.with_suffix('.meta.json')
                    if metadata_file.exists():
                        metadata_file.unlink()

                    cleared += 1

                except Exception as e:
                    print(f"Error removing cache file {cache_file}: {e}")

            return cleared

        except Exception as e:
            print(f"Error clearing cache: {e}")
            return cleared

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the diff cache

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'entry_count': 0,
            'total_size': 0,
            'oldest_entry': None,
            'newest_entry': None
        }

        try:
            if not self.cache_dir.exists():
                return stats

            oldest = None
            newest = None

            for cache_file in self.cache_dir.glob("*.diff"):
                stats['entry_count'] += 1
                stats['total_size'] += cache_file.stat().st_size

                mtime = cache_file.stat().st_mtime
                if oldest is None or mtime < oldest:
                    oldest = mtime
                if newest is None or mtime > newest:
                    newest = mtime

            if oldest:
                stats['oldest_entry'] = datetime.fromtimestamp(oldest).isoformat()
            if newest:
                stats['newest_entry'] = datetime.fromtimestamp(newest).isoformat()

        except Exception as e:
            print(f"Error getting cache stats: {e}")

        return stats

    def _format_size(self, size_bytes: int) -> str:
        """
        Format byte size as human-readable string

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string (e.g., "1.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def compute_sync_efficiency(self, changes: List[FileChange],
                                baseline_manager=None) -> Dict[str, Any]:
        """
        Compute efficiency metrics for a set of changes

        Args:
            changes: List of FileChange objects
            baseline_manager: Optional BaselineManager for accessing baseline files

        Returns:
            Dictionary with efficiency metrics
        """
        metrics = {
            'total_files': len(changes),
            'by_type': {
                'add': 0,
                'modify': 0,
                'delete': 0
            },
            'original_total_size': 0,
            'transfer_total_size': 0,
            'reduction_bytes': 0,
            'reduction_percentage': 0.0,
            'per_file_metrics': []
        }

        for change in changes:
            # Count by type
            if change.change_type == ChangeType.ADD:
                metrics['by_type']['add'] += 1
            elif change.change_type == ChangeType.MODIFY:
                metrics['by_type']['modify'] += 1
            elif change.change_type == ChangeType.DELETE:
                metrics['by_type']['delete'] += 1

            # For MODIFY changes, generate patch to get accurate diff size
            if change.change_type == ChangeType.MODIFY and change.diff_size == 0:
                if baseline_manager:
                    try:
                        baseline_file = baseline_manager.get_baseline_file('current', str(change.path))
                        if baseline_file:
                            patch_data = self.generate_patch(change.path, baseline_file, change.is_binary)
                            change.diff_size = len(patch_data)
                    except Exception as e:
                        print(f"Warning: Could not generate patch for {change.path}: {e}")
                        change.diff_size = change.size  # Fallback to full file size

            # Get efficiency for this change
            efficiency = change.get_efficiency()

            metrics['original_total_size'] += efficiency['original_size']
            metrics['transfer_total_size'] += efficiency['transfer_size']
            metrics['reduction_bytes'] += efficiency['reduction_bytes']

            # Add per-file metric
            metrics['per_file_metrics'].append({
                'path': str(change.path),
                'type': change.change_type.value,
                'original_size': efficiency['original_size'],
                'transfer_size': efficiency['transfer_size'],
                'reduction': efficiency['reduction'],
                'reduction_bytes': efficiency['reduction_bytes'],
                'is_binary': change.is_binary
            })

        # Calculate overall reduction percentage
        if metrics['original_total_size'] > 0:
            metrics['reduction_percentage'] = (
                metrics['reduction_bytes'] / metrics['original_total_size'] * 100.0
            )

        return metrics

    def format_efficiency_report(self, metrics: Dict[str, Any]) -> str:
        """
        Format efficiency metrics as a human-readable report

        Args:
            metrics: Metrics from compute_sync_efficiency()

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SYNC EFFICIENCY REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        lines.append(f"Total Files: {metrics['total_files']}")
        lines.append(f"  - ADD:    {metrics['by_type']['add']}")
        lines.append(f"  - MODIFY: {metrics['by_type']['modify']}")
        lines.append(f"  - DELETE: {metrics['by_type']['delete']}")
        lines.append("")

        # Size statistics
        lines.append(f"Original Total Size:  {self._format_size(metrics['original_total_size'])}")
        lines.append(f"Transfer Size:        {self._format_size(metrics['transfer_total_size'])}")
        lines.append(f"Reduction:            {self._format_size(metrics['reduction_bytes'])} "
                    f"({metrics['reduction_percentage']:.1f}%)")
        lines.append("")

        # Per-file details
        if metrics['per_file_metrics']:
            lines.append("Per-File Details:")
            lines.append("-" * 70)

            for file_metric in metrics['per_file_metrics']:
                path = file_metric['path']
                change_type = file_metric['type'].upper()
                orig_size = self._format_size(file_metric['original_size'])
                trans_size = self._format_size(file_metric['transfer_size'])
                reduction = file_metric['reduction']
                is_binary = " [BINARY]" if file_metric['is_binary'] else ""

                if change_type == 'MODIFY':
                    lines.append(f"  {change_type:7} {path}{is_binary}")
                    lines.append(f"           {orig_size} -> {trans_size} ({reduction:.1f}% reduction)")
                elif change_type == 'ADD':
                    lines.append(f"  {change_type:7} {path}{is_binary}")
                    lines.append(f"           {trans_size}")
                elif change_type == 'DELETE':
                    lines.append(f"  {change_type:7} {path}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def add_ignore_pattern(self, pattern: str) -> None:
        """
        Add a new ignore pattern

        Args:
            pattern: Pattern to add (gitignore-style)
        """
        if pattern not in self.ignore_patterns:
            self.ignore_patterns.append(pattern)

    def remove_ignore_pattern(self, pattern: str) -> bool:
        """
        Remove an ignore pattern

        Args:
            pattern: Pattern to remove

        Returns:
            True if pattern was removed, False if not found
        """
        if pattern in self.ignore_patterns:
            self.ignore_patterns.remove(pattern)
            return True
        return False

    def get_ignore_patterns(self) -> List[str]:
        """
        Get current ignore patterns

        Returns:
            List of ignore patterns
        """
        return self.ignore_patterns.copy()

    def set_max_file_size(self, size_bytes: int) -> None:
        """
        Set maximum file size for syncing

        Args:
            size_bytes: Maximum size in bytes
        """
        self.max_file_size = size_bytes

    def get_max_file_size(self) -> int:
        """
        Get maximum file size for syncing

        Returns:
            Maximum size in bytes
        """
        return self.max_file_size

    def save_ignore_config(self, config_file: Path = None) -> bool:
        """
        Save current ignore patterns and size limits to config file

        Args:
            config_file: Path to config file (default: .gibberish/sync_config.json)

        Returns:
            True if saved successfully
        """
        if config_file is None:
            config_file = self.root_dir / '.gibberish' / 'sync_config.json'

        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)

            config = {
                'ignore_patterns': self.ignore_patterns,
                'max_file_size': self.max_file_size,
                'updated_at': datetime.utcnow().isoformat() + 'Z'
            }

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def load_ignore_config(self, config_file: Path = None) -> bool:
        """
        Load ignore patterns and size limits from config file

        Args:
            config_file: Path to config file (default: .gibberish/sync_config.json)

        Returns:
            True if loaded successfully
        """
        if config_file is None:
            config_file = self.root_dir / '.gibberish' / 'sync_config.json'

        try:
            if not config_file.exists():
                return False

            with open(config_file, 'r') as f:
                config = json.load(f)

            if 'ignore_patterns' in config:
                self.ignore_patterns = config['ignore_patterns']

            if 'max_file_size' in config:
                self.max_file_size = config['max_file_size']

            return True

        except Exception as e:
            print(f"Error loading config: {e}")
            return False

    def update_baseline_and_log_sync(self, baseline_manager, peer_id: str,
                                    direction: str, files_synced: int,
                                    bytes_transferred: int, duration: float,
                                    tree_hash: str) -> Dict[str, any]:
        """
        Update baseline atomically and log sync session to state.json

        This is called after successful synchronization to:
        1. Update the baseline to reflect new state
        2. Log the sync session for auditing and tracking

        Args:
            baseline_manager: BaselineManager instance
            peer_id: ID of peer machine
            direction: 'send' or 'receive'
            files_synced: Number of files synchronized
            bytes_transferred: Total bytes transferred
            duration: Duration of sync in seconds
            tree_hash: Final tree hash after sync

        Returns:
            Dictionary with results:
            {
                'success': bool,
                'baseline_updated': bool,
                'sync_logged': bool,
                'error': str or None
            }
        """
        result = {
            'success': False,
            'baseline_updated': False,
            'sync_logged': False,
            'error': None
        }

        try:
            # Phase 1: Update baseline atomically using BaselineManager
            print("Updating baseline after successful sync...")
            baseline_result = baseline_manager.refresh_baseline(backup=True)

            if baseline_result['success']:
                result['baseline_updated'] = True
                print(f"Baseline updated successfully (tree_hash: {baseline_result['tree_hash'][:16]}...)")
            else:
                result['error'] = f"Failed to update baseline: {baseline_result.get('error')}"
                return result

            # Phase 2: Log sync session to state.json
            print("Logging sync session to state.json...")
            log_success = self._log_sync_session(
                peer_id=peer_id,
                direction=direction,
                files_synced=files_synced,
                bytes_transferred=bytes_transferred,
                duration=duration,
                tree_hash=tree_hash
            )

            if log_success:
                result['sync_logged'] = True
                result['success'] = True
                print("Sync session logged successfully")
            else:
                result['error'] = "Failed to log sync session"

        except Exception as e:
            result['error'] = f"Unexpected error: {e}"

        return result

    def _log_sync_session(self, peer_id: str, direction: str, files_synced: int,
                         bytes_transferred: int, duration: float, tree_hash: str) -> bool:
        """
        Log sync session to state.json with atomic write

        Args:
            peer_id: ID of peer machine
            direction: 'send' or 'receive'
            files_synced: Number of files synchronized
            bytes_transferred: Total bytes transferred
            duration: Duration of sync in seconds
            tree_hash: Final tree hash after sync

        Returns:
            True if logged successfully
        """
        import tempfile

        state_dir = self.root_dir / '.gibberish' / 'state'
        state_dir.mkdir(parents=True, exist_ok=True)

        state_file = state_dir / 'state.json'

        try:
            # Load existing state or create new
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
            else:
                state = {
                    'version': '1.0',
                    'sync_sessions': []
                }

            # Create session entry
            session = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'peer_id': peer_id,
                'direction': direction,
                'files_synced': files_synced,
                'bytes_transferred': bytes_transferred,
                'duration_seconds': duration,
                'tree_hash': tree_hash
            }

            # Add to sessions list
            state['sync_sessions'].append(session)

            # Keep only last 100 sessions to prevent unbounded growth
            if len(state['sync_sessions']) > 100:
                state['sync_sessions'] = state['sync_sessions'][-100:]

            # Update last_sync metadata
            state['last_sync'] = {
                'timestamp': session['timestamp'],
                'peer_id': peer_id,
                'direction': direction,
                'tree_hash': tree_hash
            }

            # Write atomically using temp file + rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=state_dir,
                prefix='.state.',
                suffix='.json.tmp'
            )

            try:
                # Write JSON to temp file
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(state, f, indent=2)

                # Atomic rename
                os.replace(temp_path, state_file)
                return True

            except Exception as e:
                # Cleanup temp file on error
                try:
                    os.close(temp_fd)
                except:
                    pass
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
                raise e

        except Exception as e:
            print(f"Error logging sync session: {e}")
            return False

    def get_sync_history(self, limit: int = 10) -> List[Dict[str, any]]:
        """
        Get recent sync session history

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of sync session dictionaries (most recent first)
        """
        state_file = self.root_dir / '.gibberish' / 'state' / 'state.json'

        try:
            if not state_file.exists():
                return []

            with open(state_file, 'r') as f:
                state = json.load(f)

            sessions = state.get('sync_sessions', [])
            # Return most recent sessions first
            return sessions[-limit:][::-1]

        except Exception as e:
            print(f"Error reading sync history: {e}")
            return []
