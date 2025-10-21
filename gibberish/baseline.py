"""
Baseline state management and verification.

This module handles creation, storage, and verification of filesystem baselines
using tree hashes to detect changes and ensure synchronization integrity.
"""

from pathlib import Path
from typing import Dict, Optional, List, Set
import hashlib
import json
import os
import shutil
from datetime import datetime


class BaselineManager:
    """Manages baseline state for file synchronization"""

    def __init__(self, root_dir: Path, baseline_dir: Path = None):
        """
        Initialize the baseline manager

        Args:
            root_dir: Root directory to manage
            baseline_dir: Directory to store baseline data (default: .gibberish/baseline/)
        """
        self.root_dir = Path(root_dir)
        self.baseline_dir = baseline_dir or (self.root_dir / ".gibberish" / "baseline")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of a single file

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except (IOError, OSError) as e:
            raise ValueError(f"Failed to hash file {file_path}: {e}")

    def _get_all_files(self, directory: Path, ignore_patterns: Set[str] = None) -> List[Path]:
        """
        Recursively get all files in a directory, excluding ignored patterns

        Args:
            directory: Directory to scan
            ignore_patterns: Set of patterns to ignore (default: .gibberish and .git)

        Returns:
            List of file paths relative to directory
        """
        if ignore_patterns is None:
            ignore_patterns = {'.gibberish', '.git', '__pycache__', '.DS_Store'}

        all_files = []

        for root, dirs, files in os.walk(directory):
            # Remove ignored directories from dirs list (modifies in-place to prevent os.walk from descending)
            dirs[:] = [d for d in dirs if d not in ignore_patterns]

            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                # Skip ignored files
                if file not in ignore_patterns:
                    # Store relative path
                    rel_path = file_path.relative_to(directory)
                    all_files.append(rel_path)

        return sorted(all_files)  # Sort for deterministic ordering

    def compute_tree_hash(self, directory: Path = None) -> str:
        """
        Compute tree hash for a directory using sha256(sorted([path + file_hash for all files]))

        Args:
            directory: Directory to hash (default: root_dir)

        Returns:
            SHA256 tree hash as hex string
        """
        directory = Path(directory) if directory else self.root_dir

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Get all files in sorted order
        files = self._get_all_files(directory)

        # Compute combined hash: sha256(sorted([path + file_hash for all files]))
        tree_hasher = hashlib.sha256()

        for rel_path in files:
            abs_path = directory / rel_path
            try:
                file_hash = self._compute_file_hash(abs_path)
                # Combine path and hash with a separator to avoid collisions
                combined = f"{rel_path.as_posix()}:{file_hash}\n"
                tree_hasher.update(combined.encode('utf-8'))
            except ValueError as e:
                # Log but continue - this allows partial tree hashing
                print(f"Warning: {e}")
                continue

        return tree_hasher.hexdigest()

    def create_baseline(self) -> Dict[str, any]:
        """
        Create baseline snapshot of current directory state

        Returns:
            Dictionary with baseline metadata and file hashes:
            {
                'version': '1.0',
                'created': ISO timestamp,
                'tree_hash': tree hash,
                'files': {path: hash, ...}
            }
        """
        files = self._get_all_files(self.root_dir)

        # Create file hash mapping
        file_hashes = {}
        for rel_path in files:
            abs_path = self.root_dir / rel_path
            try:
                file_hash = self._compute_file_hash(abs_path)
                file_hashes[rel_path.as_posix()] = file_hash
            except ValueError as e:
                print(f"Warning: Skipping file {rel_path}: {e}")
                continue

        # Compute tree hash
        tree_hash = self.compute_tree_hash(self.root_dir)

        # Create baseline metadata
        baseline = {
            'version': '1.0',
            'created': datetime.utcnow().isoformat() + 'Z',
            'tree_hash': tree_hash,
            'root_dir': str(self.root_dir.resolve()),
            'files': file_hashes
        }

        return baseline

    def save_baseline(self, baseline: Dict[str, any], name: str = "current") -> bool:
        """
        Save baseline to disk with complete file copies and metadata

        Args:
            baseline: Baseline data to save
            name: Baseline name

        Returns:
            True if saved successfully
        """
        try:
            # Create baseline subdirectory for this baseline
            baseline_subdir = self.baseline_dir / name
            baseline_subdir.mkdir(parents=True, exist_ok=True)

            # Save metadata JSON
            metadata_file = baseline_subdir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(baseline, f, indent=2)

            # Create files subdirectory
            files_dir = baseline_subdir / "files"
            files_dir.mkdir(exist_ok=True)

            # Copy all files to baseline storage
            if 'files' in baseline:
                for file_path in baseline['files'].keys():
                    src_path = self.root_dir / file_path
                    dst_path = files_dir / file_path

                    # Create parent directories
                    dst_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file
                    if src_path.exists():
                        shutil.copy2(src_path, dst_path)

            return True
        except Exception as e:
            print(f"Error saving baseline: {e}")
            return False

    def load_baseline(self, name: str = "current") -> Optional[Dict[str, any]]:
        """
        Load baseline from disk

        Args:
            name: Baseline name

        Returns:
            Baseline data or None if not found
        """
        baseline_subdir = self.baseline_dir / name
        metadata_file = baseline_subdir / "metadata.json"

        try:
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    baseline = json.load(f)
                return baseline
            else:
                # Try legacy format (single JSON file)
                legacy_file = self.baseline_dir / f"{name}.json"
                if legacy_file.exists():
                    with open(legacy_file, 'r') as f:
                        return json.load(f)
        except Exception as e:
            print(f"Error loading baseline: {e}")

        return None

    def get_baseline_file(self, name: str, file_path: str) -> Optional[Path]:
        """
        Get path to a specific file in baseline storage

        Args:
            name: Baseline name
            file_path: Relative file path

        Returns:
            Path to baseline file or None if not found
        """
        baseline_subdir = self.baseline_dir / name
        file_in_baseline = baseline_subdir / "files" / file_path

        if file_in_baseline.exists():
            return file_in_baseline

        return None

    def verify_baseline(self, baseline: Dict[str, any], name: str = "current") -> Dict[str, any]:
        """
        Verify baseline integrity by checking stored files against their recorded hashes

        Args:
            baseline: Baseline to verify
            name: Baseline name for file lookups

        Returns:
            Dictionary with verification results:
            {
                'valid': bool,
                'tree_hash_matches': bool,
                'corrupted_files': [list of corrupted file paths],
                'missing_files': [list of missing file paths],
                'extra_files': [list of unexpected files]
            }
        """
        result = {
            'valid': True,
            'tree_hash_matches': False,
            'corrupted_files': [],
            'missing_files': [],
            'extra_files': []
        }

        if not baseline or 'files' not in baseline:
            result['valid'] = False
            return result

        baseline_subdir = self.baseline_dir / name
        files_dir = baseline_subdir / "files"

        if not files_dir.exists():
            result['valid'] = False
            result['missing_files'] = list(baseline['files'].keys())
            return result

        # Check each file in baseline
        for file_path, expected_hash in baseline['files'].items():
            baseline_file = files_dir / file_path

            if not baseline_file.exists():
                result['missing_files'].append(file_path)
                result['valid'] = False
            else:
                try:
                    actual_hash = self._compute_file_hash(baseline_file)
                    if actual_hash != expected_hash:
                        result['corrupted_files'].append(file_path)
                        result['valid'] = False
                except ValueError as e:
                    result['corrupted_files'].append(file_path)
                    result['valid'] = False

        # Check for extra files
        if files_dir.exists():
            stored_files = set()
            for root, _, files in os.walk(files_dir):
                root_path = Path(root)
                for file in files:
                    file_path = root_path / file
                    rel_path = file_path.relative_to(files_dir)
                    stored_files.add(rel_path.as_posix())

            baseline_files = set(baseline['files'].keys())
            extra = stored_files - baseline_files
            if extra:
                result['extra_files'] = list(extra)
                result['valid'] = False

        # Verify tree hash by recomputing it from baseline files
        if result['valid'] and 'tree_hash' in baseline:
            # Temporarily compute tree hash from baseline directory
            temp_root = self.root_dir
            self.root_dir = files_dir
            try:
                computed_tree_hash = self.compute_tree_hash(files_dir)
                result['tree_hash_matches'] = (computed_tree_hash == baseline['tree_hash'])
                if not result['tree_hash_matches']:
                    result['valid'] = False
            except Exception as e:
                print(f"Error computing tree hash: {e}")
                result['valid'] = False
            finally:
                self.root_dir = temp_root

        return result

    def compare_baselines(self, baseline_a: Dict[str, any],
                         baseline_b: Dict[str, any]) -> Dict[str, List[str]]:
        """
        Compare two baselines to detect differences

        Args:
            baseline_a: First baseline (original/source)
            baseline_b: Second baseline (current/target)

        Returns:
            Dictionary with keys:
            - 'added': files in B but not in A
            - 'modified': files in both but with different hashes
            - 'deleted': files in A but not in B
            - 'tree_hash_match': boolean indicating if tree hashes match
        """
        result = {
            'added': [],
            'modified': [],
            'deleted': [],
            'tree_hash_match': False
        }

        # Quick check: if tree hashes match, no changes
        if (baseline_a.get('tree_hash') and baseline_b.get('tree_hash') and
                baseline_a['tree_hash'] == baseline_b['tree_hash']):
            result['tree_hash_match'] = True
            return result

        files_a = baseline_a.get('files', {})
        files_b = baseline_b.get('files', {})

        set_a = set(files_a.keys())
        set_b = set(files_b.keys())

        # Find added files (in B but not in A)
        result['added'] = sorted(list(set_b - set_a))

        # Find deleted files (in A but not in B)
        result['deleted'] = sorted(list(set_a - set_b))

        # Find modified files (in both but different hashes)
        common_files = set_a & set_b
        for file_path in common_files:
            if files_a[file_path] != files_b[file_path]:
                result['modified'].append(file_path)

        result['modified'] = sorted(result['modified'])

        return result

    def compare_with_current(self, baseline: Dict[str, any]) -> Dict[str, List[str]]:
        """
        Compare baseline with current directory state

        Args:
            baseline: Baseline to compare against

        Returns:
            Dictionary with change lists (same format as compare_baselines)
        """
        current_baseline = self.create_baseline()
        return self.compare_baselines(baseline, current_baseline)

    def check_file_integrity(self, file_path: Path, expected_hash: str) -> Dict[str, any]:
        """
        Check integrity of a single file

        Args:
            file_path: Path to file to check
            expected_hash: Expected SHA256 hash

        Returns:
            Dictionary with integrity check results:
            {
                'valid': bool,
                'exists': bool,
                'hash_matches': bool,
                'actual_hash': str or None,
                'error': str or None
            }
        """
        result = {
            'valid': False,
            'exists': False,
            'hash_matches': False,
            'actual_hash': None,
            'error': None
        }

        file_path = Path(file_path)

        if not file_path.exists():
            result['error'] = f"File does not exist: {file_path}"
            return result

        result['exists'] = True

        try:
            actual_hash = self._compute_file_hash(file_path)
            result['actual_hash'] = actual_hash
            result['hash_matches'] = (actual_hash == expected_hash)
            result['valid'] = result['hash_matches']

            if not result['hash_matches']:
                result['error'] = f"Hash mismatch: expected {expected_hash}, got {actual_hash}"

        except ValueError as e:
            result['error'] = str(e)

        return result

    def detect_corruption(self, baseline: Dict[str, any], check_current: bool = True) -> Dict[str, any]:
        """
        Comprehensive corruption detection for baseline and/or current files

        Args:
            baseline: Baseline to check against
            check_current: If True, also check current directory files against baseline

        Returns:
            Dictionary with corruption detection results:
            {
                'baseline_corrupted': bool,
                'current_corrupted': bool,
                'baseline_issues': {...},  # from verify_baseline
                'current_issues': {
                    'corrupted': [list of files with mismatched hashes],
                    'missing': [list of files missing from current dir]
                }
            }
        """
        result = {
            'baseline_corrupted': False,
            'current_corrupted': False,
            'baseline_issues': {},
            'current_issues': {
                'corrupted': [],
                'missing': []
            }
        }

        # Check baseline integrity
        baseline_name = 'current'  # Default baseline name
        baseline_verification = self.verify_baseline(baseline, baseline_name)
        result['baseline_issues'] = baseline_verification
        result['baseline_corrupted'] = not baseline_verification['valid']

        # Check current directory files if requested
        if check_current and 'files' in baseline:
            for file_path, expected_hash in baseline['files'].items():
                current_file = self.root_dir / file_path
                integrity = self.check_file_integrity(current_file, expected_hash)

                if not integrity['exists']:
                    result['current_issues']['missing'].append(file_path)
                    result['current_corrupted'] = True
                elif not integrity['hash_matches']:
                    result['current_issues']['corrupted'].append(file_path)
                    result['current_corrupted'] = True

        return result

    def repair_from_baseline(self, baseline: Dict[str, any], name: str = "current",
                            files_to_repair: List[str] = None) -> Dict[str, any]:
        """
        Repair corrupted or missing files from baseline storage

        Args:
            baseline: Baseline to repair from
            name: Baseline name
            files_to_repair: List of specific files to repair (None = repair all)

        Returns:
            Dictionary with repair results:
            {
                'success': bool,
                'repaired': [list of successfully repaired files],
                'failed': [list of files that couldn't be repaired],
                'errors': {file_path: error_message}
            }
        """
        result = {
            'success': True,
            'repaired': [],
            'failed': [],
            'errors': {}
        }

        if 'files' not in baseline:
            result['success'] = False
            result['errors']['_general'] = "Baseline has no file information"
            return result

        # Determine which files to repair
        files = files_to_repair if files_to_repair else baseline['files'].keys()

        for file_path in files:
            try:
                # Get file from baseline storage
                baseline_file = self.get_baseline_file(name, file_path)

                if not baseline_file:
                    result['failed'].append(file_path)
                    result['errors'][file_path] = "File not found in baseline storage"
                    result['success'] = False
                    continue

                # Verify baseline file integrity
                expected_hash = baseline['files'].get(file_path)
                if expected_hash:
                    integrity = self.check_file_integrity(baseline_file, expected_hash)
                    if not integrity['valid']:
                        result['failed'].append(file_path)
                        result['errors'][file_path] = f"Baseline file is corrupted: {integrity['error']}"
                        result['success'] = False
                        continue

                # Copy file from baseline to current directory
                current_file = self.root_dir / file_path
                current_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(baseline_file, current_file)

                # Verify repair was successful
                verify = self.check_file_integrity(current_file, expected_hash)
                if verify['valid']:
                    result['repaired'].append(file_path)
                else:
                    result['failed'].append(file_path)
                    result['errors'][file_path] = f"Repair verification failed: {verify['error']}"
                    result['success'] = False

            except Exception as e:
                result['failed'].append(file_path)
                result['errors'][file_path] = str(e)
                result['success'] = False

        return result

    def refresh_baseline(self, backup: bool = True) -> Dict[str, any]:
        """
        Refresh the current baseline atomically using temp+rename pattern

        Args:
            backup: If True, keep a backup of the old baseline

        Returns:
            Dictionary with refresh results:
            {
                'success': bool,
                'tree_hash': new tree hash,
                'backup_name': backup baseline name if created,
                'error': error message if failed
            }
        """
        result = {
            'success': False,
            'tree_hash': None,
            'backup_name': None,
            'error': None
        }

        try:
            # Create new baseline
            new_baseline = self.create_baseline()
            result['tree_hash'] = new_baseline.get('tree_hash')

            # Generate temporary name for atomic update
            temp_name = f"temp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

            # Save new baseline to temporary location
            if not self.save_baseline(new_baseline, temp_name):
                result['error'] = "Failed to save new baseline to temporary location"
                return result

            # If backup requested, rename current baseline
            if backup:
                current_baseline_dir = self.baseline_dir / "current"
                if current_baseline_dir.exists():
                    backup_name = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                    backup_dir = self.baseline_dir / backup_name

                    # Atomic rename of current to backup
                    current_baseline_dir.rename(backup_dir)
                    result['backup_name'] = backup_name

            # Atomic rename of temp to current
            temp_dir = self.baseline_dir / temp_name
            current_dir = self.baseline_dir / "current"

            # If current exists and wasn't backed up, remove it
            if current_dir.exists():
                shutil.rmtree(current_dir)

            temp_dir.rename(current_dir)

            result['success'] = True

        except Exception as e:
            result['error'] = str(e)

            # Cleanup temp baseline if it exists
            try:
                temp_dir = self.baseline_dir / temp_name
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except:
                pass

        return result

    def create_and_save_baseline(self, name: str = "current") -> Dict[str, any]:
        """
        Convenience method to create and save a baseline in one step

        Args:
            name: Baseline name

        Returns:
            Dictionary with results:
            {
                'success': bool,
                'baseline': baseline data if successful,
                'tree_hash': tree hash if successful,
                'error': error message if failed
            }
        """
        result = {
            'success': False,
            'baseline': None,
            'tree_hash': None,
            'error': None
        }

        try:
            baseline = self.create_baseline()
            if self.save_baseline(baseline, name):
                result['success'] = True
                result['baseline'] = baseline
                result['tree_hash'] = baseline.get('tree_hash')
            else:
                result['error'] = "Failed to save baseline"
        except Exception as e:
            result['error'] = str(e)

        return result

    def list_baselines(self) -> List[Dict[str, any]]:
        """
        List all available baselines

        Returns:
            List of baseline info dictionaries:
            [
                {
                    'name': baseline name,
                    'created': creation timestamp,
                    'tree_hash': tree hash,
                    'file_count': number of files,
                    'valid': integrity check result
                },
                ...
            ]
        """
        baselines = []

        if not self.baseline_dir.exists():
            return baselines

        for item in self.baseline_dir.iterdir():
            if item.is_dir():
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    try:
                        baseline = self.load_baseline(item.name)
                        if baseline:
                            verification = self.verify_baseline(baseline, item.name)
                            info = {
                                'name': item.name,
                                'created': baseline.get('created', 'unknown'),
                                'tree_hash': baseline.get('tree_hash', 'unknown'),
                                'file_count': len(baseline.get('files', {})),
                                'valid': verification.get('valid', False)
                            }
                            baselines.append(info)
                    except Exception as e:
                        print(f"Warning: Error reading baseline {item.name}: {e}")

        return sorted(baselines, key=lambda x: x['created'], reverse=True)

    def delete_baseline(self, name: str) -> bool:
        """
        Delete a baseline

        Args:
            name: Baseline name to delete

        Returns:
            True if deleted successfully
        """
        if name == "current":
            print("Warning: Cannot delete 'current' baseline. Use refresh instead.")
            return False

        baseline_dir = self.baseline_dir / name

        try:
            if baseline_dir.exists():
                shutil.rmtree(baseline_dir)
                return True
            return False
        except Exception as e:
            print(f"Error deleting baseline {name}: {e}")
            return False
