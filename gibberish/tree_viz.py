"""
Tree visualization with diff annotations and approval system for Gibberish.

This module provides a visual directory tree display showing sync operations,
conflict annotations, and an interactive approval system for both machines.
"""

from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import time
from datetime import datetime


class OperationType(Enum):
    """Types of sync operations"""
    MODIFY = "→"
    ADD = "+"
    DELETE = "-"
    CONFLICT = "⚠"


class Direction(Enum):
    """Sync direction"""
    A_TO_B = "A→B"
    B_TO_A = "B→A"
    BIDIRECTIONAL = "A↔B"


@dataclass
class TreeNode:
    """Represents a node in the directory tree"""
    path: Path
    name: str
    is_dir: bool
    operation: Optional[OperationType] = None
    size: int = 0
    diff_size: int = 0
    direction: Optional[Direction] = None
    conflict_info: Optional[str] = None
    annotation: Optional['OperationAnnotation'] = None
    children: List['TreeNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def add_child(self, node: 'TreeNode'):
        """Add a child node"""
        self.children.append(node)

    def sort_children(self):
        """Sort children: directories first, then by name"""
        self.children.sort(key=lambda n: (not n.is_dir, n.name))

    def create_annotation(self):
        """Create OperationAnnotation from node data"""
        if self.operation:
            self.annotation = OperationAnnotation(
                operation=self.operation,
                original_size=self.size,
                transfer_size=self.diff_size if self.diff_size > 0 else self.size,
                direction=self.direction,
                conflict_info=self.conflict_info
            )


class TreeGenerator:
    """Generates directory tree structures"""

    def __init__(self, root_dir: Path):
        """
        Initialize tree generator

        Args:
            root_dir: Root directory for tree generation
        """
        self.root_dir = Path(root_dir)

    def _try_system_tree(self, max_depth: Optional[int] = None) -> Optional[str]:
        """
        Try to use system 'tree' command

        Args:
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            Tree output string or None if command not available
        """
        try:
            cmd = ['tree', '-F', '--charset', 'utf-8']

            if max_depth is not None:
                cmd.extend(['-L', str(max_depth)])

            cmd.append(str(self.root_dir))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return result.stdout

            return None

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _build_tree_recursive(self, path: Path, ignore_patterns: Set[str],
                              max_depth: Optional[int] = None,
                              current_depth: int = 0) -> TreeNode:
        """
        Build tree structure recursively

        Args:
            path: Current path to process
            ignore_patterns: Set of patterns to ignore
            max_depth: Maximum depth to traverse
            current_depth: Current recursion depth

        Returns:
            TreeNode representing this path
        """
        node = TreeNode(
            path=path,
            name=path.name if path != self.root_dir else path.name or '.',
            is_dir=path.is_dir()
        )

        # Check depth limit
        if max_depth is not None and current_depth >= max_depth:
            return node

        # Process directory contents
        if path.is_dir():
            try:
                for item in sorted(path.iterdir()):
                    # Skip ignored patterns
                    if self._should_ignore(item, ignore_patterns):
                        continue

                    child = self._build_tree_recursive(
                        item,
                        ignore_patterns,
                        max_depth,
                        current_depth + 1
                    )
                    node.add_child(child)

                node.sort_children()

            except PermissionError:
                pass  # Skip directories we can't read
        else:
            # Get file size
            try:
                node.size = path.stat().st_size
            except OSError:
                node.size = 0

        return node

    def _should_ignore(self, path: Path, ignore_patterns: Set[str]) -> bool:
        """
        Check if path should be ignored

        Args:
            path: Path to check
            ignore_patterns: Set of patterns to ignore

        Returns:
            True if path should be ignored
        """
        name = path.name

        for pattern in ignore_patterns:
            if pattern == name or (pattern.endswith('*') and name.startswith(pattern[:-1])):
                return True

        return False

    def generate_tree(self, ignore_patterns: Optional[Set[str]] = None,
                     max_depth: Optional[int] = None,
                     use_system: bool = True) -> TreeNode:
        """
        Generate directory tree structure

        Args:
            ignore_patterns: Set of patterns to ignore
            max_depth: Maximum depth to traverse
            use_system: Try to use system 'tree' command first

        Returns:
            Root TreeNode of the generated tree
        """
        if ignore_patterns is None:
            ignore_patterns = {'.git', '.gibberish', '__pycache__', '.DS_Store'}

        # Build custom tree
        return self._build_tree_recursive(
            self.root_dir,
            ignore_patterns,
            max_depth
        )


class TreeVisualizer:
    """Visualizes directory trees with operation annotations"""

    def __init__(self):
        """Initialize tree visualizer"""
        self.indent_str = "│   "
        self.branch_str = "├── "
        self.last_branch_str = "└── "

    def _format_size(self, size_bytes: int) -> str:
        """
        Format byte size as human-readable string

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    def _render_node(self, node: TreeNode, prefix: str, is_last: bool,
                    show_sizes: bool = True, show_annotations: bool = True) -> List[str]:
        """
        Render a tree node and its children

        Args:
            node: TreeNode to render
            prefix: Current prefix string
            is_last: Whether this is the last child
            show_sizes: Whether to show file sizes
            show_annotations: Whether to show detailed annotations

        Returns:
            List of output lines
        """
        lines = []

        # Build connector
        connector = self.last_branch_str if is_last else self.branch_str

        # Build line
        line = prefix + connector

        # Add operation indicator with color coding (using ANSI codes)
        if node.operation:
            op_symbol = node.operation.value

            # Color coding for operations
            if node.operation == OperationType.ADD:
                op_symbol = f"\033[92m{op_symbol}\033[0m"  # Green
            elif node.operation == OperationType.MODIFY:
                op_symbol = f"\033[93m{op_symbol}\033[0m"  # Yellow
            elif node.operation == OperationType.DELETE:
                op_symbol = f"\033[91m{op_symbol}\033[0m"  # Red
            elif node.operation == OperationType.CONFLICT:
                op_symbol = f"\033[91m{op_symbol}\033[0m"  # Red with bold

            line += f"{op_symbol} "

        # Add direction indicator if specified
        if node.direction:
            line += f"[{node.direction.value}] "

        # Add name
        line += node.name
        if node.is_dir:
            line += "/"

        # Add size info
        if show_sizes and not node.is_dir:
            if node.operation == OperationType.MODIFY and node.diff_size > 0:
                # Show both original and diff size for modifications
                orig_fmt = self._format_size(node.size)
                diff_fmt = self._format_size(node.diff_size)
                reduction = ((node.size - node.diff_size) / node.size * 100) if node.size > 0 else 0
                line += f" ({orig_fmt} → {diff_fmt}, \033[92m{reduction:.0f}% saved\033[0m)"
            elif node.operation == OperationType.ADD and node.size > 0:
                line += f" (\033[92m+{self._format_size(node.size)}\033[0m)"
            elif node.size > 0:
                line += f" ({self._format_size(node.size)})"

        # Add conflict warning
        if node.conflict_info:
            line += f" \033[91m⚠ {node.conflict_info}\033[0m"

        lines.append(line)

        # Process children
        if node.children:
            # Update prefix for children
            new_prefix = prefix + ("    " if is_last else self.indent_str)

            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                child_lines = self._render_node(child, new_prefix, is_last_child, show_sizes, show_annotations)
                lines.extend(child_lines)

        return lines

    def render_tree(self, root: TreeNode, show_sizes: bool = True,
                   show_annotations: bool = True) -> str:
        """
        Render tree structure as formatted string

        Args:
            root: Root TreeNode to render
            show_sizes: Whether to show file sizes
            show_annotations: Whether to show detailed annotations

        Returns:
            Formatted tree string
        """
        lines = [root.name + ("/", "")[not root.is_dir]]

        for i, child in enumerate(root.children):
            is_last = i == len(root.children) - 1
            child_lines = self._render_node(child, "", is_last, show_sizes, show_annotations)
            lines.extend(child_lines)

        return "\n".join(lines)

    def render_legend(self) -> str:
        """
        Render legend for operation symbols

        Returns:
            Formatted legend string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("OPERATION LEGEND")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  \033[92m{OperationType.ADD.value}\033[0m  ADD     - New file will be created")
        lines.append(f"  \033[93m{OperationType.MODIFY.value}\033[0m  MODIFY  - Existing file will be updated (diff transfer)")
        lines.append(f"  \033[91m{OperationType.DELETE.value}\033[0m  DELETE  - File will be removed")
        lines.append(f"  \033[91m{OperationType.CONFLICT.value}\033[0m  CONFLICT - File has conflicting changes")
        lines.append("")
        lines.append("DIRECTION INDICATORS:")
        lines.append(f"  [A→B]  - Transfer from Machine A to Machine B")
        lines.append(f"  [B→A]  - Transfer from Machine B to Machine A")
        lines.append(f"  [A↔B]  - Bidirectional sync required")
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


class AnnotatedTreeBuilder:
    """Builds annotated trees from sync changes"""

    def __init__(self, root_dir: Path):
        """
        Initialize annotated tree builder

        Args:
            root_dir: Root directory
        """
        self.root_dir = Path(root_dir)
        self.tree_gen = TreeGenerator(root_dir)

    def build_from_changes(self, changes: List[Any],
                          conflicts: Optional[List[Any]] = None,
                          ignore_patterns: Optional[Set[str]] = None) -> TreeNode:
        """
        Build annotated tree from file changes

        Args:
            changes: List of FileChange objects
            conflicts: Optional list of conflict objects
            ignore_patterns: Patterns to ignore

        Returns:
            Annotated TreeNode
        """
        # Generate base tree
        tree = self.tree_gen.generate_tree(ignore_patterns)

        # Build path lookup
        path_map = self._build_path_map(tree)

        # Annotate changes
        for change in changes:
            rel_path = change.path

            if rel_path in path_map:
                node = path_map[rel_path]

                # Map ChangeType to OperationType
                if hasattr(change, 'change_type'):
                    from gibberish.sync import ChangeType

                    if change.change_type == ChangeType.ADD:
                        node.operation = OperationType.ADD
                    elif change.change_type == ChangeType.MODIFY:
                        node.operation = OperationType.MODIFY
                        node.diff_size = change.diff_size
                    elif change.change_type == ChangeType.DELETE:
                        node.operation = OperationType.DELETE

        # Annotate conflicts
        if conflicts:
            for conflict in conflicts:
                if hasattr(conflict, 'path'):
                    rel_path = conflict.path
                    if rel_path in path_map:
                        path_map[rel_path].operation = OperationType.CONFLICT

        return tree

    def _build_path_map(self, node: TreeNode, path_map: Optional[Dict] = None) -> Dict[Path, TreeNode]:
        """
        Build mapping of paths to nodes

        Args:
            node: Current node
            path_map: Existing path map

        Returns:
            Dictionary mapping paths to nodes
        """
        if path_map is None:
            path_map = {}

        # Add current node
        rel_path = node.path.relative_to(self.root_dir) if node.path != self.root_dir else Path('.')
        path_map[rel_path] = node

        # Process children
        for child in node.children:
            self._build_path_map(child, path_map)

        return path_map


@dataclass
class OperationAnnotation:
    """Detailed annotation for an operation"""
    operation: OperationType
    original_size: int
    transfer_size: int
    direction: Optional[Direction] = None
    conflict_info: Optional[str] = None

    def format_annotation(self) -> str:
        """
        Format annotation for display

        Returns:
            Formatted annotation string
        """
        parts = [self.operation.value]

        # Add direction if specified
        if self.direction:
            parts.append(f"[{self.direction.value}]")

        # Add size info
        if self.operation == OperationType.MODIFY:
            orig_kb = self.original_size / 1024
            trans_kb = self.transfer_size / 1024
            reduction = ((self.original_size - self.transfer_size) / self.original_size * 100) if self.original_size > 0 else 0
            parts.append(f"({orig_kb:.1f}KB → {trans_kb:.1f}KB, {reduction:.0f}% saved)")
        elif self.operation == OperationType.ADD:
            kb = self.transfer_size / 1024
            parts.append(f"({kb:.1f}KB)")
        elif self.operation == OperationType.DELETE:
            parts.append("(will be removed)")

        # Add conflict info
        if self.conflict_info:
            parts.append(f"CONFLICT: {self.conflict_info}")

        return " ".join(parts)


class SyncSummary:
    """Summary of sync operations"""

    def __init__(self, direction: Direction):
        """
        Initialize sync summary

        Args:
            direction: Sync direction
        """
        self.direction = direction
        self.add_count = 0
        self.modify_count = 0
        self.delete_count = 0
        self.conflict_count = 0
        self.total_size = 0
        self.transfer_size = 0
        self.estimated_time = 0.0
        self.annotations: List[OperationAnnotation] = []

    def add_change(self, operation: OperationType, size: int = 0, transfer_size: int = 0):
        """
        Add a change to the summary

        Args:
            operation: Type of operation
            size: Original file size
            transfer_size: Size to transfer
        """
        if operation == OperationType.ADD:
            self.add_count += 1
        elif operation == OperationType.MODIFY:
            self.modify_count += 1
        elif operation == OperationType.DELETE:
            self.delete_count += 1
        elif operation == OperationType.CONFLICT:
            self.conflict_count += 1

        self.total_size += size
        self.transfer_size += transfer_size

    def calculate_estimated_time(self, bandwidth_bps: float = 300.0):
        """
        Calculate estimated transmission time

        Args:
            bandwidth_bps: Bandwidth in bits per second
        """
        if bandwidth_bps > 0:
            # Convert bytes to bits and calculate time
            self.estimated_time = (self.transfer_size * 8) / bandwidth_bps

    def get_total_operations(self) -> int:
        """Get total number of operations"""
        return self.add_count + self.modify_count + self.delete_count

    def format_summary(self) -> str:
        """
        Format summary as string

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SYNC SUMMARY")
        lines.append("=" * 70)
        lines.append("")

        # Direction
        lines.append(f"Direction: {self.direction.value}")
        lines.append("")

        # Operations
        lines.append("Operations:")
        lines.append(f"  {OperationType.ADD.value} ADD:      {self.add_count}")
        lines.append(f"  {OperationType.MODIFY.value} MODIFY:  {self.modify_count}")
        lines.append(f"  {OperationType.DELETE.value} DELETE:  {self.delete_count}")
        if self.conflict_count > 0:
            lines.append(f"  {OperationType.CONFLICT.value} CONFLICT: {self.conflict_count}")
        lines.append(f"  Total:     {self.get_total_operations()}")
        lines.append("")

        # Data transfer
        total_mb = self.total_size / (1024 * 1024)
        transfer_mb = self.transfer_size / (1024 * 1024)
        reduction = ((self.total_size - self.transfer_size) / self.total_size * 100) if self.total_size > 0 else 0

        lines.append("Data Transfer:")
        lines.append(f"  Original Size:  {total_mb:.2f} MB")
        lines.append(f"  Transfer Size:  {transfer_mb:.2f} MB")
        lines.append(f"  Reduction:      {reduction:.1f}%")
        lines.append("")

        # Estimated time
        if self.estimated_time > 0:
            mins = int(self.estimated_time // 60)
            secs = int(self.estimated_time % 60)
            lines.append(f"Estimated Time: {mins}m {secs}s")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


class ApprovalResponse(Enum):
    """User approval responses"""
    YES = "y"
    NO = "n"
    MODIFY = "m"
    RESOLVE = "r"
    ABORT = "a"


class ApprovalPrompt:
    """Interactive approval prompt system for both machines"""

    def __init__(self, machine_name: str = "Machine"):
        """
        Initialize approval prompt

        Args:
            machine_name: Name/identifier of this machine
        """
        self.machine_name = machine_name

    def _display_prompt_options(self, has_conflicts: bool = False):
        """
        Display available prompt options

        Args:
            has_conflicts: Whether conflicts exist
        """
        print("\nOptions:")
        print("  [y]es     - Approve and proceed with sync")
        print("  [n]o      - Reject and abort sync")
        print("  [m]odify  - Modify sync selections")

        if has_conflicts:
            print("  [r]esolve - Resolve conflicts interactively")

        print("  [a]bort   - Abort immediately")
        print("")

    def prompt_approval(self, summary: SyncSummary, tree_display: str,
                       has_conflicts: bool = False,
                       timeout: Optional[float] = None) -> Tuple[ApprovalResponse, Optional[str]]:
        """
        Prompt user for approval

        Args:
            summary: Sync summary to display
            tree_display: Tree visualization string
            has_conflicts: Whether conflicts exist
            timeout: Optional timeout in seconds (None for no timeout)

        Returns:
            Tuple of (response, optional_message)
        """
        print("\n" + "=" * 70)
        print(f"SYNC APPROVAL REQUIRED - {self.machine_name}")
        print("=" * 70)
        print("")

        # Display summary
        print(summary.format_summary())

        # Display tree
        print("\nDirectory Tree Preview:")
        print(tree_display)

        # Display legend
        visualizer = TreeVisualizer()
        print("\n" + visualizer.render_legend())

        # Display conflicts warning if any
        if has_conflicts:
            print("\n" + "\033[91m" + "!" * 70 + "\033[0m")
            print("\033[91mWARNING: Conflicts detected! Review carefully before proceeding.\033[0m")
            print("\033[91m" + "!" * 70 + "\033[0m" + "\n")

        # Display timeout warning if applicable
        if timeout:
            timeout_mins = int(timeout // 60)
            print(f"\n\033[93mNote: This prompt will auto-abort in {timeout_mins} minutes if no response.\033[0m\n")

        # Show options
        self._display_prompt_options(has_conflicts)

        # Get user input with timeout support
        start_time = time.time()

        while True:
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                print("\n\033[91mTimeout reached. Auto-aborting sync.\033[0m")
                return ApprovalResponse.ABORT, "Timeout reached"

            # Prompt for input
            try:
                if timeout:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    remaining_mins = int(remaining // 60)
                    remaining_secs = int(remaining % 60)
                    prompt_msg = f"Your choice [{remaining_mins}:{remaining_secs:02d} remaining]: "
                else:
                    prompt_msg = "Your choice: "

                response = input(prompt_msg).strip().lower()

                # Parse response
                if response == 'y' or response == 'yes':
                    return ApprovalResponse.YES, None
                elif response == 'n' or response == 'no':
                    reason = input("Reason for rejection (optional): ").strip()
                    return ApprovalResponse.NO, reason or "User rejected"
                elif response == 'm' or response == 'modify':
                    return ApprovalResponse.MODIFY, None
                elif response == 'r' or response == 'resolve':
                    if has_conflicts:
                        return ApprovalResponse.RESOLVE, None
                    else:
                        print("No conflicts to resolve. Please choose another option.")
                elif response == 'a' or response == 'abort':
                    return ApprovalResponse.ABORT, "User aborted"
                else:
                    print(f"Invalid choice: '{response}'. Please try again.")

            except (EOFError, KeyboardInterrupt):
                print("\n\n\033[91mInterrupted. Aborting sync.\033[0m")
                return ApprovalResponse.ABORT, "User interrupted"

    def send_approval_to_peer(self, response: ApprovalResponse,
                             message: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare approval response for transmission to peer

        Args:
            response: Approval response
            message: Optional message to include

        Returns:
            Dictionary with approval data
        """
        return {
            'machine': self.machine_name,
            'response': response.value,
            'message': message,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

    def wait_for_peer_approval(self, timeout: float = 300.0) -> Tuple[bool, Optional[str]]:
        """
        Wait for peer machine's approval

        Args:
            timeout: Timeout in seconds (default 5 minutes)

        Returns:
            Tuple of (approved, message)
        """
        print("\n" + "=" * 70)
        print(f"WAITING FOR PEER APPROVAL - {self.machine_name}")
        print("=" * 70)
        print("\nWaiting for the other machine to approve sync...")
        print(f"Timeout: {int(timeout // 60)} minutes\n")

        start_time = time.time()

        # This is a placeholder - actual implementation would integrate with protocol
        # For now, simulate waiting
        while (time.time() - start_time) < timeout:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed

            if remaining > 0:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                print(f"\rWaiting... [{mins}:{secs:02d} remaining]", end="", flush=True)
                time.sleep(1)
            else:
                break

        print("\n")

        # Placeholder return
        # Real implementation would check actual peer response
        return False, "Timeout waiting for peer approval"

    def display_peer_response(self, peer_machine: str, response: ApprovalResponse,
                             message: Optional[str] = None):
        """
        Display peer machine's approval response

        Args:
            peer_machine: Name of peer machine
            response: Peer's approval response
            message: Optional message from peer
        """
        print("\n" + "=" * 70)
        print(f"PEER RESPONSE - {peer_machine}")
        print("=" * 70)

        if response == ApprovalResponse.YES:
            print("\n\033[92m✓ Peer approved the sync\033[0m")
        elif response == ApprovalResponse.NO:
            print("\n\033[91m✗ Peer rejected the sync\033[0m")
            if message:
                print(f"Reason: {message}")
        elif response == ApprovalResponse.ABORT:
            print("\n\033[91m✗ Peer aborted the sync\033[0m")
            if message:
                print(f"Reason: {message}")
        elif response == ApprovalResponse.MODIFY:
            print("\n\033[93m⚠ Peer requested modifications\033[0m")
        elif response == ApprovalResponse.RESOLVE:
            print("\n\033[93m⚠ Peer is resolving conflicts\033[0m")

        print("\n" + "=" * 70 + "\n")


def display_sync_preview(changes: List[Any], conflicts: Optional[List[Any]] = None,
                        root_dir: Optional[Path] = None, direction: Direction = Direction.A_TO_B,
                        bandwidth_bps: float = 300.0):
    """
    Display a complete sync preview with tree and summary

    Args:
        changes: List of FileChange objects
        conflicts: Optional list of conflicts
        root_dir: Root directory (defaults to current)
        direction: Sync direction
        bandwidth_bps: Bandwidth for time estimation
    """
    if root_dir is None:
        root_dir = Path.cwd()

    # Build annotated tree
    builder = AnnotatedTreeBuilder(root_dir)
    tree = builder.build_from_changes(changes, conflicts)

    # Render tree
    visualizer = TreeVisualizer()
    tree_output = visualizer.render_tree(tree)

    # Build summary
    summary = SyncSummary(direction)
    for change in changes:
        op_type = None
        if hasattr(change, 'change_type'):
            from gibberish.sync import ChangeType

            if change.change_type == ChangeType.ADD:
                op_type = OperationType.ADD
            elif change.change_type == ChangeType.MODIFY:
                op_type = OperationType.MODIFY
            elif change.change_type == ChangeType.DELETE:
                op_type = OperationType.DELETE

        if op_type:
            summary.add_change(op_type, change.size, change.diff_size or change.size)

    if conflicts:
        for conflict in conflicts:
            summary.add_change(OperationType.CONFLICT, 0, 0)

    summary.calculate_estimated_time(bandwidth_bps)

    # Display
    print("\n" + summary.format_summary())
    print("\nDirectory Tree:")
    print(tree_output)
    print("")


@dataclass
class ConflictInfo:
    """Information about a file conflict"""
    path: Path
    machine_a_version: Optional[str] = None
    machine_b_version: Optional[str] = None
    machine_a_size: int = 0
    machine_b_size: int = 0
    machine_a_mtime: Optional[float] = None
    machine_b_mtime: Optional[float] = None
    conflict_type: str = "modify"  # modify, delete-modify, add-add


class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    KEEP_A = "a"
    KEEP_B = "b"
    MERGE = "m"
    SKIP = "s"
    MANUAL = "manual"


class ConflictResolver:
    """Interactive conflict resolution UI"""

    def __init__(self, machine_a_name: str = "Machine A", machine_b_name: str = "Machine B"):
        """
        Initialize conflict resolver

        Args:
            machine_a_name: Name of machine A
            machine_b_name: Name of machine B
        """
        self.machine_a_name = machine_a_name
        self.machine_b_name = machine_b_name

    def _display_side_by_side(self, conflict: ConflictInfo, terminal_width: int = 140):
        """
        Display side-by-side comparison of conflicting versions

        Args:
            conflict: Conflict information
            terminal_width: Terminal width for formatting
        """
        col_width = (terminal_width - 3) // 2

        print("\n" + "=" * terminal_width)
        print("SIDE-BY-SIDE COMPARISON")
        print("=" * terminal_width)
        print("")

        # Header
        header_a = f"{self.machine_a_name} Version".center(col_width)
        header_b = f"{self.machine_b_name} Version".center(col_width)
        print(f"{header_a} │ {header_b}")
        print("─" * col_width + " │ " + "─" * col_width)

        # File path
        print(f"\nFile: {conflict.path}\n")

        # Metadata comparison
        meta_lines = []

        # Size
        if conflict.machine_a_size or conflict.machine_b_size:
            size_a = self._format_size(conflict.machine_a_size) if conflict.machine_a_size else "N/A"
            size_b = self._format_size(conflict.machine_b_size) if conflict.machine_b_size else "N/A"
            meta_lines.append((f"Size: {size_a}", f"Size: {size_b}"))

        # Modified time
        if conflict.machine_a_mtime or conflict.machine_b_mtime:
            time_a = self._format_time(conflict.machine_a_mtime) if conflict.machine_a_mtime else "N/A"
            time_b = self._format_time(conflict.machine_b_mtime) if conflict.machine_b_mtime else "N/A"
            meta_lines.append((f"Modified: {time_a}", f"Modified: {time_b}"))

        # Display metadata
        for line_a, line_b in meta_lines:
            padded_a = line_a.ljust(col_width)
            padded_b = line_b.ljust(col_width)
            print(f"{padded_a} │ {padded_b}")

        print("")

        # Content preview (if text files)
        if conflict.machine_a_version or conflict.machine_b_version:
            print("─" * col_width + " │ " + "─" * col_width)
            print("Content Preview:".center(terminal_width))
            print("─" * col_width + " │ " + "─" * col_width)

            # Get content lines
            lines_a = conflict.machine_a_version.split('\n') if conflict.machine_a_version else ["(deleted)"]
            lines_b = conflict.machine_b_version.split('\n') if conflict.machine_b_version else ["(deleted)"]

            # Display up to 10 lines
            max_lines = max(len(lines_a), len(lines_b))
            display_lines = min(max_lines, 10)

            for i in range(display_lines):
                line_a = lines_a[i] if i < len(lines_a) else ""
                line_b = lines_b[i] if i < len(lines_b) else ""

                # Truncate if too long
                if len(line_a) > col_width - 2:
                    line_a = line_a[:col_width - 5] + "..."
                if len(line_b) > col_width - 2:
                    line_b = line_b[:col_width - 5] + "..."

                # Highlight differences
                if line_a != line_b:
                    line_a_display = f"\033[93m{line_a}\033[0m"
                    line_b_display = f"\033[93m{line_b}\033[0m"
                else:
                    line_a_display = line_a
                    line_b_display = line_b

                padded_a = line_a_display.ljust(col_width + 9)  # Account for ANSI codes
                padded_b = line_b_display.ljust(col_width + 9)
                print(f"{padded_a} │ {padded_b}")

            if max_lines > display_lines:
                more_lines = max_lines - display_lines
                more_msg = f"... ({more_lines} more lines)".center(col_width)
                print(f"{more_msg} │ {more_msg}")

        print("\n" + "=" * terminal_width + "\n")

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    def _format_time(self, timestamp: float) -> str:
        """Format timestamp"""
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _display_resolution_options(self, conflict: ConflictInfo):
        """
        Display resolution options for a conflict

        Args:
            conflict: Conflict information
        """
        print("Resolution Options:")
        print(f"  [a] Keep {self.machine_a_name} version")
        print(f"  [b] Keep {self.machine_b_name} version")

        if conflict.conflict_type == "modify":
            print("  [m] Attempt automatic merge")

        print("  [s] Skip this file (resolve later)")
        print("  [manual] Manual resolution (open editor)")
        print("")

    def resolve_conflict(self, conflict: ConflictInfo) -> Tuple[ConflictResolution, Optional[str]]:
        """
        Interactively resolve a single conflict

        Args:
            conflict: Conflict to resolve

        Returns:
            Tuple of (resolution, optional_merged_content)
        """
        # Display comparison
        self._display_side_by_side(conflict)

        # Display options
        self._display_resolution_options(conflict)

        # Get user choice
        while True:
            choice = input("Your choice: ").strip().lower()

            if choice == 'a':
                print(f"\n\033[92m✓ Keeping {self.machine_a_name} version\033[0m\n")
                return ConflictResolution.KEEP_A, None
            elif choice == 'b':
                print(f"\n\033[92m✓ Keeping {self.machine_b_name} version\033[0m\n")
                return ConflictResolution.KEEP_B, None
            elif choice == 'm' and conflict.conflict_type == "modify":
                print("\n\033[93mAttempting automatic merge...\033[0m")
                merged = self._attempt_merge(conflict)
                if merged:
                    print("\033[92m✓ Merge successful\033[0m\n")
                    return ConflictResolution.MERGE, merged
                else:
                    print("\033[91m✗ Automatic merge failed. Please choose another option.\033[0m\n")
            elif choice == 's':
                print("\n\033[93m⊘ Skipping file\033[0m\n")
                return ConflictResolution.SKIP, None
            elif choice == 'manual':
                print("\n\033[93m⚠ Manual resolution not yet implemented\033[0m\n")
                print("Please choose another option.")
            else:
                print(f"Invalid choice: '{choice}'. Please try again.")

    def _attempt_merge(self, conflict: ConflictInfo) -> Optional[str]:
        """
        Attempt automatic three-way merge

        Args:
            conflict: Conflict information

        Returns:
            Merged content if successful, None otherwise
        """
        # This is a simplified merge - real implementation would use
        # difflib or a proper merge algorithm
        if not conflict.machine_a_version or not conflict.machine_b_version:
            return None

        # Simple line-based merge
        lines_a = conflict.machine_a_version.split('\n')
        lines_b = conflict.machine_b_version.split('\n')

        # If files are identical, no conflict
        if lines_a == lines_b:
            return conflict.machine_a_version

        # Very basic merge: if no overlapping changes, combine
        # Real implementation would be more sophisticated
        merged_lines = []
        max_len = max(len(lines_a), len(lines_b))

        for i in range(max_len):
            line_a = lines_a[i] if i < len(lines_a) else None
            line_b = lines_b[i] if i < len(lines_b) else None

            if line_a == line_b:
                merged_lines.append(line_a or line_b)
            elif line_a and line_b:
                # Conflict on this line - cannot auto-merge
                return None
            else:
                # One side added a line
                merged_lines.append(line_a or line_b)

        return '\n'.join(merged_lines)

    def resolve_all_conflicts(self, conflicts: List[ConflictInfo]) -> Dict[Path, Tuple[ConflictResolution, Optional[str]]]:
        """
        Resolve all conflicts interactively

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            Dictionary mapping paths to (resolution, merged_content) tuples
        """
        resolutions = {}

        print("\n" + "=" * 70)
        print(f"CONFLICT RESOLUTION - {len(conflicts)} conflicts to resolve")
        print("=" * 70)
        print("")

        for i, conflict in enumerate(conflicts, 1):
            print(f"\n{'=' * 70}")
            print(f"Conflict {i} of {len(conflicts)}")
            print(f"{'=' * 70}\n")

            resolution, content = self.resolve_conflict(conflict)
            resolutions[conflict.path] = (resolution, content)

        print("\n" + "=" * 70)
        print("CONFLICT RESOLUTION COMPLETE")
        print("=" * 70)
        print("")

        # Summary
        summary = {
            ConflictResolution.KEEP_A: 0,
            ConflictResolution.KEEP_B: 0,
            ConflictResolution.MERGE: 0,
            ConflictResolution.SKIP: 0
        }

        for resolution, _ in resolutions.values():
            if resolution in summary:
                summary[resolution] += 1

        print("Summary:")
        print(f"  Keep {self.machine_a_name}: {summary[ConflictResolution.KEEP_A]}")
        print(f"  Keep {self.machine_b_name}: {summary[ConflictResolution.KEEP_B]}")
        print(f"  Merged: {summary[ConflictResolution.MERGE]}")
        print(f"  Skipped: {summary[ConflictResolution.SKIP]}")
        print("")

        return resolutions


class SyncCoordinator:
    """Coordinates synchronization between machines with timeout handling"""

    DEFAULT_TIMEOUT = 300.0  # 5 minutes

    def __init__(self, machine_name: str = "Machine", peer_name: str = "Peer"):
        """
        Initialize sync coordinator

        Args:
            machine_name: Name of this machine
            peer_name: Name of peer machine
        """
        self.machine_name = machine_name
        self.peer_name = peer_name
        self.start_time: Optional[float] = None
        self.approval_deadline: Optional[float] = None

    def start_sync_session(self, timeout: float = DEFAULT_TIMEOUT):
        """
        Start a sync session with timeout

        Args:
            timeout: Session timeout in seconds
        """
        self.start_time = time.time()
        self.approval_deadline = self.start_time + timeout

        print("\n" + "=" * 70)
        print(f"SYNC SESSION STARTED - {self.machine_name}")
        print("=" * 70)
        print(f"\nSession timeout: {int(timeout // 60)} minutes")
        print(f"Started at: {datetime.fromtimestamp(self.start_time).strftime('%H:%M:%S')}")
        print(f"Deadline: {datetime.fromtimestamp(self.approval_deadline).strftime('%H:%M:%S')}")
        print("")

    def check_timeout(self) -> bool:
        """
        Check if session has timed out

        Returns:
            True if timed out
        """
        if self.approval_deadline is None:
            return False

        return time.time() >= self.approval_deadline

    def get_remaining_time(self) -> float:
        """
        Get remaining time before timeout

        Returns:
            Remaining seconds (0 if expired)
        """
        if self.approval_deadline is None:
            return 0.0

        remaining = self.approval_deadline - time.time()
        return max(0.0, remaining)

    def format_remaining_time(self) -> str:
        """
        Format remaining time as string

        Returns:
            Formatted time string
        """
        remaining = self.get_remaining_time()
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        return f"{mins}:{secs:02d}"

    def sync_with_peer(self, local_approval: ApprovalResponse,
                      protocol_handler=None) -> Tuple[bool, Optional[ApprovalResponse]]:
        """
        Synchronize approval with peer machine

        This method coordinates approval between machines, ensuring both
        machines agree before proceeding with sync.

        Args:
            local_approval: This machine's approval response
            protocol_handler: Optional ProtocolHandler for communication

        Returns:
            Tuple of (sync_approved, peer_response)
        """
        print("\n" + "=" * 70)
        print("SYNCHRONIZING WITH PEER")
        print("=" * 70)

        # Check if we've already timed out
        if self.check_timeout():
            print("\n\033[91m✗ Session timed out\033[0m\n")
            return False, None

        # Send our approval to peer
        print(f"\nSending approval to {self.peer_name}...")
        print(f"Our response: {local_approval.value}")

        # In a real implementation, this would use the protocol_handler
        # to transmit the approval to the peer machine
        # For now, simulate waiting

        # Wait for peer approval with remaining timeout
        remaining = self.get_remaining_time()
        print(f"\nWaiting for {self.peer_name}'s approval...")
        print(f"Timeout: {self.format_remaining_time()}\n")

        # Simulate waiting (real implementation would use protocol)
        start_wait = time.time()
        while (time.time() - start_wait) < min(remaining, 30.0):
            if self.check_timeout():
                print("\n\033[91m✗ Timeout waiting for peer approval\033[0m")
                return False, None

            # Display countdown
            elapsed = time.time() - start_wait
            remaining_wait = min(remaining, 30.0) - elapsed
            if remaining_wait > 0:
                print(f"\rWaiting... [{self.format_remaining_time()} remaining]", end="", flush=True)
                time.sleep(0.5)

        print("\n")

        # Placeholder: Real implementation would check actual peer response
        # For now, return timeout
        print("\033[91m✗ Peer approval not received (timeout)\033[0m")
        print("\nNote: Peer synchronization requires protocol integration\n")

        return False, None

    def handle_approval_mismatch(self, local: ApprovalResponse, peer: ApprovalResponse) -> bool:
        """
        Handle case where machines have different approvals

        Args:
            local: Local machine's approval
            peer: Peer machine's approval

        Returns:
            True if sync can proceed
        """
        print("\n" + "=" * 70)
        print("APPROVAL MISMATCH")
        print("=" * 70)
        print("")

        print(f"{self.machine_name}: {local.value}")
        print(f"{self.peer_name}: {peer.value}")
        print("")

        # Both must approve for sync to proceed
        if local == ApprovalResponse.YES and peer == ApprovalResponse.YES:
            print("\033[92m✓ Both machines approved - proceeding with sync\033[0m\n")
            return True

        # If either rejected or aborted, sync cannot proceed
        if local in [ApprovalResponse.NO, ApprovalResponse.ABORT] or \
           peer in [ApprovalResponse.NO, ApprovalResponse.ABORT]:
            print("\033[91m✗ Sync rejected or aborted\033[0m\n")
            return False

        # If one wants to modify or resolve, need to renegotiate
        print("\033[93m⚠ Approval mismatch - sync cannot proceed\033[0m")
        print("Both machines must approve to continue.\n")

        return False

    def abort_sync(self, reason: str = "User requested"):
        """
        Abort sync session

        Args:
            reason: Reason for abort
        """
        print("\n" + "=" * 70)
        print("SYNC ABORTED")
        print("=" * 70)
        print(f"\nReason: {reason}")

        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"Session duration: {int(elapsed)} seconds")

        print("")

    def complete_sync(self):
        """Mark sync session as complete"""
        print("\n" + "=" * 70)
        print("SYNC SESSION COMPLETE")
        print("=" * 70)

        if self.start_time:
            elapsed = time.time() - self.start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"\nSession duration: {mins}m {secs}s")

        print("")


def prompt_sync_approval(changes: List[Any], conflicts: Optional[List[Any]] = None,
                        root_dir: Optional[Path] = None, direction: Direction = Direction.A_TO_B,
                        machine_name: str = "Machine", bandwidth_bps: float = 300.0,
                        timeout: Optional[float] = 300.0) -> Tuple[ApprovalResponse, Optional[str]]:
    """
    Display sync preview and prompt for approval

    Args:
        changes: List of FileChange objects
        conflicts: Optional list of conflicts
        root_dir: Root directory (defaults to current)
        direction: Sync direction
        machine_name: Name of this machine
        bandwidth_bps: Bandwidth for time estimation
        timeout: Approval timeout in seconds (default 5 minutes, None for no timeout)

    Returns:
        Tuple of (response, message)
    """
    if root_dir is None:
        root_dir = Path.cwd()

    # Build annotated tree
    builder = AnnotatedTreeBuilder(root_dir)
    tree = builder.build_from_changes(changes, conflicts)

    # Render tree
    visualizer = TreeVisualizer()
    tree_output = visualizer.render_tree(tree)

    # Build summary
    summary = SyncSummary(direction)
    for change in changes:
        op_type = None
        if hasattr(change, 'change_type'):
            from gibberish.sync import ChangeType

            if change.change_type == ChangeType.ADD:
                op_type = OperationType.ADD
            elif change.change_type == ChangeType.MODIFY:
                op_type = OperationType.MODIFY
            elif change.change_type == ChangeType.DELETE:
                op_type = OperationType.DELETE

        if op_type:
            summary.add_change(op_type, change.size, change.diff_size or change.size)

    if conflicts:
        for conflict in conflicts:
            summary.add_change(OperationType.CONFLICT, 0, 0)

    summary.calculate_estimated_time(bandwidth_bps)

    # Create approval prompt
    prompt = ApprovalPrompt(machine_name)

    # Get approval
    has_conflicts = conflicts is not None and len(conflicts) > 0
    return prompt.prompt_approval(summary, tree_output, has_conflicts, timeout)


def coordinated_sync_approval(changes: List[Any], conflicts: Optional[List[Any]] = None,
                              root_dir: Optional[Path] = None, direction: Direction = Direction.A_TO_B,
                              machine_name: str = "Machine A", peer_name: str = "Machine B",
                              bandwidth_bps: float = 300.0, timeout: float = 300.0,
                              protocol_handler=None) -> Tuple[bool, Optional[str]]:
    """
    Coordinated approval process for both machines with timeout handling

    This is the main entry point for synchronized approval between machines.

    Args:
        changes: List of FileChange objects
        conflicts: Optional list of conflicts
        root_dir: Root directory (defaults to current)
        direction: Sync direction
        machine_name: Name of this machine
        peer_name: Name of peer machine
        bandwidth_bps: Bandwidth for time estimation
        timeout: Total session timeout in seconds (default 5 minutes)
        protocol_handler: Optional ProtocolHandler for peer communication

    Returns:
        Tuple of (approved, optional_message)
    """
    # Create coordinator
    coordinator = SyncCoordinator(machine_name, peer_name)
    coordinator.start_sync_session(timeout)

    try:
        # Get local approval
        response, message = prompt_sync_approval(
            changes, conflicts, root_dir, direction,
            machine_name, bandwidth_bps,
            coordinator.get_remaining_time()
        )

        # Check if user aborted or timeout
        if response == ApprovalResponse.ABORT:
            coordinator.abort_sync(message or "User aborted")
            return False, message

        if response == ApprovalResponse.NO:
            coordinator.abort_sync(message or "User rejected")
            return False, message

        # If user wants to resolve conflicts, handle that first
        if response == ApprovalResponse.RESOLVE:
            print("\n\033[93mConflict resolution selected\033[0m")
            print("Note: Conflict resolution integration pending\n")
            coordinator.abort_sync("Conflict resolution required")
            return False, "Conflicts need resolution"

        # Synchronize with peer
        if response == ApprovalResponse.YES:
            sync_ok, peer_response = coordinator.sync_with_peer(response, protocol_handler)

            if sync_ok and peer_response == ApprovalResponse.YES:
                coordinator.complete_sync()
                return True, "Both machines approved"
            else:
                coordinator.abort_sync("Peer approval failed or timed out")
                return False, "Peer approval not received"

        # Other responses (MODIFY, etc.)
        coordinator.abort_sync(f"Non-approval response: {response.value}")
        return False, f"Approval not granted: {response.value}"

    except Exception as e:
        coordinator.abort_sync(f"Error: {str(e)}")
        return False, f"Error during approval: {str(e)}"
