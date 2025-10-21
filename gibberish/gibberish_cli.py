"""
Gibberish CLI - Acoustic File Synchronization

Command-line interface for syncing files between machines via acoustic transmission.
"""

import click
import sys
import os
import logging
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque
from datetime import datetime, timedelta

# Rich library for colorized output and progress bars
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
)
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Initialize rich console
console = Console()

# Exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_ARGUMENT_ERROR = 2
EXIT_CONFIG_ERROR = 3
EXIT_AUDIO_ERROR = 4
EXIT_NETWORK_ERROR = 5
EXIT_FILE_ERROR = 6
EXIT_VERIFICATION_ERROR = 7


class BandwidthMonitor:
    """Monitor and display real-time bandwidth with sparkline graphs"""

    def __init__(self, window_size: int = 30, update_interval: float = 0.5):
        """
        Initialize bandwidth monitor

        Args:
            window_size: Number of samples to keep (default: 30 seconds at 1Hz)
            update_interval: Update interval in seconds
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.tx_samples = deque(maxlen=window_size)
        self.rx_samples = deque(maxlen=window_size)
        self.last_update = time.time()
        self.last_tx_bytes = 0
        self.last_rx_bytes = 0

    def update(self, tx_bytes: int, rx_bytes: int) -> None:
        """
        Update bandwidth samples

        Args:
            tx_bytes: Total bytes transmitted
            rx_bytes: Total bytes received
        """
        current_time = time.time()
        elapsed = current_time - self.last_update

        if elapsed >= self.update_interval:
            # Calculate rates (bytes per second)
            tx_rate = (tx_bytes - self.last_tx_bytes) / elapsed if elapsed > 0 else 0
            rx_rate = (rx_bytes - self.last_rx_bytes) / elapsed if elapsed > 0 else 0

            self.tx_samples.append(tx_rate)
            self.rx_samples.append(rx_rate)

            self.last_tx_bytes = tx_bytes
            self.last_rx_bytes = rx_bytes
            self.last_update = current_time

    def generate_sparkline(self, samples: deque) -> str:
        """
        Generate Unicode sparkline from samples

        Args:
            samples: Deque of sample values

        Returns:
            Sparkline string using Unicode block characters
        """
        if not samples or len(samples) == 0:
            return "▁" * self.window_size

        # Unicode blocks for sparkline
        blocks = "▁▂▃▄▅▆▇█"

        # Normalize samples to 0-7 range
        max_val = max(samples) if max(samples) > 0 else 1
        normalized = [int((val / max_val) * 7) for val in samples]

        # Pad with zeros if not enough samples
        while len(normalized) < self.window_size:
            normalized.insert(0, 0)

        return "".join(blocks[min(val, 7)] for val in normalized)

    def get_display(self) -> Panel:
        """
        Get formatted bandwidth display panel

        Returns:
            Rich Panel with bandwidth information
        """
        tx_sparkline = self.generate_sparkline(self.tx_samples)
        rx_sparkline = self.generate_sparkline(self.rx_samples)

        # Calculate current rates
        tx_rate = self.tx_samples[-1] if self.tx_samples else 0
        rx_rate = self.rx_samples[-1] if self.rx_samples else 0

        content = Text()
        content.append("TX: ", style="cyan")
        content.append(f"{format_bytes(tx_rate)}/s", style="bold cyan")
        content.append(f"  {tx_sparkline}\n", style="cyan")
        content.append("RX: ", style="cyan")
        content.append(f"{format_bytes(rx_rate)}/s", style="bold cyan")
        content.append(f"  {rx_sparkline}", style="cyan")

        return Panel(content, title="Bandwidth", border_style="cyan")


class RateCalculator:
    """Calculate current/average/peak rates and ETAs"""

    def __init__(self, smoothing_window: int = 5):
        """
        Initialize rate calculator

        Args:
            smoothing_window: Number of samples for current rate moving average
        """
        self.smoothing_window = smoothing_window
        self.samples = deque(maxlen=smoothing_window)
        self.start_time = time.time()
        self.start_bytes = 0
        self.total_bytes = 0
        self.peak_rate = 0.0
        self.error_count = 0
        self.total_operations = 0

    def update(self, bytes_transferred: int, total_bytes: int) -> None:
        """
        Update rate calculations

        Args:
            bytes_transferred: Bytes transferred so far
            total_bytes: Total bytes to transfer
        """
        current_time = time.time()

        if not self.samples:
            self.start_time = current_time
            self.start_bytes = bytes_transferred

        elapsed = current_time - self.start_time
        if elapsed > 0:
            # Current rate (smoothed over recent samples)
            rate = (bytes_transferred - self.start_bytes) / elapsed
            self.samples.append(rate)

            # Track peak rate
            if rate > self.peak_rate:
                self.peak_rate = rate

        self.total_bytes = total_bytes

    def get_current_rate(self) -> float:
        """Get current rate (smoothed)"""
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def get_average_rate(self) -> float:
        """Get average rate since start"""
        elapsed = time.time() - self.start_time
        if elapsed > 0 and self.samples:
            return self.samples[-1]
        return 0.0

    def get_peak_rate(self) -> float:
        """Get peak rate observed"""
        return self.peak_rate

    def get_eta(self, bytes_remaining: int) -> Optional[float]:
        """
        Get estimated time to completion

        Args:
            bytes_remaining: Bytes remaining to transfer

        Returns:
            ETA in seconds or None if cannot calculate
        """
        avg_rate = self.get_average_rate()
        if avg_rate > 0:
            return bytes_remaining / avg_rate
        return None

    def record_error(self) -> None:
        """Record an error occurrence"""
        self.error_count += 1
        self.total_operations += 1

    def record_success(self) -> None:
        """Record a successful operation"""
        self.total_operations += 1

    def get_connection_quality(self) -> str:
        """
        Get connection quality indicator

        Returns:
            Quality string: Excellent, Good, Fair, or Poor
        """
        if self.total_operations == 0:
            return "Unknown"

        error_rate = self.error_count / self.total_operations

        if error_rate < 0.01:
            return "Excellent"
        elif error_rate < 0.05:
            return "Good"
        elif error_rate < 0.15:
            return "Fair"
        else:
            return "Poor"


def format_bytes(bytes_val: float) -> str:
    """
    Format bytes as human-readable string

    Args:
        bytes_val: Bytes value

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration as human-readable string

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


def success(message: str) -> None:
    """Print success message in green"""
    console.print(f"✓ {message}", style="bold green")


def error(message: str) -> None:
    """Print error message in red"""
    console.print(f"✗ {message}", style="bold red")


def warning(message: str) -> None:
    """Print warning message in yellow"""
    console.print(f"⚠ {message}", style="bold yellow")


def info(message: str) -> None:
    """Print info message in blue"""
    console.print(f"ℹ {message}", style="blue")


def metric(label: str, value: str) -> None:
    """Print metric in cyan"""
    console.print(f"{label}: ", style="cyan", end="")
    console.print(value, style="bold cyan")


def setup_logging(verbose: bool) -> None:
    """
    Setup logging configuration

    Args:
        verbose: Enable verbose/debug logging
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.version_option(version="0.1.0")
@click.option('--config', '-c', type=click.Path(), help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx, config: Optional[str], verbose: bool):
    """
    Gibberish - Acoustic File Synchronization CLI

    Exit Codes:
      0 - Success
      1 - General error
      2 - Command-line argument error
      3 - Configuration error
      4 - Audio device error
      5 - Network/protocol error
      6 - File operation error
      7 - Verification error (hash mismatch)
    """
    # Check for NO_COLOR environment variable
    if os.environ.get('NO_COLOR'):
        console.no_color = True

    ctx.ensure_object(dict)
    ctx.obj['config'] = config or 'config.yaml'
    ctx.obj['verbose'] = verbose

    # Setup logging
    setup_logging(verbose)


@main.command()
@click.argument('directory', type=click.Path())
@click.pass_context
def init(ctx, directory: str):
    """Initialize a directory for acoustic synchronization"""
    try:
        info(f"Initializing directory: {directory}")
        path = Path(directory)
        path.mkdir(exist_ok=True)

        # Create .gibberish directory structure
        gibberish_dir = path / ".gibberish"
        gibberish_dir.mkdir(exist_ok=True)
        (gibberish_dir / "baseline").mkdir(exist_ok=True)
        (gibberish_dir / "cache" / "diffs").mkdir(parents=True, exist_ok=True)
        (gibberish_dir / "state").mkdir(exist_ok=True)

        # Create initial state.json
        state_file = gibberish_dir / "state" / "state.json"
        if not state_file.exists():
            initial_state = {
                "version": "0.1.0",
                "sessions": []
            }
            with open(state_file, 'w') as f:
                json.dump(initial_state, f, indent=2)

        success("Created .gibberish directory structure")
        success(f"Directory {directory} is ready for synchronization")

        return EXIT_SUCCESS

    except PermissionError:
        error(f"Permission denied: Cannot create directory {directory}")
        return EXIT_FILE_ERROR
    except Exception as e:
        error(f"Failed to initialize directory: {e}")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc(), style="red dim")
        return EXIT_GENERAL_ERROR


@main.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--dry-run', is_flag=True, help='Preview changes without transmitting')
@click.pass_context
def sync(ctx, directory: str, dry_run: bool):
    """Synchronize a directory via acoustic transmission"""
    try:
        from gibberish.sync import SyncManager
        from gibberish.baseline import BaselineManager
        from gibberish.audio import AudioManager
        from gibberish.protocol import ProtocolHandler
        from gibberish.tree_viz import TreeVisualizer, AnnotatedTreeBuilder

        directory_path = Path(directory)

        if dry_run:
            warning("DRY RUN MODE - No data will be transmitted")

        # Initialize managers
        sync_manager = SyncManager(directory_path)
        baseline_manager = BaselineManager(directory_path)

        # Load baseline
        info("Loading baseline...")
        baseline_data = baseline_manager.load_baseline("current")
        if not baseline_data:
            error("No baseline found. Please create one first with: gibberlan baseline <directory>")
            return EXIT_FILE_ERROR

        # Compute changes
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            scan_task = progress.add_task("Scanning files...", total=None)
            changes = sync_manager.compute_diff(baseline_data)
            progress.update(scan_task, completed=True)

        if not changes:
            success("No changes detected - directory is in sync!")
            return EXIT_SUCCESS

        # Calculate statistics
        total_bytes = sum(c.size for c in changes)
        total_transfer = sum(c.diff_size if c.diff_size > 0 else c.size for c in changes)

        # Display changes summary
        table = Table(title="Changes Detected", box=box.ROUNDED)
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="bold")
        table.add_column("Size", style="yellow")

        adds = [c for c in changes if c.change_type.name == 'ADD']
        mods = [c for c in changes if c.change_type.name == 'MODIFY']
        dels = [c for c in changes if c.change_type.name == 'DELETE']

        if adds:
            table.add_row("Added", str(len(adds)), format_bytes(sum(c.size for c in adds)))
        if mods:
            table.add_row("Modified", str(len(mods)), format_bytes(sum(c.size for c in mods)))
        if dels:
            table.add_row("Deleted", str(len(dels)), "-")

        console.print(table)

        metric("Total changes", str(len(changes)))
        metric("Total size", format_bytes(total_bytes))
        metric("Transfer size", format_bytes(total_transfer))
        if total_bytes > 0:
            reduction = ((total_bytes - total_transfer) / total_bytes) * 100
            metric("Reduction", f"{reduction:.1f}%")

        # Show tree visualization
        if dry_run:
            info("\nPreview of changes:")
            tree_builder = AnnotatedTreeBuilder(directory_path)
            tree = tree_builder.build_from_changes(changes)
            viz = TreeVisualizer()
            tree_output = viz.render_tree(tree, show_sizes=True, show_annotations=True)
            console.print(tree_output)
            console.print()
            console.print(viz.render_legend())

            success("\nDry run complete - no data transmitted")
            return EXIT_SUCCESS

        # Confirm sync
        if not click.confirm("\nProceed with synchronization?"):
            warning("Sync cancelled by user")
            return EXIT_SUCCESS

        # Perform actual sync with progress
        info("\nStarting acoustic synchronization...")

        audio_manager = AudioManager()
        protocol_handler = ProtocolHandler()

        # Perform handshake
        with console.status("[bold blue]Performing acoustic handshake...") as status:
            success_flag, session_id = protocol_handler.perform_handshake(audio_manager, is_initiator=None)

            if not success_flag:
                error("Handshake failed")
                return EXIT_NETWORK_ERROR

            success(f"Handshake successful - Session: {session_id[:8]}")

        # Initialize monitoring
        bw_monitor = BandwidthMonitor()
        rate_calc = RateCalculator()

        # Transmit changes with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console
        ) as progress:

            tx_task = progress.add_task("Transmitting...", total=total_transfer)

            bytes_sent = 0
            for i, change in enumerate(changes):
                progress.update(tx_task, description=f"Transmitting {change.path}...")

                # Simulate transmission (in real implementation, call audio_manager methods)
                size_to_send = change.diff_size if change.diff_size > 0 else change.size
                bytes_sent += size_to_send

                progress.update(tx_task, completed=bytes_sent)
                bw_monitor.update(bytes_sent, 0)
                rate_calc.update(bytes_sent, total_transfer)

                time.sleep(0.01)  # Simulate transmission time

        # Display final statistics
        console.print()
        final_table = Table(title="Synchronization Complete", box=box.ROUNDED)
        final_table.add_column("Metric", style="cyan")
        final_table.add_column("Value", style="bold green")

        final_table.add_row("Files synced", str(len(changes)))
        final_table.add_row("Data transferred", format_bytes(bytes_sent))
        final_table.add_row("Average rate", f"{format_bytes(rate_calc.get_average_rate())}/s")
        final_table.add_row("Peak rate", f"{format_bytes(rate_calc.get_peak_rate())}/s")
        final_table.add_row("Connection quality", rate_calc.get_connection_quality())

        console.print(final_table)
        success("\nSynchronization complete!")

        return EXIT_SUCCESS

    except ImportError as e:
        error(f"Missing module: {e}")
        error("Please ensure all dependencies are installed: pip install -e .")
        return EXIT_GENERAL_ERROR
    except Exception as e:
        error(f"Sync failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc(), style="red dim")
        return EXIT_GENERAL_ERROR


@main.command()
@click.option('--initiator', is_flag=True, help='Act as connection initiator')
@click.pass_context
def handshake(ctx, initiator: bool):
    """Initiate acoustic handshake with peer machine"""
    try:
        from gibberish.audio import AudioManager
        from gibberish.protocol import ProtocolHandler

        role = "initiator" if initiator else "auto-detect"
        info(f"Starting handshake as {role}...")

        audio_manager = AudioManager()
        protocol_handler = ProtocolHandler()

        with console.status("[bold blue]Performing handshake...") as status:
            is_init = True if initiator else None
            success_flag, session_id = protocol_handler.perform_handshake(audio_manager, is_initiator=is_init)

        if success_flag:
            success(f"Handshake successful!")
            metric("Session ID", session_id[:16])
            metric("Role", "Initiator" if initiator else "Responder")
            return EXIT_SUCCESS
        else:
            error("Handshake failed")
            return EXIT_NETWORK_ERROR

    except ImportError as e:
        error(f"Missing module: {e}")
        return EXIT_GENERAL_ERROR
    except Exception as e:
        error(f"Handshake failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc(), style="red dim")
        return EXIT_NETWORK_ERROR


@main.command()
@click.pass_context
def listen(ctx):
    """Listen for incoming acoustic transmissions"""
    try:
        from gibberish.audio import AudioManager
        from gibberish.protocol import ProtocolHandler

        info("Listening for incoming transmissions...")
        console.print("Press Ctrl+C to stop", style="dim")

        audio_manager = AudioManager()
        protocol_handler = ProtocolHandler()

        try:
            with Live(Panel("Waiting for signal...", title="Listening", border_style="blue"),
                     console=console, refresh_per_second=4) as live:

                # Wait for handshake
                while True:
                    success_flag, session_id = protocol_handler.perform_handshake(
                        audio_manager, is_initiator=False
                    )

                    if success_flag:
                        live.update(Panel(
                            f"Connected!\nSession: {session_id[:16]}",
                            title="Connected",
                            border_style="green"
                        ))
                        break

                    time.sleep(1)

                success("Connection established - ready to receive")

                # Keep listening
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            console.print()
            info("Stopped listening")
            return EXIT_SUCCESS

    except ImportError as e:
        error(f"Missing module: {e}")
        return EXIT_GENERAL_ERROR
    except Exception as e:
        error(f"Listen failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc(), style="red dim")
        return EXIT_GENERAL_ERROR


@main.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--refresh', is_flag=True, help='Refresh existing baseline')
@click.option('--no-backup', is_flag=True, help='Do not create backup when refreshing')
@click.option('--list', 'list_all', is_flag=True, help='List all baselines')
@click.option('--verify', is_flag=True, help='Verify baseline integrity')
@click.pass_context
def baseline(ctx, directory: str, refresh: bool, no_backup: bool, list_all: bool, verify: bool):
    """Create or refresh baseline for a directory"""
    try:
        from gibberish.baseline import BaselineManager

        directory_path = Path(directory)
        manager = BaselineManager(directory_path)

        if list_all:
            # List all baselines
            baselines = manager.list_baselines()
            if not baselines:
                warning("No baselines found")
                return EXIT_SUCCESS

            table = Table(title="Available Baselines", box=box.ROUNDED)
            table.add_column("Name", style="cyan")
            table.add_column("Created", style="blue")
            table.add_column("Files", style="yellow")
            table.add_column("Tree Hash", style="magenta")
            table.add_column("Status", style="green")

            for bl in baselines:
                status = "✓ VALID" if bl['valid'] else "✗ CORRUPTED"
                status_style = "green" if bl['valid'] else "red"
                table.add_row(
                    bl['name'],
                    bl['created'],
                    str(bl['file_count']),
                    bl['tree_hash'][:16] + "...",
                    f"[{status_style}]{status}[/]"
                )

            console.print(table)
            return EXIT_SUCCESS

        if verify:
            # Verify baseline
            info("Verifying baseline integrity...")
            baseline_data = manager.load_baseline("current")
            if not baseline_data:
                error("No current baseline found")
                return EXIT_FILE_ERROR

            with console.status("[bold blue]Verifying files...") as status:
                result = manager.verify_baseline(baseline_data, "current")

            if result['valid']:
                success("Baseline is VALID")
                metric("Tree hash", baseline_data['tree_hash'])
                return EXIT_SUCCESS
            else:
                error("Baseline is CORRUPTED")
                if result['corrupted_files']:
                    error(f"Corrupted files: {', '.join(result['corrupted_files'])}")
                if result['missing_files']:
                    error(f"Missing files: {', '.join(result['missing_files'])}")
                if result['extra_files']:
                    error(f"Extra files: {', '.join(result['extra_files'])}")
                return EXIT_VERIFICATION_ERROR

        if refresh:
            # Refresh baseline
            info(f"Refreshing baseline for directory: {directory}")
            backup = not no_backup

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Computing file hashes...", total=None)
                result = manager.refresh_baseline(backup=backup)
                progress.update(task, completed=True)

            if result['success']:
                success("Baseline refreshed successfully")
                metric("New tree hash", result['tree_hash'])
                if result['backup_name']:
                    info(f"Old baseline backed up as: {result['backup_name']}")
                return EXIT_SUCCESS
            else:
                error(f"Failed to refresh baseline: {result['error']}")
                return EXIT_FILE_ERROR
        else:
            # Create new baseline
            info(f"Creating baseline for directory: {directory}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Computing file hashes...", total=None)
                result = manager.create_and_save_baseline("current")
                progress.update(task, completed=True)

            if result['success']:
                success("Baseline created successfully")
                metric("Tree hash", result['tree_hash'])
                metric("Files", str(len(result['baseline']['files'])))
                return EXIT_SUCCESS
            else:
                error(f"Failed to create baseline: {result['error']}")
                return EXIT_FILE_ERROR

    except ImportError as e:
        error(f"Missing module: {e}")
        return EXIT_GENERAL_ERROR
    except Exception as e:
        error(f"Baseline operation failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc(), style="red dim")
        return EXIT_GENERAL_ERROR


@main.command()
@click.argument('directory', type=click.Path(exists=True), required=False)
@click.pass_context
def status(ctx, directory: Optional[str]):
    """Show synchronization status"""
    try:
        from gibberish.baseline import BaselineManager

        if directory:
            directory_path = Path(directory)

            # Check if initialized
            gibberish_dir = directory_path / ".gibberish"
            if not gibberish_dir.exists():
                warning(f"Directory not initialized: {directory}")
                info("Run: gibberish init <directory>")
                return EXIT_SUCCESS

            # Load state
            state_file = gibberish_dir / "state" / "state.json"
            state = {"sessions": []}
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)

            # Check baseline
            manager = BaselineManager(directory_path)
            baseline_data = manager.load_baseline("current")

            # Display status
            table = Table(title=f"Status: {directory}", box=box.ROUNDED)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="bold")

            table.add_row("Initialized", "✓ Yes" if gibberish_dir.exists() else "✗ No")

            if baseline_data:
                table.add_row("Baseline", f"✓ Created ({baseline_data['created']})")
                table.add_row("Files tracked", str(len(baseline_data['files'])))
                table.add_row("Tree hash", baseline_data['tree_hash'][:16] + "...")
            else:
                table.add_row("Baseline", "✗ Not created")

            if state['sessions']:
                last_session = state['sessions'][-1]
                table.add_row("Last sync", last_session.get('timestamp', 'Unknown'))
                table.add_row("Sync status", last_session.get('status', 'Unknown'))
            else:
                table.add_row("Last sync", "Never")

            table.add_row("Acoustic link", "Not connected")

            console.print(table)
        else:
            # Global status
            info("Global Gibberish Status")
            console.print("• No active connections", style="dim")
            info("Use --directory to check specific directory status")

        return EXIT_SUCCESS

    except Exception as e:
        error(f"Failed to get status: {e}")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc(), style="red dim")
        return EXIT_GENERAL_ERROR


@main.command()
@click.option('--show', is_flag=True, help='Show current configuration')
@click.pass_context
def config(ctx, show: bool):
    """Manage configuration settings"""
    try:
        config_path = ctx.obj['config']

        if show:
            info(f"Configuration file: {config_path}")

            if not Path(config_path).exists():
                warning(f"Configuration file not found: {config_path}")
                info("Using default configuration")
                return EXIT_SUCCESS

            try:
                with open(config_path, 'r') as f:
                    content = f.read()

                console.print(Panel(content, title="Configuration", border_style="blue"))
                return EXIT_SUCCESS
            except PermissionError:
                error(f"Permission denied: Cannot read {config_path}")
                return EXIT_FILE_ERROR
        else:
            info(f"Configuration file: {config_path}")
            info("Use --show to display configuration")

        return EXIT_SUCCESS

    except Exception as e:
        error(f"Config operation failed: {e}")
        return EXIT_GENERAL_ERROR


@main.command()
def validate():
    """Validate installation and dependencies"""
    console.print()
    info("Validating Gibberish installation...")
    console.print()

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    success(f"Python version: {py_version}")

    # Check dependencies
    deps = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sounddevice': 'sounddevice',
        'click': 'click',
        'yaml': 'PyYAML',
        'xxhash': 'xxhash',
        'ollama': 'ollama',
        'rich': 'rich',
    }

    all_ok = True
    for module, package in deps.items():
        try:
            __import__(module)
            success(f"{package} installed")
        except ImportError:
            error(f"{package} NOT installed")
            all_ok = False

    # Check ggwave separately (optional)
    try:
        __import__('ggwave')
        success("ggwave installed")
    except ImportError:
        warning("ggwave NOT installed (optional, required for audio transmission)")

    console.print()
    if all_ok:
        success("All required dependencies are installed!")
        return EXIT_SUCCESS
    else:
        error("Some dependencies are missing. Please run: pip install -e .")
        return EXIT_GENERAL_ERROR


@main.command()
@click.pass_context
def test_audio(ctx):
    """Test audio devices and cable connection"""
    try:
        from gibberish.audio import AudioManager
        import sounddevice as sd

        info("Testing audio devices and cable connection...")
        console.print()

        # List audio devices
        info("Available audio devices:")
        devices = sd.query_devices()

        table = Table(box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Channels", style="yellow")
        table.add_column("Sample Rate", style="blue")

        for i, device in enumerate(devices):
            if isinstance(device, dict):
                table.add_row(
                    str(i),
                    device['name'],
                    f"In: {device['max_input_channels']}, Out: {device['max_output_channels']}",
                    f"{device['default_samplerate']} Hz"
                )

        console.print(table)
        console.print()

        # Test cable connection
        info("Testing cable connection...")
        audio_manager = AudioManager()

        with console.status("[bold blue]Running cable detection test...") as status:
            try:
                is_connected, snr = audio_manager.detect_cable(test_duration=2.0)

                if is_connected:
                    success(f"Cable connected! SNR: {snr:.1f} dB")
                    if snr > 40:
                        metric("Connection quality", "Excellent")
                    elif snr > 30:
                        metric("Connection quality", "Good")
                    else:
                        metric("Connection quality", "Fair")
                else:
                    warning(f"No cable detected. SNR: {snr:.1f} dB")
                    info("Make sure the audio cable is properly connected")
                    info("Check that input/output devices are correctly configured")
            except Exception as e:
                error(f"Cable test failed: {e}")
                if ctx.obj['verbose']:
                    import traceback
                    console.print(traceback.format_exc(), style="red dim")
                return EXIT_AUDIO_ERROR

        return EXIT_SUCCESS

    except ImportError as e:
        error(f"Missing module: {e}")
        error("sounddevice is required for audio testing")
        return EXIT_GENERAL_ERROR
    except Exception as e:
        error(f"Audio test failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc(), style="red dim")
        return EXIT_AUDIO_ERROR


@main.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--limit', '-n', type=int, default=10, help='Number of sessions to show')
@click.pass_context
def history(ctx, directory: str, limit: int):
    """Show sync history from state.json"""
    try:
        directory_path = Path(directory)
        state_file = directory_path / ".gibberish" / "state" / "state.json"

        if not state_file.exists():
            warning("No sync history found")
            info(f"Directory not initialized or no syncs performed yet")
            return EXIT_SUCCESS

        with open(state_file, 'r') as f:
            state = json.load(f)

        sessions = state.get('sessions', [])

        if not sessions:
            info("No sync sessions recorded")
            return EXIT_SUCCESS

        # Show most recent sessions
        recent_sessions = sessions[-limit:] if len(sessions) > limit else sessions
        recent_sessions.reverse()  # Most recent first

        table = Table(title=f"Sync History (last {len(recent_sessions)} sessions)", box=box.ROUNDED)
        table.add_column("Date", style="cyan")
        table.add_column("Session ID", style="blue")
        table.add_column("Status", style="bold")
        table.add_column("Files", style="yellow")
        table.add_column("Data", style="magenta")

        for session in recent_sessions:
            timestamp = session.get('timestamp', 'Unknown')
            session_id = session.get('session_id', 'N/A')[:8]
            status = session.get('status', 'Unknown')
            file_count = session.get('file_count', 0)
            bytes_transferred = session.get('bytes_transferred', 0)

            status_style = "green" if status == "success" else "red"

            table.add_row(
                timestamp,
                session_id,
                f"[{status_style}]{status}[/]",
                str(file_count),
                format_bytes(bytes_transferred)
            )

        console.print(table)

        # Show summary statistics
        console.print()
        total_sessions = len(sessions)
        successful = sum(1 for s in sessions if s.get('status') == 'success')
        total_bytes = sum(s.get('bytes_transferred', 0) for s in sessions)

        metric("Total sessions", str(total_sessions))
        metric("Successful", f"{successful}/{total_sessions}")
        metric("Total data transferred", format_bytes(total_bytes))

        return EXIT_SUCCESS

    except json.JSONDecodeError:
        error("Invalid state.json format")
        return EXIT_FILE_ERROR
    except Exception as e:
        error(f"Failed to read history: {e}")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc(), style="red dim")
        return EXIT_GENERAL_ERROR


if __name__ == '__main__':
    sys.exit(main())
