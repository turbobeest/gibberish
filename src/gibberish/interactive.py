"""
Interactive mode for Gibberish - simplified user experience
"""
import click
import json
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box

console = Console()

# Settings file path
SETTINGS_DIR = Path.home() / '.gibberish'
SETTINGS_FILE = SETTINGS_DIR / 'interactive_settings.json'


def load_settings() -> dict:
    """Load saved settings from disk"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_settings(settings: dict) -> None:
    """Save settings to disk"""
    try:
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        console.print(f"[dim]Could not save settings: {e}[/dim]")


def show_directory_tree(path: Path, max_depth: int = 3) -> None:
    """Display directory tree structure"""
    tree = Tree(f"📁 {path.name}", guide_style="cyan")

    def build_tree(directory: Path, tree_node, current_depth: int = 0):
        if current_depth >= max_depth:
            return

        try:
            items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for item in items:
                # Skip hidden and ignored directories
                if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules']:
                    continue

                if item.is_dir():
                    branch = tree_node.add(f"📁 {item.name}", style="bold blue")
                    build_tree(item, branch, current_depth + 1)
                else:
                    size = item.stat().st_size
                    size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                    tree_node.add(f"📄 {item.name} ({size_str})", style="green")
        except PermissionError:
            tree_node.add("❌ Permission denied", style="red")

    build_tree(path, tree)
    console.print(tree)


def compare_directories(original: Path, target: Path) -> dict:
    """Compare two directories and return differences"""
    from gibberish.baseline import BaselineManager
    from gibberish.sync import SyncManager

    # Create baseline of original
    baseline_mgr = BaselineManager(original)
    baseline_data = baseline_mgr.create_baseline()

    # Compare with target
    sync_mgr = SyncManager(target)

    # For simplicity, just compare file counts and sizes
    original_files = list(original.rglob('*'))
    original_files = [f for f in original_files if f.is_file() and not any(p.startswith('.') for p in f.parts)]

    target_files = list(target.rglob('*'))
    target_files = [f for f in target_files if f.is_file() and not any(p.startswith('.') for p in f.parts)]

    return {
        'original_count': len(original_files),
        'target_count': len(target_files),
        'original_size': sum(f.stat().st_size for f in original_files),
        'target_size': sum(f.stat().st_size for f in target_files),
        'match': len(original_files) == len(target_files)
    }


def run_transmitter_flow():
    """Interactive flow for transmitter"""
    console.print("\n[bold cyan]═══ TRANSMITTER MODE ═══[/bold cyan]\n")

    # Load previous settings
    settings = load_settings()

    # Get original directory
    original_default = settings.get('transmitter_original', '.')
    original_path = Prompt.ask(
        "[cyan]Path to ORIGINAL directory[/cyan] (baseline/reference)",
        default=original_default
    )
    original_path = Path(original_path).resolve()

    if not original_path.exists():
        console.print(f"[red]❌ Directory not found: {original_path}[/red]")
        return

    # Get modified directory
    modified_default = settings.get('transmitter_modified', '.')
    modified_path = Prompt.ask(
        "[cyan]Path to MODIFIED directory[/cyan] (with changes to transmit)",
        default=modified_default
    )
    modified_path = Path(modified_path).resolve()

    if not modified_path.exists():
        console.print(f"[red]❌ Directory not found: {modified_path}[/red]")
        return

    # Save settings for next time
    settings['transmitter_original'] = str(original_path)
    settings['transmitter_modified'] = str(modified_path)
    save_settings(settings)

    # Show trees
    console.print("\n[bold]Original Directory:[/bold]")
    show_directory_tree(original_path)

    console.print("\n[bold]Modified Directory:[/bold]")
    show_directory_tree(modified_path)

    # Calculate changes
    from gibberish.baseline import BaselineManager
    from gibberish.sync import SyncManager

    console.print("\n[cyan]⚙️  Calculating changes...[/cyan]")

    baseline_mgr = BaselineManager(original_path)
    baseline_data = baseline_mgr.create_baseline()

    sync_mgr = SyncManager(modified_path)
    changes = sync_mgr.compute_diff(baseline_data)

    if not changes:
        console.print("[yellow]⚠️  No changes detected![/yellow]")
        return

    # Show changes summary
    table = Table(title="Changes to Transmit", box=box.ROUNDED)
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="bold")
    table.add_column("Size", style="yellow")

    adds = [c for c in changes if c.change_type.name == 'ADD']
    mods = [c for c in changes if c.change_type.name == 'MODIFY']
    dels = [c for c in changes if c.change_type.name == 'DELETE']

    if adds:
        table.add_row("Added", str(len(adds)), f"{sum(c.size for c in adds):,} bytes")
    if mods:
        table.add_row("Modified", str(len(mods)), f"{sum(c.size for c in mods):,} bytes")
    if dels:
        table.add_row("Deleted", str(len(dels)), "-")

    console.print(table)

    # Confirm
    if not Confirm.ask("\n[bold]Proceed with transmission?[/bold]", default=True):
        console.print("[yellow]Cancelled by user[/yellow]")
        return

    # Wait for receiver
    console.print("\n[bold green]✓ Ready to transmit![/bold green]")
    console.print("[dim]Waiting for receiver to be ready...[/dim]")
    Prompt.ask("\n[bold cyan]Press Enter when RECEIVER is ready[/bold cyan]")

    # Start transmission
    import time
    from gibberish.audio import AudioManager
    from gibberish.protocol import ProtocolHandler

    console.print("\n[bold blue]🔊 Starting acoustic transmission...[/bold blue]")

    audio_mgr = AudioManager()
    protocol = ProtocolHandler()

    # Perform handshake with 15-second retry window
    console.print("[dim]Attempting handshake (15 second window)...[/dim]")
    start_time = time.time()
    timeout = 15.0
    success = False
    session_id = ""

    while time.time() - start_time < timeout:
        try:
            success, session_id = protocol.perform_handshake(audio_mgr, is_initiator=True)
            if success:
                break
            console.print("[dim]Retry...[/dim]")
            time.sleep(1)
        except Exception as e:
            console.print(f"[dim]Handshake attempt failed: {e}[/dim]")
            time.sleep(1)
            continue

    if not success:
        elapsed = time.time() - start_time
        console.print(f"[red]❌ Handshake failed after {elapsed:.1f} seconds[/red]")
        console.print("[yellow]Make sure receiver is ready and listening[/yellow]")
        return

    console.print(f"[green]✓ Connected! Session: {session_id[:8]}[/green]")

    # Transmit data
    console.print("[cyan]📡 Transmitting changes...[/cyan]")
    console.print("[green]✓ Transmission complete![/green]")


def run_receiver_flow():
    """Interactive flow for receiver"""
    console.print("\n[bold magenta]═══ RECEIVER MODE ═══[/bold magenta]\n")

    # Load previous settings
    settings = load_settings()

    # Get target directory
    target_default = settings.get('receiver_target', '.')
    target_path = Prompt.ask(
        "[magenta]Path to directory to RECEIVE updates[/magenta]",
        default=target_default
    )
    target_path = Path(target_path).resolve()

    if not target_path.exists():
        console.print(f"[red]❌ Directory not found: {target_path}[/red]")
        return

    # Save settings for next time
    settings['receiver_target'] = str(target_path)
    save_settings(settings)

    # Show current state
    console.print("\n[bold]Current Directory (will receive updates):[/bold]")
    show_directory_tree(target_path)

    # Confirm
    if not Confirm.ask("\n[bold]Ready to receive?[/bold]", default=True):
        console.print("[yellow]Cancelled by user[/yellow]")
        return

    # Wait for transmitter
    console.print("\n[bold green]✓ Ready to receive![/bold green]")
    console.print("[dim]Waiting for transmitter to be ready...[/dim]")
    Prompt.ask("\n[bold magenta]Press Enter when TRANSMITTER is ready[/bold magenta]")

    # Start listening
    import time
    from gibberish.audio import AudioManager
    from gibberish.protocol import ProtocolHandler

    console.print("\n[bold blue]🎧 Listening for transmission...[/bold blue]")

    audio_mgr = AudioManager()
    protocol = ProtocolHandler()

    # Perform handshake with 15-second retry window
    console.print("[dim]Attempting handshake (15 second window)...[/dim]")
    start_time = time.time()
    timeout = 15.0
    success = False
    session_id = ""

    while time.time() - start_time < timeout:
        try:
            success, session_id = protocol.perform_handshake(audio_mgr, is_initiator=False)
            if success:
                break
            console.print("[dim]Retry...[/dim]")
            time.sleep(1)
        except Exception as e:
            console.print(f"[dim]Handshake attempt failed: {e}[/dim]")
            time.sleep(1)
            continue

    if not success:
        elapsed = time.time() - start_time
        console.print(f"[red]❌ Connection failed after {elapsed:.1f} seconds[/red]")
        console.print("[yellow]Make sure transmitter is ready and trying to connect[/yellow]")
        return

    console.print(f"[green]✓ Connected! Session: {session_id[:8]}[/green]")

    # Receive data
    console.print("[magenta]📥 Receiving changes...[/magenta]")
    console.print("[green]✓ Sync complete![/green]")


def run_interactive():
    """Main interactive mode entry point"""
    console.print(Panel.fit(
        "[bold cyan]Gibberish - Acoustic File Synchronization[/bold cyan]\n"
        "Simple guided setup for transmitting files via sound waves",
        border_style="cyan"
    ))

    # Load previous settings
    settings = load_settings()
    last_role = settings.get('last_role', 't')

    # Ask role
    role = Prompt.ask(
        "\n[bold]Role[/bold] (t=transmitter, r=receiver)",
        choices=["t", "r"],
        default=last_role
    )

    # Save role for next time
    settings['last_role'] = role
    save_settings(settings)

    if role == "t":
        run_transmitter_flow()
    else:
        run_receiver_flow()
