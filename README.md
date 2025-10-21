# Gibberish

Acoustic file synchronization for air-gapped machines using sound waves.

## Installation

### Requirements
- Python 3.10 (for ggwave support)
- Audio hardware (speakers + microphone or audio cable)

### Setup

```bash
# Install Python 3.10
brew install python@3.10  # macOS
# or use your system package manager

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install Gibberish
pip install -e .

# Validate installation
gibberish validate
```

## Quick Start

```bash
# Initialize directory
gibberish init /path/to/sync

# Create baseline
gibberish baseline /path/to/sync

# Preview changes (dry-run)
gibberish sync /path/to/sync --dry-run

# Sender: transmit files
gibberish sync /path/to/sync

# Receiver: listen for transmission
gibberish listen
```

## Commands

- `gibberish init <dir>` - Initialize directory
- `gibberish baseline <dir>` - Create/manage baseline
- `gibberish sync <dir>` - Transmit files acoustically
- `gibberish listen` - Receive acoustic transmission
- `gibberish status <dir>` - Show sync status
- `gibberish validate` - Check installation

## Configuration

Edit `config.yaml` to customize audio settings, transmission modes, and sync options.

## License

MIT
