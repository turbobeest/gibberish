# Gibberish

Acoustic file synchronization for air-gapped machines using sound waves.

## Installation

### Requirements
- Python 3.10+
- Audio hardware (speakers + microphone or audio cable)

### Setup

**macOS:**
```bash
# Install Python 3.10
brew install python@3.10

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install Gibberish
pip install -e .

# Validate installation
gibberish validate
```

**Windows:**
```powershell
# Download and install Python 3.10+ from python.org

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install Gibberish
pip install -e .

# Validate installation
gibberish validate
```

## Quick Start

### Interactive Mode (Recommended)

The simplest way to use Gibberish - guided step-by-step:

```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Run interactive mode
gibberish interactive
```

The interactive mode will:
1. Ask if you're the **transmitter** or **receiver**
2. Guide you through selecting directories
3. Show file tree comparisons
4. Coordinate the sync between both machines
5. Transmit changes via sound waves

### Advanced Mode (Manual Commands)

For advanced users who want more control:

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

- `gibberish interactive` - **Guided interactive mode (recommended)**
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
