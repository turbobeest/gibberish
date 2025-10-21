"""
Gibberish - Acoustic File Synchronization

A Python package for synchronizing files between machines using acoustic data transmission.
Leverages ggwave for acoustic communication to enable air-gapped file synchronization.
"""

__version__ = "0.1.0"
__author__ = "Gibberish Team"

from .gibberish_cli import main
from .audio import AudioManager
from .sync import SyncManager
from .baseline import BaselineManager
from .protocol import ProtocolHandler
from .llm import LLMClient

__all__ = [
    "main",
    "AudioManager",
    "SyncManager",
    "BaselineManager",
    "ProtocolHandler",
    "LLMClient",
]
