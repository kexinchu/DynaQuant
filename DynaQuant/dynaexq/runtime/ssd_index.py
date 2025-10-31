"""
SSD Indexer - Memory-mapped expert storage on SSD with fast lookup
"""

import os
import json
import mmap
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import struct

from .types import ExpertID

logger = logging.getLogger(__name__)


class SSDIndex:
    """
    Memory-mapped SSD storage for expert weights.

    Format:
    - experts.bin: Raw binary file with all expert weights concatenated
    - experts.index: JSON index mapping ExpertID -> (offset, size, bitwidth)

    Supports fast random access via mmap.
    """

    def __init__(
        self,
        ssd_path: str,
        index_path: str,
        create_if_missing: bool = True,
    ):
        """
        Args:
            ssd_path: Path to binary file (experts.bin)
            index_path: Path to index file (experts.index)
            create_if_missing: Create new files if they don't exist
        """
        self.ssd_path = Path(ssd_path)
        self.index_path = Path(index_path)

        # Index structure: ExpertID -> (offset, size, bitwidth)
        self.index: Dict[str, Tuple[int, int, str]] = {}

        # Memory-mapped file
        self.mmap_file = None
        self.file_handle = None

        # Statistics
        self.read_count = 0
        self.write_count = 0
        self.total_bytes_read = 0
        self.total_bytes_written = 0

        # Initialize
        self._initialize(create_if_missing)

        logger.info(
            f"SSDIndex initialized: {self.ssd_path}, {len(self.index)} experts")

    def _initialize(self, create_if_missing: bool):
        """Initialize or load the SSD storage"""
        # Load index if it exists
        if self.index_path.exists():
            self._load_index()
        elif create_if_missing:
            # Create empty index
            self.index = {}
            self._save_index()
            # Create empty binary file
            self.ssd_path.parent.mkdir(parents=True, exist_ok=True)
            self.ssd_path.touch()
        else:
            raise FileNotFoundError(f"SSD index not found: {self.index_path}")

        # Open binary file for memory mapping
        if self.ssd_path.exists() and self.ssd_path.stat().st_size > 0:
            self._open_mmap()

    def _load_index(self):
        """Load index from JSON file"""
        with open(self.index_path, 'r') as f:
            raw_index = json.load(f)

        # Convert keys from "L{layer}E{idx}" to tuple
        self.index = {}
        for key, value in raw_index.items():
            self.index[key] = tuple(value)  # (offset, size, bitwidth)

        logger.info(f"Loaded index with {len(self.index)} entries")

    def _save_index(self):
        """Save index to JSON file"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

        logger.debug(f"Saved index with {len(self.index)} entries")

    def _open_mmap(self):
        """Open the binary file for memory mapping"""
        if self.mmap_file is not None:
            return  # Already open

        try:
            self.file_handle = open(self.ssd_path, 'r+b')
            self.mmap_file = mmap.mmap(
                self.file_handle.fileno(),
                0,
                access=mmap.ACCESS_READ
            )
            logger.debug("Memory-mapped SSD file")
        except Exception as e:
            logger.warning(f"Failed to mmap file: {e}, using regular I/O")
            if self.file_handle:
                self.file_handle.close()
            self.file_handle = None
            self.mmap_file = None

    def _expert_key(self, expert: ExpertID) -> str:
        """Convert ExpertID to string key"""
        return f"L{expert.layer}E{expert.idx}"

    def write_expert(
        self,
        expert: ExpertID,
        data: bytes,
        bitwidth: str
    ) -> bool:
        """
        Write expert weights to SSD.

        Args:
            expert: Expert identifier
            data: Binary weight data
            bitwidth: "W4" or "W2"

        Returns:
            True if successful
        """
        try:
            key = self._expert_key(expert)

            # Close mmap if open (need write access)
            if self.mmap_file is not None:
                self.mmap_file.close()
                self.mmap_file = None

            # Append to file
            with open(self.ssd_path, 'ab') as f:
                offset = f.tell()
                f.write(data)

            # Update index
            self.index[key] = (offset, len(data), bitwidth)
            self._save_index()

            # Reopen mmap
            self._open_mmap()

            self.write_count += 1
            self.total_bytes_written += len(data)

            logger.debug(
                f"Wrote {expert} to SSD: {len(data)} bytes at offset {offset}")
            return True

        except Exception as e:
            logger.error(f"Failed to write {expert} to SSD: {e}")
            return False

    def read_expert(self, expert: ExpertID) -> Optional[Tuple[bytes, str]]:
        """
        Read expert weights from SSD.

        Args:
            expert: Expert identifier

        Returns:
            Tuple of (data, bitwidth) or None if not found
        """
        key = self._expert_key(expert)

        if key not in self.index:
            logger.warning(f"Expert {expert} not found in SSD index")
            return None

        offset, size, bitwidth = self.index[key]

        try:
            if self.mmap_file is not None:
                # Use mmap for fast access
                data = bytes(self.mmap_file[offset:offset + size])
            else:
                # Fall back to regular I/O
                with open(self.ssd_path, 'rb') as f:
                    f.seek(offset)
                    data = f.read(size)

            self.read_count += 1
            self.total_bytes_read += size

            logger.debug(f"Read {expert} from SSD: {size} bytes")
            return (data, bitwidth)

        except Exception as e:
            logger.error(f"Failed to read {expert} from SSD: {e}")
            return None

    def has_expert(self, expert: ExpertID) -> bool:
        """Check if expert exists in SSD storage"""
        key = self._expert_key(expert)
        return key in self.index

    def get_expert_info(self, expert: ExpertID) -> Optional[Tuple[int, int, str]]:
        """
        Get expert metadata without reading data.

        Returns:
            Tuple of (offset, size, bitwidth) or None if not found
        """
        key = self._expert_key(expert)
        return self.index.get(key)

    def get_statistics(self) -> Dict:
        """Get SSD index statistics"""
        total_size = sum(size for _, size, _ in self.index.values())

        return {
            "num_experts": len(self.index),
            "total_size_gb": total_size / 1e9,
            "ssd_path": str(self.ssd_path),
            "index_path": str(self.index_path),
            "read_count": self.read_count,
            "write_count": self.write_count,
            "total_bytes_read_gb": self.total_bytes_read / 1e9,
            "total_bytes_written_gb": self.total_bytes_written / 1e9,
        }

    def close(self):
        """Close the memory-mapped file"""
        if self.mmap_file is not None:
            self.mmap_file.close()
            self.mmap_file = None

        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

        logger.info("SSDIndex closed")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()
