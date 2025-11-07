from __future__ import annotations

import abc
import logging
import os
import pathlib as p
import threading
import typing as t

import watchdog.events as evt
import watchdog.observers as obsrv
import watchdog.observers.api as obsrv_api

logger = logging.getLogger("personax.resources")


class FileHandler(evt.FileSystemEventHandler):
    """File system event handler for resource auto-reloading.

    Monitors file modifications and triggers a callback when changes are
    detected. Used by WatchedResource to automatically reload data when the
    underlying file is modified.

    Attributes:
        callback: Function to call when file is modified.
        fpath: Path to the file being monitored.

    Args:
        callback: No-argument function to invoke on file modification.
        fpath: Path to the file to monitor.
    """

    def __init__(self, callback: t.Callable[[], None], fpath: p.Path):
        self.callback = callback
        self.fpath = fpath

    def on_modified(self, event: evt.FileModifiedEvent | evt.DirModifiedEvent) -> None:
        """Handle file modification events.

        Invokes the callback when the monitored file is modified.

        Args:
            event: File system event (modification).
        """
        if not (
            event.is_directory
            and p.Path(
                event.src_path
                if isinstance(event.src_path, str)
                else event.src_path.decode("utf-8")
            )
            == self.fpath
        ):
            self.callback()


T = t.TypeVar("T")


class Resource(abc.ABC, t.Generic[T]):
    """Base class for file-based resources with automatic loading.

    Provides a framework for loading and managing data from files with built-in
    error handling and backup functionality. Subclasses define how to parse
    file content into structured data.

    Type Parameters:
        T: The type of parsed data this resource manages.

    Attributes:
        fpath: Path to the resource file.
        data: Currently loaded parsed data.
        backup: Backup of previous data for rollback on parse errors.
        lock: Thread lock for safe concurrent access.

    Args:
        fpath: Path to the resource file.

    Example:
        ```python
        class ConfigResource(Resource[dict]):
            def _parse(self) -> dict:
                import json
                return json.loads(self.fpath.read_text())

        config = ConfigResource("config.json")
        print(config.data["api_key"])
        ```
    """

    def __init__(self, fpath: str | p.Path | os.PathLike[str]):
        self.fpath = p.Path(fpath)
        self.data: T | None = None
        self.backup: T | None = None
        self.lock = threading.RLock()
        self.load()

    @abc.abstractmethod
    def _parse(self) -> T:
        """Parse file content into structured data.

        Must be implemented by subclasses to define how to read and
        parse the resource file.

        Returns:
            Parsed data of type T.

        Raises:
            Any parsing errors should be raised to trigger backup restoration.
        """

    def load(self) -> None:
        """Load and parse the resource file.

        Attempts to parse the file using _parse(). On success, updates data and
        creates a backup. On failure, restores from backup if available and
        logs the error.

        Thread-safe via internal lock.
        """
        try:
            with self.lock:
                data = self._parse()
                self.backup = self.data
                self.data = data
                logging.info("Loaded data from file: %s", self.fpath)
        except Exception as e:
            logging.error("Failed to load data from file: %s - %s", self.fpath, e)
            if self.backup is not None:
                self.data = self.backup

    def __fspath__(self) -> str:
        """Return file path for os.PathLike protocol.

        Returns:
            String representation of the file path.
        """
        return str(self.fpath)


class WatchedResource(Resource[T], abc.ABC):
    """Resource with automatic file monitoring and reloading.

    Extends Resource with file system monitoring capabilities. Automatically
    reloads data when the underlying file is modified, enabling hot-reload
    functionality for configuration files, templates, etc.

    Type Parameters:
        T: The type of parsed data this resource manages.

    Attributes:
        observer: File system observer for monitoring changes.

    Example:
        ```python
        class WatchedConfig(WatchedResource[dict]):
            def _parse(self) -> dict:
                import json
                return json.loads(self.fpath.read_text())

        config = WatchedConfig("config.json")
        # Modify config.json externally
        # config.data automatically updates
        ```

    Note:
        - Observer is started automatically on initialization
        - Observer is stopped automatically on deletion
        - Use stop() to manually stop monitoring
    """

    def __init__(self, fpath: str | p.Path | os.PathLike[str]):
        super().__init__(fpath)
        self.observer: obsrv_api.BaseObserver | None = None
        self.watch()

    def watch(self) -> None:
        """Start monitoring the resource file for changes.

        Sets up a file system observer to watch for modifications and
        trigger automatic reloading via load().
        """
        if not self.fpath.exists():
            return

        self.observer = obsrv.Observer()
        handler = FileHandler(self.load, self.fpath)
        self.observer.schedule(handler, str(self.fpath.parent), recursive=False)
        self.observer.start()
        logger.info("Monitoring file: %s", self.fpath)

    def stop(self) -> None:
        """Stop monitoring the resource file.

        Stops and joins the file system observer thread.
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def __del__(self) -> None:
        """Cleanup observer on deletion."""
        self.stop()
