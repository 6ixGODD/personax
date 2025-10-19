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

logger = logging.getLogger('personax.resources')


class FileHandler(evt.FileSystemEventHandler):

    def __init__(self, callback: t.Callable[[], None], fpath: p.Path):
        self.callback = callback
        self.fpath = fpath

    def on_modified(self, event: evt.FileModifiedEvent | evt.DirModifiedEvent) -> None:
        if not (
            event.is_directory and p.Path(
                event.src_path if isinstance(event.src_path, str) else event.src_path.
                decode('utf-8')
            ) == self.fpath
        ):
            self.callback()


T = t.TypeVar("T")


class Resource(abc.ABC, t.Generic[T]):

    def __init__(self, fpath: str | p.Path | os.PathLike[str]):
        self.fpath = p.Path(fpath)
        self.data: T | None = None
        self.backup: T | None = None
        self.lock = threading.RLock()
        self.load()

    @abc.abstractmethod
    def _parse(self) -> T:
        pass

    def load(self) -> None:
        try:
            with self.lock:
                data = self._parse()
                self.backup = self.data  # Backup current data
                self.data = data
                logging.info("Loaded data from file: %s", self.fpath)
        except Exception as e:
            logging.error("Failed to load data from file: %s - %s", self.fpath, e)
            if self.backup is not None:
                self.data = self.backup

    def __fspath__(self) -> str:
        return str(self.fpath)


class WatchedResource(Resource[T], abc.ABC):

    def __init__(self, fpath: str | p.Path | os.PathLike[str]):
        super().__init__(fpath)
        self.observer: t.Optional[obsrv_api.BaseObserver] = None
        self.watch()

    def watch(self) -> None:
        if not self.fpath.exists():
            return

        self.observer = obsrv.Observer()
        handler = FileHandler(self.load, self.fpath)
        self.observer.schedule(handler, str(self.fpath.parent), recursive=False)
        self.observer.start()
        logger.info("Monitoring file: %s", self.fpath)

    def stop(self) -> None:
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def __del__(self) -> None:
        self.stop()
