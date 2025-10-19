from __future__ import annotations

import logging
import typing as t

import jinja2 as j2

from personax.exceptions import ResourceException
from personax.resources import Resource, WatchedResource

logger = logging.getLogger('personax.resources.template')

# pylint: disable=too-few-public-methods
@t.runtime_checkable
class Template(t.Protocol):

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        pass


class J2Template(Resource[j2.Template], Template):

    def _parse(self) -> j2.Template:
        content = self.fpath.read_text(encoding='utf-8').strip()
        return j2.Template(content)

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        if self.data is None:
            raise ResourceException("Template not loaded.")
        return self.data.render(*args, **kwargs)


class WatchedJ2Template(WatchedResource[j2.Template], Template):

    def _parse(self) -> j2.Template:
        content = self.fpath.read_text(encoding='utf-8').strip()
        return j2.Template(content)

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        if self.data is None:
            raise ResourceException("Template not loaded.")
        return self.data.render(*args, **kwargs)
