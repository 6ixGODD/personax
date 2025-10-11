from __future__ import annotations

import typing as t

import jinja2 as j2

from personax.resources import WatchedResource


@t.runtime_checkable
class Template(t.Protocol):

    def render(self, **kwargs: t.Any) -> str:
        pass


class WatchedTemplate(WatchedResource[j2.Template]):

    def _parse(self) -> j2.Template:
        pass

    def render(self, **kwargs: t.Any) -> str:
        pass
