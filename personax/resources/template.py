from __future__ import annotations

import logging
import typing as t

import jinja2 as j2

from personax.exceptions import ResourceError
from personax.resources import Resource
from personax.resources import WatchedResource

logger = logging.getLogger("personax.resources.template")


@t.runtime_checkable
class Template(t.Protocol):
    """Protocol for template rendering.

    Defines the interface for template objects that can render content with
    variable substitution. Used throughout PersonaX for formatting context data
    into text.

    Example:
        ```python
        def use_template(template: Template):
            output = template.render(name="Alice", age=30)
            print(output)
        ```
    """

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        """Render template with provided variables.

        Args:
            *args: Positional arguments for template rendering.
            **kwargs: Keyword arguments for template variable substitution.

        Returns:
            Rendered template as string.
        """


class J2Template(Resource[j2.Template], Template):
    """Jinja2 template resource loaded from file.

    Loads a Jinja2 template from a file and provides rendering capabilities.
    Template content is read once on initialization and can be rendered
    multiple times with different variables.

    Example:
        ```python
        # template.j2:
        # Hello {{ name }}, you are {{ age }} years old.

        template = J2Template("template.j2")
        output = template.render(name="Alice", age=30)
        print(output)  # "Hello Alice, you are 30 years old."
        ```

    Raises:
        ResourceError: If template is not loaded before rendering.
    """

    def _parse(self) -> j2.Template:
        """Parse Jinja2 template from file.

        Returns:
            Compiled Jinja2 template object.
        """
        content = self.fpath.read_text(encoding="utf-8").strip()
        return j2.Template(content)

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        """Render the template with provided variables.

        Args:
            *args: Positional arguments passed to Jinja2 render.
            **kwargs: Template variables for substitution.

        Returns:
            Rendered template string.

        Raises:
            ResourceError: If template data is not loaded.
        """
        if self.data is None:
            raise ResourceError("Template not loaded.")
        return self.data.render(*args, **kwargs)


class WatchedJ2Template(WatchedResource[j2.Template], Template):
    """Auto-reloading Jinja2 template resource.

    Extends J2Template with automatic reloading when the template file is
    modified. Useful for development and dynamic template updates.

    Example:
        ```python
        template = WatchedJ2Template("template.j2")

        # Initial render
        print(template.render(name="Alice"))

        # Modify template.j2 externally
        # Next render uses updated template automatically
        print(template.render(name="Bob"))
        ```

    Note:
        - Template is reloaded automatically on file modification
        - Thread-safe rendering during reload
        - Previous template is preserved if reload fails
    """

    def _parse(self) -> j2.Template:
        """Parse Jinja2 template from file.

        Returns:
            Compiled Jinja2 template object.
        """
        content = self.fpath.read_text(encoding="utf-8").strip()
        return j2.Template(content)

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        """Render the template with provided variables.

        Args:
            *args: Positional arguments passed to Jinja2 render.
            **kwargs: Template variables for substitution.

        Returns:
            Rendered template string.

        Raises:
            ResourceError: If template data is not loaded.
        """
        if self.data is None:
            raise ResourceError("Template not loaded.")
        return self.data.render(*args, **kwargs)
