"""
Internationalisation support for xarray-zarr-xgroup.

All user-facing strings in the package should be wrapped with the _()
function exported from this module. This ensures they are extractable
for translation and correctly localised at runtime.

Usage::

    from zarr_xgroup.i18n import _

    raise XGroupReferenceError(_("Reference target not found: {path}").format(path=path))

Adding a new language:

1. Run ``xgettext`` to update the .pot template::

       xgettext -d zarr_xgroup -o zarr_xgroup/locale/zarr_xgroup.pot \\
           zarr_xgroup/*.py zarr_xgroup/**/*.py

2. Copy the template to a new locale directory::

       mkdir -p zarr_xgroup/locale/<lang>/LC_MESSAGES
       cp zarr_xgroup/locale/zarr_xgroup.pot \\
          zarr_xgroup/locale/<lang>/LC_MESSAGES/zarr_xgroup.po

3. Edit the .po file to fill in the translated strings.

4. Compile::

       msgfmt zarr_xgroup/locale/<lang>/LC_MESSAGES/zarr_xgroup.po \\
           -o zarr_xgroup/locale/<lang>/LC_MESSAGES/zarr_xgroup.mo
"""

from __future__ import annotations

import gettext
import importlib.resources


def _get_translator() -> gettext.NullTranslations:
    try:
        localedir = importlib.resources.files("zarr_xgroup") / "locale"
        return gettext.translation(
            domain="zarr_xgroup",
            localedir=str(localedir),
            languages=None,  # use system locale
        )
    except FileNotFoundError:
        # No compiled .mo files present; fall back to identity translation.
        # This is the normal state during development before .mo files are
        # compiled, and for the English base locale.
        return gettext.NullTranslations()


_translator = _get_translator()

#: Translation function. Import and use as ``from zarr_xgroup.i18n import _``
_ = _translator.gettext
