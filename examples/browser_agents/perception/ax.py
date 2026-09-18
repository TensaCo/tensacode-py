"""macOS Accessibility (AX) provider (stub).

The interface a real implementation fills in, using ``pyobjc`` (``ApplicationServices``):

* ``AXUIElementCreateApplication(pid)`` per application, or ``AXUIElementCreateSystemWide()``;
* walk ``kAXChildrenAttribute``, reading ``kAXRoleAttribute``, ``kAXTitleAttribute`` /
  ``kAXDescriptionAttribute`` / ``kAXValueAttribute``, ``kAXPositionAttribute`` +
  ``kAXSizeAttribute`` for the box, ``kAXEnabledAttribute``, ``kAXFocusedAttribute``;
* map ``AXButton`` -> "button", ``AXTextField``/``AXTextArea`` -> "textbox", ``AXCheckBox``
  -> "checkbox", ``AXPopUpButton`` -> "combobox", ``AXRadioButton`` -> "option",
  ``AXStaticText`` -> text blocks, ``AXWindow``/``AXSheet`` -> regions;
* the hit point is the box centre; AX has no hit test, so occlusion must come from window
  order (``kAXWindowsAttribute`` is front-to-back) or from vision;
* requires the user to grant Accessibility permission to the running process, which a real
  implementation should detect (``AXIsProcessTrusted``) and report through ``available()``.
"""

from __future__ import annotations

from .protocol import PerceivedScene, Target


class AxProvider:
    name = "ax"
    reliability = 0.9

    def available(self) -> bool:
        import sys

        if sys.platform != "darwin":
            return False
        try:
            from ApplicationServices import AXIsProcessTrusted  # noqa: F401
        except Exception:
            return False
        return bool(AXIsProcessTrusted())

    def perceive(self, target: Target) -> PerceivedScene:
        raise NotImplementedError("macOS AX provider is a stub: see this module's docstring for the mapping it needs")
