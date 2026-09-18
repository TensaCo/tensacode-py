"""Windows UI Automation provider (stub).

The interface a real implementation fills in, using ``comtypes`` with the UIAutomationCore
type library (or the ``uiautomation`` package):

* ``IUIAutomation.ElementFromHandle`` for a window handle, or ``GetRootElement`` for the desktop;
* ``FindAll(TreeScope_Subtree, CreateTrueCondition())`` with a ``CacheRequest`` for
  ``Name``, ``ControlType``, ``BoundingRectangle``, ``IsEnabled``, ``IsOffscreen``,
  ``ToggleState``, ``ValuePattern.Value``, ``AutomationId``;
* map ``UIA_ButtonControlTypeId`` -> "button", ``EditControlTypeId`` -> "textbox",
  ``CheckBoxControlTypeId`` -> "checkbox", ``ComboBoxControlTypeId`` -> "combobox",
  ``TabItemControlTypeId`` -> "tab", ``TextControlTypeId`` -> text blocks, ``WindowControlTypeId``
  -> regions; ``ClickablePoint`` (``GetClickablePoint``) is the hit point, and None when the
  call fails, which is how UIA says "not hittable";
* provenance locator: the ``AutomationId`` or the runtime id; reliability ~0.95.

Nothing here runs on Linux, so it reports unavailable rather than pretending.
"""

from __future__ import annotations

from .protocol import PerceivedScene, Target


class UiaProvider:
    name = "uia"
    reliability = 0.95

    def available(self) -> bool:
        try:
            import comtypes.client  # noqa: F401
        except Exception:
            return False
        import sys

        return sys.platform == "win32"

    def perceive(self, target: Target) -> PerceivedScene:
        raise NotImplementedError("Windows UI Automation provider is a stub: see this module's docstring for the mapping it needs")
