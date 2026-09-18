"""A body whose perception is a provider: the same agents, a different way of seeing.

    ui = ProvidedBrowser(page, WebDomProvider(), episode="e1")        # what agents use today
    ui = ProvidedBrowser(page, VisionProvider(), episode="e1")        # pixels only
    ui = ProvidedBrowser(page, FusedProvider([WebDomProvider(), VisionProvider()]), episode="e1")

``Browser`` keeps doing the acting (clicks, typing, verified focus). Only ``observe``
changes. The last raw scene stays on ``last_scene``, so a task can look at sources,
confidence and conflicts rather than the flattened ``Screen``.
"""

from __future__ import annotations

import io
import time
from typing import Callable

from ..browser import Browser, Screen
from .protocol import PerceivedScene, Provider, Target


class ProvidedBrowser(Browser):
    def __init__(self, page: object, provider: Provider, *, episode: str, settle_ms: int = 0, adapt: Callable[[Screen], Screen] | None = None, on_step: Callable[[], None] | None = None) -> None:
        super().__init__(page, episode=episode, on_step=on_step)
        self.provider, self.settle_ms, self.adapt = provider, settle_ms, adapt
        self.last_scene: PerceivedScene | None = None
        self.perception_ms: list[float] = []

    def grab(self):
        """A screenshot as an RGB array (providers that need pixels call this)."""
        import numpy as np
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(self.page.screenshot(type="png"))).convert("RGB"))

    def target(self) -> Target:
        return Target(page=self.page, grab=self.grab)

    def observe(self, *, settle_ms: int | None = None) -> Screen:
        t0 = time.perf_counter()
        scene = self.provider.perceive(self.target())
        self.last_scene = scene
        screen = scene.to_screen()
        if self.adapt:
            screen = self.adapt(screen)
        dt = time.perf_counter() - t0
        self.perception_ms.append(dt * 1e3)
        self.stats.browser_s += dt
        self.stats.observations += 1
        return screen
