"""The same small object graph, used for both the legacy TCIR probe and the proposed records."""

from dataclasses import dataclass, field


@dataclass
class Site:
    label: str  # the legacy parser crashes on a field called `name` (see probe), so the shared fixture avoids it
    timezone: str


@dataclass
class Printer:
    asset_id: str
    model: str
    site: Site
    floor: int
    status: str


@dataclass
class Ticket:
    id: str
    printer: Printer
    reporter: str
    text: str
    priority: str
    tags: list[str] = field(default_factory=list)


@dataclass
class SiteWithName:
    name: str
    timezone: str


@dataclass(eq=False)
class CyclicSite:
    label: str
    printers: list = field(default_factory=list)


@dataclass(eq=False)
class CyclicPrinter:
    asset_id: str
    site: CyclicSite


def tickets():
    hq = Site("HQ", "America/Chicago")
    p3 = Printer("PRN-3", "LaserJet M507", hq, 2, "online")
    t1 = Ticket("T-101", p3, "dana", "P3 jams on every duplex job", "normal", ["hardware", "jam"])
    t2 = Ticket("T-102", p3, "lee", "P3 shows offline from floor 2 laptops", "high", ["network"])
    return t1, t2


def cyclic_site():
    s = CyclicSite("HQ")
    s.printers.append(CyclicPrinter("PRN-3", s))
    return s
