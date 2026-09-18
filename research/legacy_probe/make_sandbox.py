"""Copy the committed legacy `tensorcode` package into a sandbox and apply the minimum shims to import TCIR.

    python legacy_probe/make_sandbox.py SANDBOX_DIR

Every shim is listed here. None changes what `parse_node` produces for dataclasses,
containers, or primitives; they only get the modules to import.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
subprocess.run(f"git -C {REPO} archive HEAD tensorcode | tar -x -C {out}", shell=True, check=True)
pkg = out / "tensorcode"

SHIMS = []


def shim(path: str, old: str, new: str, why: str) -> None:
    p = pkg / path
    text = p.read_text()
    assert old in text, f"shim target missing in {path}: {old[:60]!r}"
    p.write_text(text.replace(old, new, 1))
    SHIMS.append((path, why))


# 1. `tensorcode.internal.utils.functional` was deleted in 1159261; its historical `polymorphic`
#    raises TypeError on first registration. Re-implement its documented dispatch intent.
(pkg / "internal/utils/functional.py").write_text(
    '''def polymorphic(fn):
    overrides = []
    def wrapper(value, *args, **kwargs):
        for _, _, cond, impl in sorted(overrides, key=lambda o: (-o[0], o[1])):
            if cond(value):
                return impl(value, *args, **kwargs)
        return fn(value, *args, **kwargs)
    def register(cond, /, priority=0):
        def deco(impl):
            overrides.append((priority, len(overrides), cond, impl))
            return impl
        return deco
    wrapper.register = register
    return wrapper
'''
)
SHIMS.append(("internal/utils/functional.py", "module missing since 1159261; historical polymorphic() is itself broken"))
# 2. pydantic.py imports `consts` from the wrong package.
(pkg / "internal/utils/consts.py").write_text("from tensorcode.internal.consts import *\n")
SHIMS.append(("internal/utils/consts.py", "re-export: pydantic.py imports tensorcode.internal.utils.consts, which never existed"))
# 3. decorator order: abstractmethod cannot wrap a property object.
shim("internal/tcir/nodes.py", "    @abstractmethod\n    @property\n    def python_value(self): ...", "    @property\n    @abstractmethod\n    def python_value(self): ...", "abstractmethod applied to a property raises AttributeError at import")
# 4. pydantic 2.5 cannot build a schema for `complex`.
shim("internal/tcir/nodes.py", "class Node(BaseModel):\n", "class Node(BaseModel):\n    model_config = {'arbitrary_types_allowed': True}\n", "ComplexNumberNode.value: complex has no pydantic schema")
# 5. recursive `Tensor` union recurses forever during schema generation.
shim("internal/tcir/nodes.py", "class TensorNode(AtomicValueNode):\n    value: Tensor", "class TensorNode(AtomicValueNode):\n    value: Any", "Tensor = Union[Number, list['Tensor']] | ndarray recursion")
# 6. cached_property has no .setter.
shim("internal/tcir/nodes.py", "    @type.setter\n    def type(self, value: TypeNode):\n        self.types[0] = value\n", "", "OptionalTypeNode uses @cached_property ... .setter")

print(f"sandbox: {out}")
for path, why in SHIMS:
    print(f"  shim {path}: {why}")
