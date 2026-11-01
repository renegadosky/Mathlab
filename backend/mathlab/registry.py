import inspect
from . import numeric, programming, physics

MODULES = {"physics": physics, "programming": programming, "numeric": numeric}


def describe():
    out = {}
    for name, mod in MODULES.items():
        items = []
        for m, f in sorted(mod.METHODS.items()):
            ps = [{"name": p.name, "required": p.default is inspect.Parameter.empty,
                   "default": None if p.default is inspect.Parameter.empty else p.default}
                  for p in inspect.signature(f).parameters.values()]
            items.append({"name": m, "doc": (inspect.getdoc(f) or "").split("\n")[0], "params": ps})
        out[name] = items
    return out
