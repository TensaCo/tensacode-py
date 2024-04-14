_DEFAULT_GLOBAL_NAMESPACE = {}


def unique(name: str, namespace=_DEFAULT_GLOBAL_NAMESPACE, scope_delimitor="/"):
    if name not in namespace:
        namespace[name] = 0
        return name
    else:
        namespace[name] += 1
        return name + scope_delimitor + str(namespace[name])
