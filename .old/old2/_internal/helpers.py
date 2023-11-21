class Lockable
    # TODO: I want to rename everything here to is_enabled, CanDisable, etc.
    _default_unlocked: bool = True
    _unlocked: dict[str, bool] = {}
    def unlock(self, key=None):
        self._set_unlocked(key, True)
    def lock(self, key=None):
        self._set_unlocked(key, False)
    def _get_unlocked(self, key):
        if key is None:
            return self._default_unlocked
        else:
            return self._unlocked[key]
    def _set_unlocked(self, key, val):
        if key is None:
            _default_unlocked = val
        else:
            self._unlocked[key] = val

    @staticmethod
    def if_unlocked(key_or_fn, fn=None):
        assert key_or_fn is (str, Callable), \
            f'Invalid type for `key_or_fn`. Expecting `str` or `Callable`, but got {type(key_or_fn)}'
        match key_or_fn:
            case _ is str
                key = key_or_fn
                fn = fn
            case _ is Callable
                key = None
                fn = key_or_fn
        @wraps(fn)
        def _fn(self, *args, **kwargs):
            assert self is Lockable, \
                "self is not `Lockable`. Did you decorate `if_unlocked` on a method that doesn't belong to a Lockable?"
            if not self._get_unlocked(key): return
            return fn(self, *args, **kwargs)

if_unlocked = Lockable.if_unlocked

def join(vals, delim=', '):
    match vals:
        case _ is (tuple, list):
            return delim.join(vals)
        case _ is dict:
            return vals.items() \
                |> map$((key, val)->f'{key}={val}', ?) \
                |> delim.join$(?)
    else:
        raise TypeError()
