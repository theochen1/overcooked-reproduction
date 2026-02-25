try:
    from gym.envs.registration import register as _register
except Exception:
    _register = None


def register(*args, **kwargs):
    if _register is None:
        return None
    try:
        return _register(*args, **kwargs)
    except Exception:
        return None
