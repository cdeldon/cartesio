try:
    import numba

    jit = numba.njit

except ModuleNotFoundError:

    import warnings
    from functools import wraps

    warnings.warn("Numba module not found. Performance will be degraded")

    def jit(signature_or_function=None, **numba_kwargs):
        if signature_or_function is None:

            def decorator(signature_or_function_inner):
                @wraps(signature_or_function_inner)
                def inner(*args, **kwargs):
                    setattr(inner, "py_func", signature_or_function_inner)
                    return signature_or_function_inner(*args, **kwargs)

                return inner

            return decorator
        else:
            @wraps(signature_or_function)
            def inner(*args, **kwargs):
                setattr(inner, "py_func", signature_or_function)
                return signature_or_function(*args, **kwargs)

            return inner

__all__ = [
    "jit"
]
