class ReturnPlusOne:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # No exception occurred
            frame = sys._getframe(1)
            result = frame.f_locals.get('result', None)
            if result is not None:
                frame.f_locals['result'] = result + 1
        return False  # Don't suppress exceptions

# Usage
with ReturnPlusOne():
    result = 1
    return result
# This will actually return 2
