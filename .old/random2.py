import sys

class ReturnPlusOne:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            frame = sys._getframe(1)
            result = frame.f_locals.get('result', None)
            if result is not None:
                frame.f_locals['result'] = result + 1
        return False

def main():
    with ReturnPlusOne():
        result = 1
        return result

if __name__ == "__main__":
    print(main())
