def render_function_call(fn, args, kwargs):
    fn_name = fn.__name__
    arg_strings = []

    # Render positional arguments
    for arg in args:
        arg_strings.append(repr(arg))

    # Render keyword arguments
    for key, value in kwargs.items():
        arg_strings.append(f"{key}={repr(value)}")

    # Combine all arguments
    args_str = ", ".join(arg_strings)

    # Construct the function call string
    return f"{fn_name}({args_str})"
