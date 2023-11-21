import inspect


data Argument(name: str, val: any, annotation: any = None)

def get_arguments(fn, supplied_args, supplied_kwargs) -> Agrument[]:
	arguments = {}
	fullargspec = inspect.getfullargspec(fn)	
	remaining_args = fullargspec.args
    remaining_kwonlyargs = fullargspec.kwonlyargs

    # shift through kwargs before args, since they might eat up some positional parameters
	for kw, val in supplied_kwargs.items():
		if kw in fullargspec.kwonlyargs:
			remaining_kwonlyargs.pop(kw)
			arguments.append(Argument(kw, val, annotations[kw]))
		elif kw in remaining_args:
			remaining_args.pop(kw)
			arguments.append(Argument(kw, val, annotations[kw]))
		elif len(remaining_args) == 0 and fullargspec.varargs is not None:
			arguments.append(Argument(kw, val, annotations[fullargspec.varargs]))
		elif fullargspec.varkwargs is not None:
			arguments.append(Argument(kw, val, annotations[fullargspec.varkwargs]))
		else:
			raise Error('Arguments supplied to not match signature')
	for val in supplied_args:
		if len(remaining_args) > 0:
			arguments.append(Argument(remaining_args.pop(0), val, annotations[kw]))
		elif len(remaining_args) == 0 and fullargspec.varargs is not None:
			arguments.append
				<| Argument(fullargspec.varargs, val, annotations[fullargspec.varargs])
		else:
			raise Error('Arguments supplied to not match signature')
	# add defaults
	argnames = arguments |> map$(.name)
	remaining_args
		|> dropwhile$(_ in argnames)
		|> map$(Argument$())
		|> arguments.extend
	# TODO think about what to do here