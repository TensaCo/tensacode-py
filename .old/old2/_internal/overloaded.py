## FullArgSpec(
##   args, varargs, varkw, defaults, 
##   kwonlyargs, kwonlydefaults, 
##   annotations)

class ArgType(Enum):
    POSITIONAL_ONLY = 0
    POSITIONAL_OR_KEYWORD = 1
    VAR_POSITIONAL = 2
    KEYWORD_ONLY = 3
    VAR_KEYWORD = 4
data Argument(name: str, val: any, annotation: any = None,
              argtype: ArgType = ArgType.POSITIONAL_OR_KEYWORD)

@memoized
def get_arguments(fn, supplied_args, supplied_kwargs, ignore_extra=False) -> Agrument[]:
	arguments = []
	fullargspec = inspect.getfullargspec(fn)	
	remaining_args = fullargspec.args
  # extract kwargs first, since they may eat up some of the args
	for kw, val in supplied_kwargs.items():
		if kw in fullargspec.kwonlyargs:
			arguments.append <| Argument(kw, val, annotations[kw], ArgType.KEYWORD_ONLY)
		elif kw in remaining_args:
			remaining_args.pop(kw)
			arguments.append <| Argument(kw, val, annotations[kw], ArgType.POSITIONAL_OR_KEYWORD)
		elif len(remaining_args) == 0 and fullargspec.varargs is not None:
			arguments.append <| Argument(kw, val, annotations[fullargspec.varargs], ArgType.VAR_POSITIONAL)
		elif fullargspec.varkwargs is not None:
			arguments.append <| Argument(kw, val, annotations[fullargspec.varkwargs], ArgType.VAR_KEYWORD)
		else:
      if not ignore_extra:
  			raise Error('Arguments supplied to not match signature')
  # now extract args
	for val in supplied_args:
		if len(remaining_args) > 0:
			arguments.append <| Argument(remaining_args.pop(0), val, annotations[kw], ArgType.POSITIONAL_OR_KEYWORD)
		elif len(remaining_args) == 0 and fullargspec.varargs is not None:
			arguments.append <| Argument(fullargspec.varargs, val, 
                                   annotations[fullargspec.varargs],
                                   ArgType.VAR_POSITIONAL)
		else:
      if not ignore_extra:
  			raise Error('Arguments supplied to not match signature')
	# add defaults
	argnames = map(.name, arguments)
	fullargspec.args
		|> dropwhile(-> _ in argnames)
		|> map$(-> Argument(_, fullargspec.defaults[_], fullargspec.annotations[_]))
		|> arguments.extend
	fullargspec.kwonlyargs
		|> dropwhile (-> _ in argnames)
		|> map$(-> Argument(_, fullargspec.kwonlydefaults[_], fullargspec.annotations[_]))
		|> arguments.extend
  
  return arguments

def make_accept_extra_args(fn):
  # better if we could manually enable varargs and varkwargs
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    arguments = get_arguments(fn, args, kwargs)
    return fn(
      *(a.val for a in arguments if a.argtype < ArgType.KEYWORD_ONLY),
      **{a.name: a.val for a in arguments if a.argtype >= ArgType.KEYWORD_ONLY})
  return wrapper

def attempt(fn, args, kwargs):
  try:
    return fn(*args, **kwargs)
  except:
    return None

def arg_annotation_match(val, annotation, current_arguments=None):
  # assign value from 0 to 1, if val is a type of annotation
  # (1 is a perfect match and approaching 0 is an infinite chain of subclasses between val and annotation)
  if val is annotation:
    num_parents = 0
    # TODO recurse over annotation.__subclassess__ and find the shortest path
    return 1/(num_parents)
  # also support predicate annotations
  elif annotation is Predicate:
    predicate = make_accept_extra_args(annotation)
    current_args = [], current_kwargs = {}
    for a in current_arguments or []:
      if a.argtype < ArgType.KEYWORD_ONLY: current_args.append(a.val)
      else: current_kwargs[a.name] = a.val
    return predicate(*current_args, **current_kwargs)
  # also handle `like` annotations
  elif annotation is like:
    return 0
  else:
    return -inf

def total_arg_annotation_match(arguments) =
    0 if len(arguments) == 0
    else range(len(arguments))
    |> map$(i -> arguments[:i+1])
    |> map$(-> arg_annotation_match(_[-1].val, _[-1].annotation, _))
    |> sum

def overloaded(self, fn):
    overloads = [(fn, const(True))]

    def overload(condition_or_method=None, method=None):
      condition, method = unscramble_optional_first(condition_or_method, method)
      _condition = make_accept_extra_args(condition) if condition is not None else const(True)
      _condition = bool .. _condition
      overloads.append <| (method, _condition)
      return method

    @wraps(fn)
    def _fn(*args, **kwargs):
      return overloads
        # first filter out overloads that don't accept the given arguments
        |> map$(->(**_, arguments=attempt(get_arguments, _.fn, args, kwargs)))
        |> filter$(-> _.arguments is not None)
        # next filter using the condition
        |?> filter$(-> _.condition(*args, **kwargs))
        # next sort by the distance between the arguments and the annotations
        |?> map$(->(**_, distance=total_arg_annotation_match(_.arguments)))
        |> sorted$(-> _.distance)
        # finally, call the first overload
        |> $[0]
        |> .fn(*args, **kwargs)

    setattr(_fn, 'overload', overload)
    setattr(_fn, 'overloads', overloads)
    return _fn