'''
encode (obj, ctx) -> Tensor:
  encode a python object into a tensor

retrieve (obj, num=1, depth_limit=None, ctx) -> objs:
  retrieves `num` objects from inside `obj` at arbitrary depth
select (items, num=1, ctx) -> objs:
  special case of retrieve for depth_limit=1 items where items is a finite iterable
sort (items, ctx) -> items:
  aggregates global state, then assigns energy to each item and sorts

decode (tensor, Type, ctx) -> obj:
  decodes a tensor to an object of a given type
  custom decoders for atomic types
  recursive creation of objects of arbitrary types
get (Type, allowed, disallowed, rules, num=1, depth_limit=None, ctx) -> objs:
  retrieves `num` objects of a given type from the options available in
  rules (allowed and disallowed are convenience parameters for rules).
  Uses exec, but with a limited statement vocabulary.

modify (obj, depth_limit=None, ctx, clone=False) -> obj'?
  recursively keeps|modifies|replaces an object.
  Modification only applies to mutable containers and composite objects,
  and it can add, remove, rename, duplicate, or replace items.
call (fn, ctx) -> obj:
  chooses the args on the function (using get)
  (`get`/`exec` both use `call` for FUNCTION_CALL statements)
exec(tensorbytecodes, params?, ctx) -> obj?
  selects and interprets a sequence of tensorbytecodes (select, call, return)
  maybe just use Python bytecodes? no, that's too complex

'''