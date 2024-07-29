# README

## TODOs

- make sure all op fn args and kwargs become context in the scope. no extra kwargs should be supplied
- define the op fn separately from registering it with a handler
  - the op fn will still need a @operator decorator in order to wrap it in a traced scope
  - then register
- make a shorthand decorator for overloading operators based on input arg types
- implement various select algos "beam" | "greedy" | "breadth" | "depth" | ... (Right now we just do greedy with replacement which is pretty terrible)

Algorithms:

- get a single item, depth=1
  - engine.search
- set a single item
  - assign operator (=)

- locate an exact address, arbitrary depth
  - engine.locate
- get a value at an exact address, arbitrary depth
  - engine.locate(...).get
- set a value at an exact address, arbitrary depth
  - engine.locate(...).set
- get a dynamically generated value at an exact address, arbitrary depth
  - engine.locate(...).get -> engine.encode -> engine.transform
  - engine.locate(...).get -> engine.modify
- set a dynamically generated value at an exact address, arbitrary depth
  - engine.transform -> engine.decode ->  engine.locate(...).set
  - engine.modify -> engine.locate(...).set

- locate multiple exact address, arbitrary depth
  - engine.locate -> engine.locate -> ... (repeat)
- get a value at multiple exact addresses, arbitrary depth
  - engine.locate(object, ..., top_k>1).foreach.get(object, ...)
- set a value at multiple exact addresses, arbitrary depth
  - engine.locate(object, ..., top_k>1).foreach.set(object, value, ...)
- get a dynamically generated value at multiple exact addresses, arbitrary depth
  - engine.query(object, ...)
  - engine.locate(object, ..., top_k>1).foreach.get(object, ...) -> engine.encode
- set a dynamically generated value at multiple exact addresses, arbitrary depth
  - engine.locate(object, ..., top_k>1).foreach.set(engine.transform(it, ...), ...)

- get a value at a dynamic address, arbitrary depth
  - engine.query(query=object, max_rounds=1, ...)
- set a value at a dynamic address, arbitrary depth
  - engine.modify(key=..., value=object, max_rounds=1, ...)
- get a dynamically generated value at a dynamic address, arbitrary depth
  - engine.query(query_latent=object, max_rounds=1, ...)
- set a dynamically generated value at a dynamic address, arbitrary depth
  - engine.modify(key_latent=..., value_latent=object, max_rounds=1, ...)

- get a value at multiple dynamic addresses, arbitrary depth
- set a value at multiple dynamic addresses, arbitrary depth
- get a dynamically generated value at multiple dynamic addresses, arbitrary depth
- set a dynamically generated value at multiple dynamic addresses, arbitrary depth

```python
select(
  engine,
  target: Any,
  query_latent: Latent = None,
  search_strategy: Literal["beam", "greedy", "breadth", "depth"] = "greedy",
  top_k=1,
  top_p=1.0,
  ...
) -> object:
  ...

query(
  engine,
  target: Any,
  query_latent: Latent = None,
  search_strategy: Literal["beam", "greedy", "breadth", "depth"] = "greedy",
  top_k=1,
  top_p=1.0,
  max_rounds=1,
  ...
) -> Latent:
  while not done:
    engine.select next kv pair
    update latent state

modify(
  engine,
  target: Any,
  key: Locator = None,
  key_latent: Latent = None,
  value=None,
  value_latent=None,
  max_rounds=1,
  ...
):
  while not done:
    engine.select next kv pair
    engine.update latent state
```
