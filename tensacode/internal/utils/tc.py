def loop_until_done(limit: int | None = None, prompt: str = "Continue?"):
    # TODO: put this into the ops
    iteration = 0
    while True:
        if limit is not None and iteration >= limit:
            raise StopIteration("Iteration limit reached")
        if not engine.decide(prompt):
            raise StopIteration("Engine decided to stop")
        yield iteration
        iteration += 1
