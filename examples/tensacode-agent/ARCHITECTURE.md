# Agent

## TODO

- Make relation nodes for the property defined relationships. This will enable me to compute the hop distance between arbitrary pairs of nodes which is a measure of intelligence

- Figure out the task tracking

- Implement emotional/motivational states for determining the weighing given to relevant thoughts. arousal/valency/intensity

- Integrate [this](https://github.com/zenbase-ai/core) for prompt engineering (will need to dynamically compute the prompts)


## Abilities

- Task tracking
- RAG
- Introspection
  - logs
  - code (put it in the graph)
  - git logs (put it in the graph)
  - codebase (put the files, modules, functions, classes, etc. in the graph)
  - callstack (put the callstack in the graph)
- self-modification
  - hot reloading/dispatching (will require putting everything in a separate file)
  - pre-deployment testing
  - failsafe recovery
  - benchmarking and evolution
  - compute and api resources (put the resources in the graph)
  - llm calls (put the calls in the graph)

## Integrations

- Raw Python
- Payman
- [Twitter client](https://github.com/RubyResearch/agent-twitter-client)