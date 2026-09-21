the conceptual schemas we have been codifying are a very, very ugly hardcoded mess. i was hoping we'd get something much cleaner in the python purist's PoV and more general from the cognitive primitives toolbox PoV. something like:

```python
from idc_where import exec_parallel # made up

from tensorcode.vec import Latent
from tensorcode.vec.decide import Decider
from tensorcode.vec.decide import Decider
from tensorcode.vec.encode.text import TextEncoder
from tensorcode.vec.encode.images import ImageEncoder
from tensorcode.vec.decode.text import TextDecoder

class SimpleChatbot:

    def __init__(...):
        self.text_encoder = TextEncoder(...)
        self.image_encoder = ImageEncoder(...)
        # TODO: initialize all the other operators, each of them is ultimately some combination of learned transforms and programmatic de/re/structuring
        
        self.objective = Latent.zero()
        self.context = dict[str, Latent]() # idk, lol
        self.memory = DB[Latent]() # idk, lol
    
    def submit(prompt: str, attached_images: Image[]):
        current_objective_enc = self.text_encoder.encode(self.objective)

        prompt_enc = self.text_encoder.encode(prompt, objective=self.objective_enc, context=self.context)

        prelim_update_objective_latent = self.integrate_prompt_enc_with_objective.forward(prompt_enc, ) # shared backbone for decide and speculative update
        if self.decide_update_objective(text_enc, context={"objective": objective, **self.context})
            self.objective = self.update_objective.forward(slef.objective)
        
        image_encs = exec_parallel(self.image_encoder, attached_images, context=self.context)

        ...


server = FastAPI(...)

# TODO: serve the javascript chatbot frontend

@server.get("/submit")
def submit(prompt: str, attachments: File[]):
    subjective_perception = text_encoder(prompt)
    
```

forgive me, thisis very incomplete but it begins to sketch out the idea

the basic idea is that the architecture should be expressed as a small set of general composable primitives—encoders, a cognitive core, agents, and interfaces—rather than accumulating domain-specific schemas and hardcoded wiring.

this was actually sketched out pretty clearly in the parent tensorcode repo and the 2022 docs of this repo but got completely forgotten in the chaos.

we want to be able to use any of the cognitive operations with either vectors as the intermediate representaion, llm messages as the intermediate representation, or symbol graphs as the intermediate representation. and then its just `tensorcode.<IR>.<OP> for all the ops. this modularizes and clearly establishes each of the conceptual primitives we want to use

also most of the ops should share a common interface. perhaps torch style .forward(kwargs)? perhaps <op_instance>(primary_arg, *, context: dict[str, T_Enc (different for vec, llm, graph)])? perhaps something else?

also, i think the agent harness not the ops should all be repsonsible for logging the inputs and outputs to diffierentaible graphs in each step and then be responsible for differentiating. but perhaps tensorcode can provide utils for this. because we want users of tensorcode to be able to train their software / make their software learn from its experience

the difference between what i am proposing and the current direction this repo has gone is so large that you should make a private copy of the state of this repo at `JacobFV/old-tensorcode-2026-09-20` and then aggressively delete all (not deprecate, not move) the shit we added in this official repo.