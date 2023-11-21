from tensacode._utils.export import export, ContextManaged
from tensacode._utils.ledger import Ledger
from tensacode._utils.fp import Test
import tensacode as tc
from tensacode.utils.annotations import like, enc, encode_args, overloaded

import ivy
ivy.set_framework('numpy') # need to set a framework for ivy

class NNModel(Model):

    dim: int

    @export
    @ledger.ledgered
    @memoize # TODO: useful for internal repeated encoding of objects
    @overloaded
    @encode_args
    def encode(self,
        object: object,
        depth_limit: int = 10,
        sample_rounds: int = 4,
        sample_heads: int = 4,
        max_early_exit_depth: int = 1,
        depth: int = 0,
        context: any = None,
        instructions: enc[str] = None,
        focus: enc[str] = None,
        examples: list = None):
        """Default encode function. Encodes objects
        
        The main difference between encoding objects and dicts
        is that with objects, we only what what keys are available, 
        but their values aren't necesarily known until after lookup.
        dicts don't have the ability to do magic like that. 
        """
        name = getattr(object, '__name__', '')
        enc = focus ?? self.encode(name)

        samples_it = range(sample_rounds)
        if depth<=max_early_exit_depth:
            question = f'Continue sampling `{name}`?' if name else 'Continue sampling?'
            samples_it = self.takewhile(question, samples_it, ()->self.ledger.read())
        for _ in samples_it:
            # TODO are context, examples, and instructions really just LLM-specific args? It seems so
            keys = self.select(dir(object), n=sample_heads, ...)
            vals = [getattr(self, key) for key in keys] 
            enc = self.merge(enc, *vals)
            enc = self._encode_process(enc)
        return enc


    @encode.overload
    @encode_args
    def encode_bool(self, input: bool, **kw):
      """Encode boolean input"""
      tensor = ivy.full((self.dim, ), fill_value= input, dtype= bool).astype(ivy.int32)

      return tensor

    @encode.overload
    @encode_args
    def encode_int(self, input: int, **kw):
        args = locals()
        args.pop('self')
        return self.encode_float(step=1., **args)

    @encode.overload
    @encode_args
    def encode_float(self, input: float, step: float = 1e-2, **kw):
        """Encode float using positional embeddings"""
        
        # generate frequencies for PE
        k = ivy.arange(start=0, stop=self.dim//2, step=1)
        frequencies = 1 / (base**(2*k/self.dim))
        
        # make 1 step = 1 unit
        input = input / step
        
        # compute PE
        pe_sin = sin(2*pi*frequencies*input)
        pe_cos = cos(2*pi*frequencies*input)
        pe = ivy.concat([pe_sin, pe_cos], axis=-1)
        
        # handle odd depth case
        if self.dim % 2:
            # pad with extra 0
            pe = ivy.concat([pe, ivy.zeros_like(pe)[..., :1]], axis=-1)
        
        return pe

    @encode.overload
    @encode_args
    def encode_str(self,
        object: str,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        # try parsing it as a literal
        # if that doesn't work, try parsing as a variable name
        # if that doesn't work, try parsing as a zero-argument function or iterator
        # otherwise, build semantic embedding of the string
        TODO()

    @encode.overload
    @encode_args
    def encode_ordered_collection(self,
        object: list|tuple,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        TODO()

    @encode.overload
    @encode_args
    def encode_unordered_collection(self,
        object: set|frozenset,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        TODO()

    @encode.overload
    @encode_args
    def encode_mapping(self,
        object: dict|Mapping,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        TODO()

    @encode.overload
    @encode_args
    def encode_code(self,
        object: code,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        TODO()

  ...
