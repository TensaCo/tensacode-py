# must use Python 3.10+
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from types import Mapping

from tensacode._utils.export import export, ContextManaged
from tensacode._utils.ledger import Ledger
from tensacode._utils.fp import Test
import tensacode as tc
from tensacode.utils.annotations import like, enc, encode_args, overloaded


@ContextManaged
class Model(tc.utils.annotations.SupportsEncode[any]):
    """I don't want to put so many methods in the same file, but I don't see a better
    way to group all the call signatures togethor. Also, many of these methods call each other.
    Finally, it just makes sense for LLM-based methods to all enter context simultaneously.

    I should group the items in this class into sub-class/modules so it looks like a real module
    even though its a class
    """

    # TODO: in the top-level tensorcode.whatever methods, you should be able to pass in an arg
    # to indicate that you'd like to use fresh weights for this operation. Maybe model_name, and
    # if the name is new, a new set of parameters are initialized
    # actually, most function calls contain some parameters that are particular to the shape
    # of the inputs. 

    ledger = Ledger()

    # this is actually the NN implementation
    HPARAMS = {
        'c_q_name': 1.0
        'c_q_instr': 1.0
        'c_q_focus': 1.0
    }
    def _build_query(self, name: enc[str], instructions: enc[str]?, focus: enc[str]?):
        return ((name, self.HPARAMS['c_q_name']), 
                (instructions, self.HPARAMS['c_q_instr']),
                (focus, self.HPARAMS['c_q_focus']))
        |> dropwhile$(x, c -> x is None, ?) 
        |> sum
    self._encode_process(enc)
    
    @export
    @ledger.ledgered
    @memoize # TODO: useful for internal repeated encoding of objects
    @overloaded
    @encode_args
    def merge(self, *encs: enc[]):
        # eg, for NN, just return sum(args), maybe with learnable weights
        # for LLM's, return newline-separated (for long text) or comma separated (short text)
        raise NotImplemented("Subclasses should implement this")

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
        is that with objects, we only know what keys are available,
        but their values aren't necesarily known until after lookup.
        dicts don't have the ability to do magic like that.
        """
        name = getattr(object, '__name__', '')
        enc = focus or self.encode(name)

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
    def encode_bool(self,
        obj: bool,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        subclasses_should_implement()

    @encode.overload
    @encode_args
    def encode_int(self,
        object: int,
        bounds: (int, int) = None,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        args = locals()
        args.pop('self')
        return self.encode_float(epsilon=1., **args)

    @encode.overload
    @encode_args
    def encode_float(self,
        object: float,
        bounds: (float, float) = None,
        epsilon: float = 0.01,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        # use positional embeddings in NN implementation
        # use adjectives in LLM implementation
        subclasses_should_implement()

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
        subclasses_should_implement()

    @encode.overload
    @encode_args
    def encode_ordered_collection(self,
        object: list|tuple,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        subclasses_should_implement()

    @encode.overload
    @encode_args
    def encode_unordered_collection(self,
        object: set|frozenset,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        subclasses_should_implement()

    @encode.overload
    @encode_args
    def encode_mapping(self,
        object: dict|Mapping,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        raise NotImplementedError("Subclasses should implement this")

    @encode.overload
    @encode_args
    def encode_code(self,
        object: code,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        TODO()

    
    # in child classes, directly write
    # @encode.overload(x -> x is Number)
    # def encode_number(self, number): ...
    # this is possible since python classes are basically a module

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def create(description: R[str] | like[str], type: type|R[type], max_depth=1, recursions=1):
        pass
        """tc.create('the highest number in the ', int)"""

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    # create bool with prompt engineered for decision:
    def decide(question: str):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def write(prompt: str):  # create str with prompt engineered for writing:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def define(description: str):  # define class:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def get(obj: any, n: int | str, description: str, max_depth=0) -> any:
        pass

    @export
    @overloaded
    @encode_args
    def put(obj: any, val: any, description: str, max_depth=0):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def sort(items: list, criteria: str) -> list:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def similarity(a: any, b: any) -> float:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def segment(obj: any, ...) -> any:
        """
        for getting masks from images and highlights from text 
        Useful for breaking long documents into smaller pieces
        """
        pass

    @export
    @overloaded
    @encode_args
    def closest(item: T, choices: set[T]) -> T:  # find the closest match:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def identify_anomalies(examples: set) -> set:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def identify_patterns(examples: set, n=None, min=None, max=None, recognition_threshold=None) -> Pattern:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def identify_isomorphisms(examples: set) -> set[tuple[Pattern, set]]:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    # organize items into groups (number of groups is auto-detected if not specified):
    def group(items, n=None, min=None, max=None):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def predict(seq, steps=1):  # predict next n elements in sequence:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    # produces an item with the average appearance, shape, value, etc of the inputs:
    def average(items):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    # produces an item with a semantic value equaling the combination of the inputs:
    def combine(items):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def _if(condition1: bool, fn1, condition2: bool, fn2, ..., conditionN: bool, fnN, else_fn):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def _elif(condition: bool, fn):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def _else(fn):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    # coconut is useful here:
    def switch(condition: bool, cases: dict, default=None):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def _while(condition: bool, loop):
        pass

    @export
    @ledger.ledgered
    def takewhile(self,
            continue_question: str = None,
            it: iter = None,
            get_context: ()->dict = ()->{},
            stop_quesiton: str = None,
            decision_freq=1):
        """get_context should be like a log of actions and results
        so the model can see what the effect of continue iteration is
        """
        assert not continue_question and stop_quesiton, \
            'Specify `continue_question` or `stop_question` but not both'
        for i in count():
            yield next(it)
            if i % decision_freq == 0 and (
                (continue_question is not None and 
                    self.decide(question=question, context=get_context()) == False)
                (stop_quesiton is not None and 
                    self.decide(question=stop_quesiton, context=get_context()) == True)):
                raise StopIteratorException()

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def call(fn):
        # produced wrapped function that intelligently selects parameters when called. You should pre-curry mandatory arguments and initiate the call afterwards if you don't have any conditioning parameters to supply. Useful as a wrapper for auto-converting inputs to the proper type. EG, tc._if asks for a bool, but you can give it a string and it will convert it to a bool automatically. Also, call avoids decoding from R only to have the function re-encode it to R in the backend.:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def do(instructions: R) -> any?:
        # executes a series of operations (including tensorcode operations) and returns result. The central function for tensorcode-based interpreter:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def analyze(items):
        # returns {'clusters': ..., 'trends': ..., 'statistics': ..., notes: ['blah blah blah', ...], ...}:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def visualize(object):
        # auto-generates visualizations that may be useful to understand the object(s) (code, data, or class definitions) and runs them. Useful in an interactive computing environment:
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def write_tests(self,
                    function: Callable,
                    instructions: str,
                    n: int = 1,
                    intended_behavior: str = None,
                    examples: list[Test] = None) -> list[Test]:
        '''auto-generates tests that may be useful to run and runs them. 
        Useful in an interactive computing environment'''
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def benchmark(function: Callable, instructions: str, metrics: list[str] = None):
        pass

    @export
    @ledger.ledgered
    @overloaded
    @encode_args
    def debug(self, program: str | Callable):
        """
        program:
          - path to program
          - 
        """
        # uses 3rd party tools to decompose program into modules of interest, tests, their expected behavior, and their actual behavior. Based on test result divergence, makes appropriate changes to code. This should be a future todo rather than main item:
        pass

    @export
    @ledger.ledgered
    def add_loss(self, loss, info): ...
    @export
    @ledger.ledgered
    def add_metric(self, metric, info): ...
