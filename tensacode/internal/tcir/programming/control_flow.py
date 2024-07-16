class ControlFlowStatement(ImperativeStatement, ABC):
    pass


class ConditionalBranching(ControlFlowStatement):
    class Branch(BaseModel):
        condition: Expression
        body: ExpressionBlock

    name = "conditional_branching"
    branches: list[Branch]


class WhileLoop(ControlFlowStatement):
    name = "while_loop"
    condition: Expression
    body: ExpressionBlock


class ForEachLoop(ControlFlowStatement):
    name = "for_each_loop"
    iterable: IsIterable
    body: ExpressionBlock
