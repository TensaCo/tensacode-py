# class DataAffordance(ABC):
#     pass


# class IsIterable(DataAffordance):
#     def next(self) -> TCIRAny:
#         pass


# class IsCollection(DataAffordance):
#     pass


# class IsSequence(DataAffordance):
#     ordered = True


# class IsMapping(DataAffordance):
#     pass


# class IsFunction(DataAffordance):
#     pass


# class IsCoroutine(DataAffordance):
#     pass


# class IsAtomic(DataAffordance):
#     pass


# class IsComposite(DataAffordance):
#     @property
#     def relations


# class CanSetDirectly(DataAffordance):
#     pass


# class IsNumeric(DataAffordance):
#     pass


# class IsImmutable(DataAffordance):
#     pass


# class Entity(BaseModel):
#     pass

# @polymorphic
# def parse(data: TCIRAny) -> Entity:
#     pass

# @parse.register(int)
# def parse_int(data: int) -> Entity:
#     return Entity(data=data)
