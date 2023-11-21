def singleton(cls):
  """Ensures a class can only be a singleton"""

  cls.INSTANCE = None
  def MakeOrGetSingleton(*args, **kwargs):
    if cls.INSTANCE is None:
      cls.INSTANCE = cls.__new__(*args, **kwargs)
    return cls.INSTANCE
  cls.__new__ = MakeOrGetSingleton

  return cls