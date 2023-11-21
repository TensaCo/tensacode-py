class DefaultModel:

  encode = Encode()
  decode = Decode()
  select = Select()

  @property
  def operations(self):
    return self.encode, self.decode, self.select