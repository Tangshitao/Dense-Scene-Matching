class AttrDict(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = AttrDict(value)
        return value
