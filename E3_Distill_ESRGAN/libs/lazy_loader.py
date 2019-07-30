import importlib
from .settings import singleton


@singleton
class LazyLoader:
    def __init__(self):
        self.__import_dict = {}

    @property
    def import_dict(self):
        return self.__import_dict

    @import_dict.setter
    def import_dict(self, key, value):
        self.__import_dict[key] = value

    def import_(self, name, alias=None, parent=None, return_=True):
        if alias is None:
            alias = name
        if self.__import_dict.get(alias, None) is not None and return_:
            return self.__import_dict[alias]
        if not parent is None:
            exec("from {} import {} as {}".format(parent, name, "_" + alias))
            self.__import_dict[alias] = locals()["_" + alias]
        else:
            module = importlib.import_module(name)
            self.__import_dict[alias] = module
        if return_:
            return self.__import_dict[alias]

    def __getitem__(self, key):
        return self.__import_dict[key]
