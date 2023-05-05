import os
import pickle


class Cache:
    _cache_disabled=False

    @classmethod
    def set_cache_disabled(cls,val:bool):
        cls._cache_disabled=val

    @classmethod
    def is_cache_disabled(cls)->bool:
        return cls._cache_disabled

    @classmethod
    def get_cached_object(cls,name: str):
        try:
            with open(os.path.dirname(__file__) + f"/tmp/{name}.bin", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    @classmethod
    def cache_object(cls, o, name: str):
        with open(os.path.dirname(__file__) + f"/tmp/{name}.bin", "wb+") as f:
            pickle.dump(o, f)
