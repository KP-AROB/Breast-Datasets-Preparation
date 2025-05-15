from typing import Dict, Type, TypeVar, Any
from src.core.df_loaders import BaseDataframeLoader, VindrDataframeLoader, CBISDataframeLoader, INBreastDataframeLoader
from src.core.converters.base import BaseConverter
from src.core.converters.vindr import VindrH5Converter

T = TypeVar('T')

def create_registry(base_cls: Type[T]):
    registry: Dict[str, Type[T]] = {}

    def register(name: str):
        def decorator(cls: Type[T]):
            if not issubclass(cls, base_cls):
                raise TypeError(f"{cls.__name__} must inherit from {base_cls.__name__}")
            registry[name.lower()] = cls
            return cls
        return decorator

    def get(name: str, *args: Any, **kwargs: Any) -> T:
        try:
            cls = registry[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown {base_cls.__name__} name: {name}. Available: {list(registry)}")
        return cls(*args, **kwargs)

    return register, get


register_loader, get_dataframe_loader = create_registry(BaseDataframeLoader)
register_converter, get_converter = create_registry(BaseConverter)

register_loader("vindr")(VindrDataframeLoader)
register_loader("cbis")(CBISDataframeLoader)
register_loader("inbreast")(INBreastDataframeLoader)

register_converter("vindr")(VindrH5Converter)