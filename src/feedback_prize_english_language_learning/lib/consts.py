from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class PrizeTypes:
    cohesion: str = "cohesion"
    syntax: str = "syntax"
    vocabulary: str = "vocabulary"
    phraseology: str = "phraseology"
    grammar: str = "grammar"
    conventions: str = "conventions"

    @staticmethod
    def all() -> list[str]:
        return [field for field in PrizeTypes.__annotations__.keys()]

    @staticmethod
    def all_scaled() -> list[str]:
        return [f"{field}_scaled" for field in PrizeTypes.__annotations__.keys()]

    def __getattribute__(self, name: str) -> Any:
        if name in PrizeTypes.all():
            return f"{object.__getattribute__(self, name)}_scaled"
        else:
            return object.__getattribute__(self, name)
