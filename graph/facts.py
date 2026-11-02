"""The verified facts store: the only source of truth about the candidate."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class FactsStoreError(RuntimeError):
    """The facts store is missing or invalid."""


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    text: str = Field(min_length=1)


class FactsStore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_name: str = Field(min_length=1)
    facts: list[Fact] = Field(min_length=1)

    @field_validator("facts")
    @classmethod
    def ids_are_unique(cls, facts: list[Fact]) -> list[Fact]:
        ids = [fact.id for fact in facts]
        if len(ids) != len(set(ids)):
            raise ValueError("fact ids must be unique")
        return facts

    @property
    def ids(self) -> set[str]:
        return {fact.id for fact in self.facts}

    def render(self) -> str:
        lines = [f"Candidate name: {self.candidate_name}"]
        lines += [f"[{fact.id}] ({fact.category}) {fact.text}" for fact in self.facts]
        return "\n".join(lines)


def load_facts(path: Path) -> FactsStore:
    try:
        return FactsStore.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FactsStoreError(f"cannot read facts store at {path}: {exc}") from exc
    except ValidationError as exc:
        raise FactsStoreError(f"invalid facts store at {path}: {exc}") from exc
