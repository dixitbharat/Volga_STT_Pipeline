from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] | None = None


class TranscriptionResult(BaseModel):
    text: str
    language: str | None = None
    language_probability: float | None = None
    duration_seconds: float
    chunk_count: int = 1
    model: str
    segments: list[Segment] = Field(default_factory=list)
