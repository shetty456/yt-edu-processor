from __future__ import annotations
from enum import Enum
from typing import Annotated, Any
from pydantic import BaseModel, Field, field_validator, model_validator


# ── API request ───────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    youtube_url: str

    @field_validator("youtube_url")
    @classmethod
    def must_be_youtube(cls, v: str) -> str:
        v = v.strip()
        if not any(d in v for d in ("youtube.com/watch", "youtu.be/")):
            raise ValueError("Must be a YouTube watch URL")
        return v


# ── MCQ / Quiz models ─────────────────────────────────────────────────────────

class MCQOptions(BaseModel):
    A: str = Field(..., min_length=1)
    B: str = Field(..., min_length=1)
    C: str = Field(..., min_length=1)
    D: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def all_unique(self) -> "MCQOptions":
        if len({self.A, self.B, self.C, self.D}) != 4:
            raise ValueError("All four options must be distinct")
        return self


class MCQItem(BaseModel):
    question: str = Field(..., min_length=10)
    options: MCQOptions
    answer: str = Field(..., pattern=r"^[ABCD]$")
    description: str = Field(..., min_length=20)


class QuizPayload(BaseModel):
    quiz: Annotated[list[MCQItem], Field(min_length=15, max_length=15)]

    @field_validator("quiz")
    @classmethod
    def no_duplicate_questions(cls, v: list[MCQItem]) -> list[MCQItem]:
        qs = [q.question.strip().lower() for q in v]
        if len(set(qs)) != len(qs):
            raise ValueError("All questions must be unique")
        return v


# ── API response ──────────────────────────────────────────────────────────────

class ProcessResponse(BaseModel):
    video_title: str
    summary_markdown: str
    learning_objectives: list[str]
    quiz: list[dict[str, Any]]
    eval_passed: bool


# ── Internal pipeline DTOs ────────────────────────────────────────────────────

class VideoMeta(BaseModel):
    title: str
    duration_seconds: int
    transcript: str = Field(..., min_length=50)
    word_count: int


class ChunkSummary(BaseModel):
    core_concepts: list[str] = Field(..., min_length=1)
    important_examples: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(..., min_length=1)
    definitions: list[str] = Field(default_factory=list)


class MergedSummary(BaseModel):
    core_concepts: list[str]
    important_examples: list[str]
    key_points: list[str]
    definitions: list[str]


# ── Eval model ────────────────────────────────────────────────────────────────

class EvalSeverity(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class EvalResult(BaseModel):
    overall: EvalSeverity
    summary_complete: bool
    summary_grounded: bool
    quiz_answers_correct: bool
    quiz_descriptions_helpful: bool
    objectives_measurable: bool
    issues: list[dict[str, str]] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    recommendation: str

    @property
    def passed(self) -> bool:
        return self.overall in (EvalSeverity.OK, EvalSeverity.WARN)