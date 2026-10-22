"""Chat models served by Groq.

Groq's free tier limits tokens per minute per model, so the agent gets its own model and the
short structured calls (planner, judge) share a smaller one with low reasoning effort.
"""

from functools import lru_cache

from langchain_groq import ChatGroq

from bookclub.config import get_settings


def _chat(model: str, **kwargs) -> ChatGroq:
    return ChatGroq(model=model, api_key=get_settings().groq_api_key, temperature=0, max_retries=6, **kwargs)


@lru_cache
def get_llm() -> ChatGroq:
    return _chat(get_settings().llm_model)


@lru_cache
def get_planner_llm() -> ChatGroq:
    return _chat(get_settings().planner_model, reasoning_effort="low", max_tokens=600)


@lru_cache
def get_judge_llm() -> ChatGroq:
    return _chat(get_settings().judge_model, reasoning_effort="low", max_tokens=600)
