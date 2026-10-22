"""Chat models served by Groq.

Agent, planner and judge use different models on purpose: Groq's free tier limits tokens
per minute per model, so splitting the work keeps the agent model's quota for the agent.
"""

from functools import lru_cache

from langchain_groq import ChatGroq

from bookclub.config import get_settings


def _chat(model: str) -> ChatGroq:
    return ChatGroq(model=model, api_key=get_settings().groq_api_key, temperature=0, max_retries=5)


@lru_cache
def get_llm() -> ChatGroq:
    return _chat(get_settings().llm_model)


@lru_cache
def get_planner_llm() -> ChatGroq:
    return _chat(get_settings().planner_model)


@lru_cache
def get_judge_llm() -> ChatGroq:
    return _chat(get_settings().judge_model)
