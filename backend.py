import os
import certifi
from dotenv import load_dotenv

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_mistralai import ChatMistralAI
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from tools.currency_tool import get_exchange_rates
from tools.weather_tool import get_weather_details

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# LLM

llm = ChatMistralAI(model="mistral-small-2506")

# State


class TeavelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    currency_results: str
    weather_results: str
    llm_calls: int


# =========================
# Hotel Agent
# =========================


def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent
# =========================


def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Currency Agent
# =========================


def currency_agent(state: TravelState):
    query = f"What is the exchange rate for {state['user_query']}? and what is the converted amount? how much will it cost?"
    currency_results = get_exchange_rates(query)

    return {
        "currency_results": currency_results,
        "messages": [AIMessage(content="Currency information fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ===================================
#   Weather Agent
# ===================================


def weather_agent(state: TravelState):
    query = f"What is the weather like in {state['user_query']}?"
    weather_results = get_weather_details(query)
    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Final Response Agent
# =========================


def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final, comprehensive travel response for the user based on all the gathered information.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Weather Forecast:
{state['weather_results']}

Currency Exchange Info:
{state['currency_results']}

Format the final answer beautifully using these sections. Ensure every section is included.

1.  **Trip Summary:** A brief, engaging overview of the trip.
2.  **Flight Information:** Present the flight details clearly. Mention that the flight data is for live status and may not include prices.
3.  **Hotel Suggestions:** List the recommended hotels.
4.  **Weather Forecast:** Provide the weather details for the destination.
5.  **Day-by-Day Itinerary:** The detailed plan for each day.
6.  **Estimated Budget & Currency:** Discuss the budget and provide the currency exchange information.
7.  **Final Recommendations:** Offer practical tips and final thoughts for the trip.

Important Notes:
- Be clear, concise, and practical.
- If the flight API did not return pricing information, explicitly state that.
- Your final output should be a complete and helpful travel plan.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}


# ==========================
# Build the Graph
# ==========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("currency_agent", currency_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "currency_agent")
graph.add_edge("currency_agent", "weather_agent")
graph.add_edge("weather_agent", "final_agent")
graph.add_edge("final_agent", END)

# =====================================
# Database - PostgreSQL Checkpointer
# =====================================

DATABASE_URL = get_database_url()

_conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)

# ============================
# This function for FastAPI
# ============================


def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "currency_results": "",
            "weather_results": "",
            "llm_calls": 0,
        },
        config=config,
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "currency_results": result.get("currency_results", ""),
        "weather_results": result.get("weather_results", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
