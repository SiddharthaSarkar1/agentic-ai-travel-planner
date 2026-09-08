from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent

# res = tavily_search("Best hotels in India")
# print(res)


# res = search_flights("Plan a 5 days tour from India to UK.")
# print(res)

user_input = input("Enter your travel request: ")

response = run_travel_agent(
    user_input=user_input,
    thread_id="sidd_test_id_11"
)

print("\nFINAL RESPONSE:\n")
print(response["answer"])