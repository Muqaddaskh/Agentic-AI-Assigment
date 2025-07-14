import os
from dotenv import load_dotenv
from agents import Agent,AsyncOpenAI,Runner,OpenAIChatCompletionsModel,function_tool
from agents.run import RunConfig



load_dotenv()
gemini_apikey=os.getenv("GEMINI_API_KEY")
if not gemini_apikey:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

client = AsyncOpenAI(
    api_key=gemini_apikey,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

external_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
)
config= RunConfig(
   model_provider=client,
    model=external_model,
    tracing_disabled=True,
)

@function_tool()
async def getweather(city:str)-> str:
    """
    Get The Weather Of a city
    Args:
        city (str): The name of the city to get the weather for.
    """
    return f"The weather in {city} is sunny with a high of 38°C and a low of 28°C."

@ function_tool()
async def getresturant(name:str)-> str:
    """
    Get The Best Resturant Of a Location
    Args:
        Location (str): The name of the Location to get the best resturant for."""
    return f"The best resturant in {name} is 'The Choupal'. It is known for its budget friendly buffet"
@function_tool()
async def getlocation(area:str)-> str:
    """
    Get The Location Of a Area
    Args:
        area (str): The name of the area to get the location for.
    """
    return f"It is located in {area} and has many branches."

agent= Agent(
    name="Tools_Agent",
    instructions="You are a helpful assistant that can answer questions about weather, resturants and locations.",
    tools=[getweather,getresturant,getlocation]
)
result = Runner.run_sync(
    agent,
    input="What is the Weather in Karachi? What is the best resturant in katachi? Where is it Located?",
    run_config=config,
)
print(result.final_output)