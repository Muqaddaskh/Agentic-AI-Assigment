import os
from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,FunctionTool

from agents.run import RunConfig 
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
   raise ValueError("Gemini Api Key has been not ")

external_client = AsyncOpenAI(
   api_key=gemini_api_key,
   base_url="https://generativelanguage.googleapis.com/v1beta/openai/",)


external_model=OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

config=RunConfig(
    model=external_model,
    model_provider=external_client,
    tracing_disabled=True,
)

@FunctionTool()
async def getweather(city:str) -> str:
   """fetch the weather for a given loaction
   Args:
   city:The city to fetch the weather for."""
   return f"weather sunny in {city}"


@FunctionTool(
)
async def getresturant(name:str) -> str:
   """fetch the name for a given resturant
   Args:
   city:The name to fetch the location for."""
   return f"weather sunny in {name}"
@FunctionTool(
)
async def getlocation(name:str) -> str:
   """fetch the name for a given resturant
   Args:
   city:The name to fetch the location for."""
   return f"weather sunny in {name}"

agent=Agent(
   name="Function Calling",
   instructions="Tell about the function tools calling",
   tools=[getweather,getresturant,getlocation],
)
for tool in agent.tools:
       print(tool.name)

result = Runner.run_sync(
    agent=agent,
    input="According to the weather of city tell me the restaurant available on good location",
    run_config=config
)

print(result.final_output)       
       

