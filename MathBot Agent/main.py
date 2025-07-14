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


agent=Agent(
   name="Math Bot",
   instructions="Solve All Math Problems",
)


result = Runner.run_sync(
      agent,
    input="what is 409*23?",
    run_config=config
)

print(result.final_output) 
