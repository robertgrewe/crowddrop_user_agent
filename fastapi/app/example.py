import os
from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_BASE_URL"),
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = os.getenv("AZURE_VERSION")
)

# client = AzureOpenAI(
#   api_key = "1d45471d83ce40dcb03a81df8366248d",  
#   api_version = "2024-06-01",
#   azure_endpoint = "https://oai-seaisb-tailored-ke.openai.azure.com/"
# )

response = client.chat.completions.create(
    model="gpt-4o", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "Do you know what TMS is?"}
    ]
)

#print(response)
print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)