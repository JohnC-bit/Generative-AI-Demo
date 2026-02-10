import os
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
import google.generativeai as genai

load_dotenv()
print("Loading API KEYs")

# google_key = os.getenv("GOOGLE_API_KEY")
# if  google_key:
#     print("Linking to google key ... ")
#     try:
#         genai.configure(api_key=google_key)

#         # List the name of useable models
#         # for m in genai.list_models():
#         # if 'generateContent' in m.supported_generation_methods:
#         #     print(m.name)
#         model = genai.GenerativeModel('gemini-2.5-flash')
#         response = model.generate_content("请用鲁迅的语气夸一下我的代码写得好")
#         print(response.text)
#     except Exception as e:
#         print(f"google key error: ", {e})

# ds_key = os.getenv("DEEPSEEK_API_KEY")
# if ds_key:
#     print("Calling deepseek key ... ")
#     try:
#         client = OpenAI(api_key=ds_key, base_url="http://api.deepseek.com")
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages = [{"role": "user", "content":"Introduce yourself by one sentence"}],
#             stream = False
#         )
#         print(f"DeepSeek answer: {response.choices[0].message.content}")
#     except Exception as e:
#         print(f"DeepSeek key error: ", {e})

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("Linking to hf Qwen")
    try:
        client = InferenceClient(token=hf_token)
        output = client.chat_completion(
            messages=[{"role":"user","content":"As an AI assistant, what's your goal?"}],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=50
        )
        print(f"Hugging face answer: {output.choices[0].message.content}")
    except Exception as e:
        print(f"Hugging face token error: ", {e})