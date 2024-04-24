from langchain.llms import Bedrock

def bedrock_chatbot(input_text):
    bedrock_lim = Bedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2:1',
        model_kwargs= {
            'prompt': '\n\nHuman:<prompt>\n\nAssistant:',
            'temperature': 0.5,
            'top_p': 1,
            'top_k': 250,
            'max_tokens_to_sample': 512
        })
    return bedrock_lim.predict(input_text)

res = bedrock_chatbot('프리미어리그 역대 득점 순위 알려줘')
print(res)