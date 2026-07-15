import os
import requests

from fastapi import FastAPI, Request
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


app = FastAPI()


# Load keys from Heroku Config Vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")


# Main settings
GROQ_MODEL = "llama-3.1-8b-instant"
PINECONE_INDEX_NAME = "small-shop-excel-agent"
PINECONE_NAMESPACE = "inventory-excel"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Create clients
groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def retrieve_context(customer_message, top_k=5):
    query_embedding = embedding_model.encode(customer_message).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE
    )

    chunks = []

    for match in results["matches"]:
        chunks.append(match["metadata"]["text"])

    return "\n\n".join(chunks)


def generate_whatsapp_reply(customer_message):
    context = retrieve_context(customer_message)

    system_prompt = """
You are a friendly WhatsApp AI agent for a small shop.

Rules:
- Use only the provided shop context.
- Do not invent prices, quantity, stock, or policies.
- Keep the answer short and natural like a WhatsApp message.
- After every answer, ask one open-ended question to guide the customer toward purchase.
"""

    user_prompt = f"""
Shop Context:
{context}

Customer Message:
{customer_message}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.4,
        max_tokens=300
    )

    return response.choices[0].message.content


def send_whatsapp_message(to_number, message_text):
    url = f"https://graph.facebook.com/v20.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {
            "body": message_text
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "WhatsApp shop agent is active"
    }


@app.get("/webhook")
async def verify_webhook(request: Request):
    params = dict(request.query_params)

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)

    return {
        "error": "Webhook verification failed"
    }


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()

    try:
        messages = data["entry"][0]["changes"][0]["value"].get("messages", [])

        if not messages:
            return {
                "status": "no customer message"
            }

        message = messages[0]
        customer_number = message["from"]
        customer_text = message["text"]["body"]

        reply = generate_whatsapp_reply(customer_text)

        whatsapp_response = send_whatsapp_message(customer_number, reply)

        return {
            "status": "reply sent",
            "customer_message": customer_text,
            "agent_reply": reply,
            "whatsapp_response": whatsapp_response
        }

    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }