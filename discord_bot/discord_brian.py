import discord
import os
import requests
import re

# Nice way to load environment variables for deployments
from dotenv import load_dotenv
load_dotenv()

# Discord API key
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]

# extBrain API Key
API_KEY = os.environ["API_KEY"]

# URL Enpoint for extBrain
BRAIN_URL = os.environ["BRAIN_URL"]

# Get some identity info for the bot
BOT_IDENTITY = os.environ["BOT_IDENTITY"]

# Configure discord intent for chatting
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Removes discord IDs from strings
def remove_id(text):
    return re.sub(r'<@\d+>', '', text)

@client.event
async def on_ready():
    print(f'Bot logged in as {client.user}')

@client.event
async def on_message(message):

    # Never reply to yourself
    if message.author == client.user:
        return
    
    # Bots answer questions when messaged directly, if we do this, don't bother with triggers
    if client.user.mentioned_in(message):

        clean_message = remove_id(message.content)

        params = {
            "api_key": API_KEY,
            "prompt": clean_message,
            "system_message": BOT_IDENTITY
        }

        brain_response = requests.get(BRAIN_URL, params=params).json()
    
        # Send response to discord
        await message.channel.send(brain_response["completion"][:2000])

# Run the main loop
client.run(DISCORD_TOKEN)
