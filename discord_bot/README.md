# External Brain - Discord Bot

A Discord bot for external brain.  This allows you to provide a bot to answer questions based on whatever knowledge you've added into externalbrain in your own Discord servers.

## Compatibility

External Brain was designed around using llama.cpp in server mode, however it can now be run with OpenAI or Mistral.ai keys! Check out the sample.env file for configuration options.

## Basic Installation

```
pip install -r requirements.txt
```

## Configuration

Copy the sample.env file to .env and edit this file.  This contains all the important configuration variables for the application to run.

* BRAIN_URL - This is the URL to wherever extBrain is running.  If running on localhost the default is correct.
* API_KEY - This is the security API key you configured in extBrain, just make sure they match here.
* DISCORD_TOKEN - Get one from the discord app portal:  https://discord.com/developers/applications
* BOT_IDENTITY - This is the "personality" your bot will have, experiment!

## Running

### MacOS / Linux

```
python3 discord_brain.py
```

### Windows

```
python discord_brain.py
```