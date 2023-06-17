import keyboard
import os
import tempfile

import numpy as np
import openai
import sounddevice as sd
import soundfile as sf
import tweepy

from elevenlabs import generate, play, set_api_key
import time
from gtts import gTTS
from playsound import playsound
from io import BytesIO

from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.utilities.zapier import ZapierNLAWrapper


set_api_key("8993831abd4e6722f54d375b50b16d32")
openai.api_key = "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"

# Set recording parameters
duration = 5  # duration of each recording in seconds
fs = 44100  # sample rate
channels = 1  # number of channels


def record_audio(duration, fs, channels):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    print("Finished recording.")
    return recording


def transcribe_audio(recording, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recording, fs)
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        os.remove(temp_audio.name)
    return transcript["text"].strip()


def play_generated_audio(text, voice="Bella", model="eleven_monolingual_v1"):
    audio = generate(text=text, voice=voice, model=model)
    play(audio)

def play_gtts_audio(text):
    current_time = int(time.time())
    time_str = str(current_time)
    tts = gTTS(text=text, lang="en", tld='us')
    audio_file_path = "./audio/" + time_str +"_message_agent.mp3"
    tts.save(audio_file_path)
    playsound(audio_file_path)
    os.remove(audio_file_path)


# Replace with your Twitter API keys
consumer_key = "aERfVTF4NWpHb2d4MFJtS2JEV3U6MTpjaQ"
consumer_secret = "7lLaLMDwlEIgaTI50Ac1xuYKmAkEwgjNRG7yroLmOzR-ey1RQm"
access_token = "KUMPz2iNb1SvsQIGWtEs2ytuL"
access_token_secret = "PE4CxKVtHG92TpW2BPkOCtfgu47wHEePSFwgHLct260eZ1gi2B"

client = tweepy.Client(
    consumer_key=consumer_key, consumer_secret=consumer_secret,
    access_token=access_token, access_token_secret=access_token_secret
)


class TweeterPostTool(BaseTool):
    name = "Twitter Post Tweet"
    description = "Use this tool to post a tweet to twitter."

    def _run(self, text: str) -> str:
        """Use the tool."""
        return client.create_tweet(text=text)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async")


if __name__ == '__main__':

    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history")

    zapier = ZapierNLAWrapper(zapier_nla_api_key="sk-ak-74FXQodsqeLTA5aHhMLQIWsWbW")
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    #tools = [TweeterPostTool()] + toolkit.get_tools() + load_tools(["human"])

    tools = toolkit.get_tools() + load_tools(["human"])

    agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)

    while True:
        print("Press spacebar to start recording.")
        keyboard.wait("space")  # wait for spacebar to be pressed
        recorded_audio = record_audio(duration, fs, channels)
        message = transcribe_audio(recorded_audio, fs)
        print(f"You: {message}")
        assistant_message = agent.run(message)
        #play_generated_audio(assistant_message)
        play_gtts_audio(assistant_message)
