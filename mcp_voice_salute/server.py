from mcp.server.fastmcp import FastMCP

import os
import uuid
import asyncio
import httpx
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import io
import wave

mcp = FastMCP("mcp_salutespeech")

@mcp.tool()
async def sber_stt_record_and_recognize() -> str:
    """
    Records audio from microphone until 3 seconds of silence, obtains a Sber token via OAuth,
    sends the recorded PCM (16 kHz, 16-bit) to SmartSpeech API, and returns recognized text.
    Requires SALUTE_SPEECH environment variable for Basic auth.
    """
    auth_token = os.getenv("SALUTE_SPEECH")
    if not auth_token:
        raise ValueError("Environment variable SALUTE_SPEECH not set")

    # Получаем access_token
    rq_uid = str(uuid.uuid4())
    oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RqUID": rq_uid,
        "Authorization": f"Basic {auth_token}"
    }
    payload = {"scope": "SALUTE_SPEECH_PERS"}

    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.post(oauth_url, headers=headers, data=payload)
        resp.raise_for_status()
        access_token = resp.json().get("access_token")
    if not access_token:
        raise RuntimeError("Failed to obtain access token from Sber OAuth API")

    # Запись аудио
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 3.0
    print("Recording... Speak into the microphone.")
    try:
        with sr.Microphone(sample_rate=16000) as mic:
            audio = await asyncio.to_thread(recognizer.listen, mic)
    except Exception as e:
        raise RuntimeError(f"Microphone error: {e}")

    pcm_data = audio.get_raw_data(convert_rate=16000, convert_width=2)

    stt_url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
    stt_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "audio/x-pcm;bit=16;rate=16000"
    }
    async with httpx.AsyncClient(verify=False) as client:
        resp2 = await client.post(stt_url, headers=stt_headers, content=pcm_data)
    if resp2.status_code == 200:
        result = resp2.json()
        return result.get("result", "")
    else:
        raise RuntimeError(f"SmartSpeech STT API error {resp2.status_code}: {resp2.text}")

def play_audio(audio_data: bytes):
    """
    Воспроизводит аудио из бинарных WAV-данных с безопасным освобождением устройства.
    """
    wav_data = io.BytesIO(audio_data)
    try:
        with wave.open(wav_data, 'rb') as wav_file:
            framerate = wav_file.getframerate()
            audio_array = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
        
        print("Starting audio playback")
        sd.play(audio_array, framerate)
        # Ждем воспроизведения не более 15 секунд
        asyncio.run(asyncio.wait_for(asyncio.to_thread(sd.wait), timeout=15))
        print("Playback completed")
    except Exception as e:
        print(f"Playback error: {e}")
    finally:
        sd.stop()
        print("Audio device released")

@mcp.tool()
async def synthesize_speech(text: str, format: str = "wav16", voice: str = "Bys_24000") -> str:
    """
    Synthesizes speech from text using SaluteSpeech API and plays it through speakers.
    """
    auth_token = os.getenv("SALUTE_SPEECH")
    if not auth_token:
        raise ValueError("Environment variable SALUTE_SPEECH not set")

    rq_uid = str(uuid.uuid4())
    oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RqUID": rq_uid,
        "Authorization": f"Basic {auth_token}"
    }
    payload = {"scope": "SALUTE_SPEECH_PERS"}

    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.post(oauth_url, headers=headers, data=payload)
        resp.raise_for_status()
        token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("Failed to obtain access token from Sber OAuth API")

    url = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
    synth_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/text"
    }
    params = {"format": format, "voice": voice}

    async with httpx.AsyncClient(verify=False) as client:
        resp2 = await client.post(url, headers=synth_headers, params=params, content=text.encode())

    if resp2.status_code == 200:
        print("Synthesized audio received, starting playback")
        await asyncio.to_thread(play_audio, resp2.content)
        return "Audio played successfully"
    else:
        raise RuntimeError(f"Speech synthesis API error {resp2.status_code}: {resp2.text}")

def run():
    """
    Запускает MCP сервер.
    """
    mcp.run(transport="stdio")
