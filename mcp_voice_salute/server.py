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
from pathlib import Path

mcp = FastMCP("mcp_salutespeech")

@mcp.tool()
async def sber_stt_record_and_recognize() -> str:
    """
    Records audio from microphone until 3 seconds of silence, obtains a Sber token via OAuth,
    sends the recorded PCM (16 kHz, 16-bit) to SmartSpeech API, and returns recognized text.
    Requires SALUTE_SPEECH environment variable for Basic auth.
    """
    # Получаем базовый токен из окружения
    auth_token = os.getenv("SALUTE_SPEECH")
    if not auth_token:
        raise ValueError("Environment variable SALUTE_SPEECH not set")

    # Получение access_token
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

    # Запись аудио с микрофона до 3 секунд тишины
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 3.0
    with sr.Microphone(sample_rate=16000) as mic:
        print("Recording... Speak into the microphone. Recording will stop after 3 seconds of silence.")
        audio = await asyncio.to_thread(recognizer.listen, mic)

    # Получаем PCM-данные 16kHz, 16-bit
    pcm_data = audio.get_raw_data(convert_rate=16000, convert_width=2)

    # Отправляем данные на SmartSpeech API
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
    Воспроизводит аудио из бинарных данных WAV.
    """
    # Создаем временный буфер для WAV данных
    wav_data = io.BytesIO(audio_data)
    with wave.open(wav_data, 'rb') as wav_file:
        # Получаем параметры WAV файла
        framerate = wav_file.getframerate()
        # Читаем все фреймы и конвертируем в numpy array
        audio_array = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
    
    # Воспроизводим аудио
    sd.play(audio_array, framerate)
    sd.wait()  # Ждем окончания воспроизведения

@mcp.tool()
async def synthesize_speech(text: str, format: str = "wav16", voice: str = "Bys_24000") -> str:
    """
    Synthesizes speech from text using SaluteSpeech API and plays it through speakers.
    """
    # Получаем базовый токен
    auth_token = os.getenv("SALUTE_SPEECH")
    if not auth_token:
        raise ValueError("Environment variable SALUTE_SPEECH not set")
    
    # Получение access_token
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

    # Синтез речи
    url = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
    synth_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/text"
    }
    params = {"format": format, "voice": voice}
    
    async with httpx.AsyncClient(verify=False) as client:
        resp2 = await client.post(url, headers=synth_headers, params=params, content=text.encode())
    
    if resp2.status_code == 200:
        # Воспроизводим аудио напрямую
        await asyncio.to_thread(play_audio, resp2.content)
        return "Audio played successfully"
    else:
        raise RuntimeError(f"Speech synthesis API error {resp2.status_code}: {resp2.text}")

def run():
    """
    Запускает MCP сервер.
    """
    mcp.run(transport="stdio") 