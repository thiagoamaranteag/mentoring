# pip install SpeechRecognition pydub

import os
import sys
import math
import datetime
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
AudioSegment.converter = "ffmpeg.exe"

# Split do arquivo principal
DURACAO_MAX_TRECHO_MS = 15 * 60 * 1000  # 15 minutos

# Remoção de silêncio
LIMITE_SILENCIO_MS = 1000   # >= 1s
VOLUME_SILENCIO_DBFS = -40  # ajuste conforme seu áudio
KEEP_SILENCE_MS = 200       # 200ms nas bordas

# Limite por requisição do Google Web Speech (recomendado <= 50s)
GOOGLE_SEGMENTO_S = 50

# ---------- Utilitários ----------
def log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def ms_to_hhmmss(ms: int) -> str:
    s = ms // 1000
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ---------- Pipeline ----------
def processar(audio_file):

    try:
        log(f"Iniciando processamento do arquivo: {audio_file}")
        audio = AudioSegment.from_wav(audio_file)
        total_ms = len(audio)
        num_trechos = math.ceil(total_ms / DURACAO_MAX_TRECHO_MS)
        log(f"Duração total: {ms_to_hhmmss(total_ms)} ({total_ms} ms)")
        log(f"Trechos de até 15min: {num_trechos}")

        recognizer = sr.Recognizer()

        for i in range(num_trechos):
            inicio_ms = i * DURACAO_MAX_TRECHO_MS
            fim_ms = min(inicio_ms + DURACAO_MAX_TRECHO_MS, total_ms)
            faixa = audio[inicio_ms:fim_ms]

            log(f"Trecho {i+1}/{num_trechos}: {ms_to_hhmmss(inicio_ms)} - {ms_to_hhmmss(fim_ms)} (removendo silêncios)")
            partes = split_on_silence(
                faixa,
                min_silence_len=LIMITE_SILENCIO_MS,
                silence_thresh=VOLUME_SILENCIO_DBFS,
                keep_silence=KEEP_SILENCE_MS
            )

            if partes:
                faixa_limpa = sum(partes, AudioSegment.silent(duration=0))
            else:
                log("Nenhum silêncio detectado para este trecho; usando áudio original do trecho.")
                faixa_limpa = faixa

            temp_wav = f"temp_trecho_{i:03d}.wav"
            faixa_limpa.export(temp_wav, format="wav")
            log(f"Trecho {i+1}: arquivo temporário gerado: {temp_wav}")

            # Transcreve em janelas de até 50s para respeitar limites da API
            log(f"Trecho {i+1}: iniciando transcrição (janelas de {GOOGLE_SEGMENTO_S}s)")
            texto_trecho = []

            try:
                with sr.AudioFile(temp_wav) as source:
                    while True:
                        audio_data = recognizer.record(source, duration=GOOGLE_SEGMENTO_S)
                        if len(audio_data.frame_data) == 0:
                            break
                        try:
                            parte_txt = recognizer.recognize_google(audio_data, language='pt-BR')
                            if parte_txt.strip():
                                texto_trecho.append(parte_txt.strip())
                        except sr.UnknownValueError:
                            log("Google não entendeu um subtrecho; seguindo...")
                            continue
                        except sr.RequestError as e:
                            log(f"Erro de requisição à API Google: {e}. Prosseguindo para o próximo trecho.")
                            break
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                    log(f"Trecho {i+1}: temporário removido: {temp_wav}")

            texto_concatenado = " ".join(texto_trecho).strip()

            transcription_file = f"{audio_file}_transcription.txt"

            with open(transcription_file, "a", encoding="utf-8") as f:
                f.write(f"\n===== Trecho {i+1} ({ms_to_hhmmss(inicio_ms)} - {ms_to_hhmmss(fim_ms)}) =====\n")
                if texto_concatenado:
                    f.write(texto_concatenado + "\n")
                else:
                    f.write("[Sem transcrição válida para este trecho]\n")

            log(f"Trecho {i+1} concluído ({len(texto_concatenado)} caracteres).")

        log(f"Processamento concluído. Arquivo final: {transcription_file}")

    except sr.UnknownValueError:
        log("A API do Google não conseguiu entender o áudio.")
    except sr.RequestError as e:
        log(f"Não foi possível solicitar resultados da API Google; {e}")
    except FileNotFoundError:
        log(f"ERRO: Arquivo '{audio_file}' não encontrado.")
    except Exception as e:
        log(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    
    # Verifica se foi fornecido o argumento do arquivo
    if len(sys.argv) != 2:
        log("ERRO: Uso incorreto do script.")
        log("Uso correto: python speech_recongnition.py <arquivo.wav>")
        log("Exemplo: python speech_recongnition.py aula_19_1.wav")
        exit(1)
    
    path_file = sys.argv[1]
    
    # Verifica se o arquivo existe
    if not os.path.exists(path_file):
        log(f"ERRO: Arquivo '{path_file}' não encontrado.")
        log("Certifique-se de fornecer o caminho correto do arquivo.")
        exit(1)
    
    # Verifica se é um arquivo .wav
    if not path_file.lower().endswith('.wav'):
        log(f"ERRO: '{path_file}' não é um arquivo .wav")
        exit(1)
    
    processar(path_file)