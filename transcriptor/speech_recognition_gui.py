# pip install SpeechRecognition pydub
import os
import shutil
import sys
import math
import datetime
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pathlib import Path
import subprocess

# ===== CONFIGURAÇÃO AUTOMÁTICA DO FFMPEG =====
def configurar_ffmpeg():
    """Detecta e configura o FFmpeg automaticamente"""
    # Primeiro tenta encontrar via PATH do sistema (funciona no Linux/Mac)
    ffmpeg_no_path = shutil.which("ffmpeg")
    if ffmpeg_no_path:
        return ffmpeg_no_path

    # Fallback: caminhos relativos ao executável (útil no Windows)
    if getattr(sys, 'frozen', False):
        diretorio_base = os.path.dirname(sys.executable)
    else:
        diretorio_base = os.path.dirname(os.path.abspath(__file__))

    possiveis_caminhos = [
        os.path.join(diretorio_base, "ffmpeg.exe"),
        os.path.join(diretorio_base, "ffmpeg"),
        os.path.join(diretorio_base, "ffmpeg", "ffmpeg.exe"),
        os.path.join(diretorio_base, "ffmpeg", "bin", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]

    for caminho in possiveis_caminhos:
        if not os.path.exists(caminho):
            continue
        try:
            subprocess.run(
                [caminho, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=True
            )
            return caminho
        except:
            continue

    return None

ffmpeg_path = configurar_ffmpeg()
if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    ffprobe_path = shutil.which("ffprobe") or ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    if os.path.exists(ffprobe_path):
        AudioSegment.ffprobe = ffprobe_path

# Configurações
DURACAO_MAX_TRECHO_MS = 15 * 60 * 1000
LIMITE_SILENCIO_MS = 1000
VOLUME_SILENCIO_DBFS = -40
KEEP_SILENCE_MS = 200
GOOGLE_SEGMENTO_S = 50

FORMATOS_SUPORTADOS = {
    'wav': 'wav', 'mp3': 'mp3', 'mp4': 'mp4', 'mkv': 'matroska',
    'avi': 'avi', 'flv': 'flv', 'mov': 'mov', 'webm': 'webm',
    'ogg': 'ogg', 'flac': 'flac', 'm4a': 'm4a'
}

IDIOMAS = {
    "Português (Brasil)":   "pt-BR",
    "Inglês (EUA)":         "en-US",
    "Inglês (Reino Unido)": "en-GB",
    "Espanhol":             "es-ES",
    "Francês":              "fr-FR",
    "Alemão":               "de-DE",
    "Italiano":             "it-IT",
    "Japonês":              "ja-JP",
    "Chinês (Mandarim)":    "zh-CN",
    "Russo":                "ru-RU",
}


class TranscritorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Transcritor de Áudio/Vídeo - v2.0")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Success.TLabel', foreground='#28a745', font=('Segoe UI', 9, 'bold'))
        style.configure('Warning.TLabel', foreground='#ffc107', font=('Segoe UI', 9, 'bold'))
        style.configure('Error.TLabel',   foreground='#dc3545', font=('Segoe UI', 9, 'bold'))
        style.configure('Info.TLabel',    foreground='#17a2b8', font=('Segoe UI', 9))
        style.configure('Custom.Horizontal.TProgressbar',
                        troughcolor='#e9ecef', background='#007bff',
                        bordercolor='#adb5bd', lightcolor='#007bff', darkcolor='#0056b3')

        self.audio_file  = None
        self.processing  = False
        self.total_trechos = 0
        self.trecho_atual  = 0

        self.verificar_ffmpeg()
        self.setup_ui()

    def verificar_ffmpeg(self):
        if not ffmpeg_path:
            resposta = messagebox.askyesno(
                "⚠️ FFmpeg Não Encontrado",
                "O FFmpeg não foi encontrado!\n\n"
                "Sem ele, apenas arquivos WAV podem ser processados.\n\n"
                "Deseja abrir o site de download?",
                icon='warning')
            if resposta:
                import webbrowser
                webbrowser.open("https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip")
                messagebox.showinfo("📥 Instruções",
                    "1. Baixe o arquivo ZIP\n"
                    "2. Extraia ffmpeg.exe E ffprobe.exe\n"
                    "3. Cole no mesmo diretório deste programa\n"
                    "4. Reinicie o aplicativo")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # ===== HEADER =====
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        ttk.Label(header_frame, text="🎙️ Transcritor de Áudio/Vídeo",
                  font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(header_frame, text="Transcreva seus arquivos de áudio e vídeo com facilidade",
                  font=("Segoe UI", 9), foreground="gray").grid(row=1, column=0, sticky=tk.W)

        # ===== SEÇÃO 1: Seleção de Arquivo =====
        file_frame = ttk.LabelFrame(main_frame, text="📁 Seleção de Arquivo", padding="15")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="📄", font=("Segoe UI", 24)).grid(
            row=0, column=0, rowspan=2, padx=(0, 15))

        self.file_label = ttk.Label(file_frame, text="Nenhum arquivo selecionado",
                                    foreground="gray", font=("Segoe UI", 10))
        self.file_label.grid(row=0, column=1, sticky=tk.W)

        self.file_info_label = ttk.Label(file_frame,
                                         text="Clique em 'Selecionar Arquivo' para começar",
                                         foreground="gray", font=("Segoe UI", 8))
        self.file_info_label.grid(row=1, column=1, sticky=tk.W)

        # Botões + seletor de idioma
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(15, 0))

        self.btn_select = ttk.Button(button_frame, text="🔍 Selecionar Arquivo",
                                     command=self.selecionar_arquivo, width=20)
        self.btn_select.grid(row=0, column=0, padx=(0, 10))

        self.btn_process = ttk.Button(button_frame, text="🚀 Iniciar Transcrição",
                                      command=self.iniciar_processamento,
                                      state=tk.DISABLED, width=20)
        self.btn_process.grid(row=0, column=1, padx=(0, 20))

        # Seletor de idioma
        ttk.Label(button_frame, text="🌐 Idioma:", font=("Segoe UI", 9)).grid(
            row=0, column=2, padx=(0, 5))

        self.idioma_var = tk.StringVar(value="Inglês (EUA)")
        idioma_combo = ttk.Combobox(button_frame,
                                    textvariable=self.idioma_var,
                                    values=list(IDIOMAS.keys()),
                                    state="readonly",
                                    width=22)
        idioma_combo.grid(row=0, column=3)

        # ===== SEÇÃO 2: Progresso =====
        progress_frame = ttk.LabelFrame(main_frame, text="⏳ Progresso da Transcrição", padding="15")
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        progress_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(progress_frame, text="Aguardando arquivo...",
                                      style='Info.TLabel', font=("Segoe UI", 10, "bold"))
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        progress_container = ttk.Frame(progress_frame)
        progress_container.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_container.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(progress_container, mode='determinate',
                                             style='Custom.Horizontal.TProgressbar', length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))

        self.percent_label = ttk.Label(progress_container, text="0%",
                                       font=("Segoe UI", 10, "bold"))
        self.percent_label.grid(row=0, column=1)

        stats_frame = ttk.Frame(progress_frame)
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        for col in range(3):
            stats_frame.columnconfigure(col, weight=1)

        stat1 = ttk.Frame(stats_frame)
        stat1.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(stat1, text="Trecho:", font=("Segoe UI", 8), foreground="gray").grid(row=0, column=0, sticky=tk.W)
        self.trecho_label = ttk.Label(stat1, text="0/0", font=("Segoe UI", 10, "bold"))
        self.trecho_label.grid(row=1, column=0, sticky=tk.W)

        stat2 = ttk.Frame(stats_frame)
        stat2.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Label(stat2, text="Tempo Decorrido:", font=("Segoe UI", 8), foreground="gray").grid(row=0, column=0, sticky=tk.W)
        self.tempo_label = ttk.Label(stat2, text="00:00:00", font=("Segoe UI", 10, "bold"))
        self.tempo_label.grid(row=1, column=0, sticky=tk.W)

        stat3 = ttk.Frame(stats_frame)
        stat3.grid(row=0, column=2, sticky=tk.W)
        ttk.Label(stat3, text="Caracteres:", font=("Segoe UI", 8), foreground="gray").grid(row=0, column=0, sticky=tk.W)
        self.chars_label = ttk.Label(stat3, text="0", font=("Segoe UI", 10, "bold"))
        self.chars_label.grid(row=1, column=0, sticky=tk.W)

        # ===== SEÇÃO 3: Log =====
        log_frame = ttk.LabelFrame(main_frame, text="📋 Log de Processamento", padding="15")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, width=80, height=15,
            font=("Consolas", 9), state=tk.DISABLED,
            background="#f8f9fa", relief=tk.FLAT, borderwidth=2)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.log_text.tag_config("success", foreground="#28a745", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("error",   foreground="#dc3545", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("warning", foreground="#ffc107", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("info",    foreground="#17a2b8")

        ttk.Button(log_frame, text="🗑️ Limpar Log", command=self.limpar_log, width=15).grid(
            row=1, column=0, sticky=tk.E)

        # ===== RODAPÉ =====
        footer_frame = ttk.Frame(main_frame)
        footer_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        ttk.Separator(footer_frame, orient='horizontal').grid(
            row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        if ffmpeg_path:
            ffprobe_path = shutil.which("ffprobe") or ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
            footer_text  = "✅ FFmpeg e FFprobe detectados" if os.path.exists(ffprobe_path) else "⚠️ FFmpeg OK, mas falta ffprobe"
            footer_style = 'Success.TLabel' if os.path.exists(ffprobe_path) else 'Warning.TLabel'
        else:
            footer_text  = "⚠️ FFmpeg não encontrado - Apenas WAV suportado"
            footer_style = 'Warning.TLabel'

        ttk.Label(footer_frame, text=footer_text, style=footer_style,
                  font=("Segoe UI", 8)).grid(row=1, column=0, sticky=tk.W)

        # Log inicial
        self.log("🚀 Aplicativo iniciado", "info")
        if ffmpeg_path:
            self.log(f"✅ FFmpeg: {ffmpeg_path}", "success")
            ffprobe_path = shutil.which("ffprobe") or ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
            if os.path.exists(ffprobe_path):
                self.log(f"✅ FFprobe: {ffprobe_path}", "success")
            else:
                self.log("⚠️ FFprobe não encontrado", "warning")
        else:
            self.log("⚠️ FFmpeg não detectado", "warning")

        self.inicio_processamento = None
        self.total_chars = 0

    def atualizar_tempo_decorrido(self):
        if self.processing and self.inicio_processamento:
            delta = datetime.datetime.now() - self.inicio_processamento
            h, resto = divmod(delta.seconds, 3600)
            m, s = divmod(resto, 60)
            self.tempo_label.config(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self.atualizar_tempo_decorrido)

    def atualizar_progresso(self, trecho_atual: int, total: int):
        self.trecho_atual  = trecho_atual
        self.total_trechos = total
        porcentagem = int((trecho_atual / total) * 100) if total > 0 else 0
        self.progress_bar['value']   = porcentagem
        self.progress_bar['maximum'] = 100
        self.percent_label.config(text=f"{porcentagem}%")
        self.trecho_label.config(text=f"{trecho_atual}/{total}")
        self.root.update_idletasks()

    def selecionar_arquivo(self):
        if ffmpeg_path:
            ffprobe_path = shutil.which("ffprobe") or ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
            if os.path.exists(ffprobe_path):
                filetypes = [
                    ("Todos suportados", "*.wav *.mp3 *.mp4 *.mkv *.avi *.flv *.mov *.webm *.ogg *.flac *.m4a"),
                    ("Arquivos de Áudio", "*.wav *.mp3 *.ogg *.flac *.m4a"),
                    ("Arquivos de Vídeo", "*.mp4 *.mkv *.avi *.flv *.mov *.webm"),
                    ("Todos os arquivos", "*.*")]
            else:
                filetypes = [("Arquivos de Áudio", "*.wav *.mp3 *.ogg *.flac *.m4a"),
                             ("Todos os arquivos", "*.*")]
        else:
            filetypes = [("Arquivos WAV", "*.wav"), ("Todos os arquivos", "*.*")]

        filename = filedialog.askopenfilename(
            title="Selecione o arquivo de áudio ou vídeo", filetypes=filetypes)

        if filename:
            extensao = Path(filename).suffix.lower().replace('.', '')
            if not ffmpeg_path and extensao != 'wav':
                messagebox.showerror("❌ Formato Não Suportado",
                    f"Sem o FFmpeg, apenas arquivos WAV são suportados.\n\nArquivo: .{extensao}")
                return

            self.audio_file = filename
            tipo = "🎵 Áudio" if extensao in ['wav', 'mp3', 'ogg', 'flac', 'm4a'] else "🎬 Vídeo"
            tamanho = os.path.getsize(filename) / (1024 * 1024)

            self.file_label.config(text=Path(filename).name, foreground="#007bff")
            self.file_info_label.config(
                text=f"{tipo} • {extensao.upper()} • {tamanho:.2f} MB", foreground="gray")
            self.btn_process.config(state=tk.NORMAL)
            self.status_label.config(text="✅ Arquivo carregado - Pronto para processar",
                                     style='Success.TLabel')
            self.log(f"📂 Arquivo selecionado: {Path(filename).name}", "info")
            self.log(f"📊 Tipo: {tipo} | Formato: {extensao.upper()} | Tamanho: {tamanho:.2f} MB", "info")

    def log(self, msg: str, tipo: str = ""):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{now}] {msg}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_msg, tipo) if tipo else self.log_text.insert(tk.END, log_msg)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def limpar_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log("🗑️ Log limpo", "info")

    def ms_to_hhmmss(self, ms: int) -> str:
        s = ms // 1000
        h, resto = divmod(s, 3600)
        m, sec = divmod(resto, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def carregar_audio(self, caminho_arquivo: str) -> AudioSegment:
        extensao = Path(caminho_arquivo).suffix.lower().replace('.', '')
        self.log(f"📂 Carregando arquivo {extensao.upper()}...", "info")
        try:
            if extensao == 'wav':
                audio = AudioSegment.from_wav(caminho_arquivo)
            elif extensao == 'mp3':
                audio = AudioSegment.from_mp3(caminho_arquivo)
            elif extensao in ['mp4', 'mkv', 'avi', 'flv', 'mov', 'webm']:
                self.log("🎬 Extraindo áudio do vídeo...", "info")
                audio = AudioSegment.from_file(caminho_arquivo)
            else:
                audio = AudioSegment.from_file(caminho_arquivo)
            self.log("✅ Áudio carregado com sucesso!", "success")
            self.log(f"📊 Canais: {audio.channels} | Sample Rate: {audio.frame_rate} Hz", "info")
            return audio
        except Exception as e:
            raise Exception(f"Erro ao carregar arquivo: {str(e)}")

    def iniciar_processamento(self):
        if self.processing:
            messagebox.showwarning("Aviso", "Já existe um processamento em andamento!")
            return
        if not self.audio_file:
            messagebox.showerror("Erro", "Selecione um arquivo primeiro!")
            return

        self.total_chars = 0
        self.chars_label.config(text="0")
        self.inicio_processamento = datetime.datetime.now()
        self.btn_select.config(state=tk.DISABLED)
        self.btn_process.config(state=tk.DISABLED)
        self.processing = True
        self.status_label.config(text="🔄 Processando arquivo...", style='Info.TLabel')
        self.atualizar_tempo_decorrido()

        thread = threading.Thread(target=self.processar_audio, daemon=True)
        thread.start()

    def processar_audio(self):
        try:
            # Captura o idioma selecionado antes de entrar na thread
            idioma_nome   = self.idioma_var.get()
            idioma_codigo = IDIOMAS.get(idioma_nome, "en-US")

            self.log(f"🎵 Iniciando processamento: {Path(self.audio_file).name}", "info")
            self.log(f"🌐 Idioma selecionado: {idioma_nome} ({idioma_codigo})", "info")

            audio = self.carregar_audio(self.audio_file)
            total_ms   = len(audio)
            num_trechos = math.ceil(total_ms / DURACAO_MAX_TRECHO_MS)

            self.log(f"⏱️ Duração total: {self.ms_to_hhmmss(total_ms)}", "info")
            self.log(f"📊 Trechos de até 15min: {num_trechos}", "info")

            recognizer = sr.Recognizer()

            for i in range(num_trechos):
                inicio_ms = i * DURACAO_MAX_TRECHO_MS
                fim_ms    = min(inicio_ms + DURACAO_MAX_TRECHO_MS, total_ms)
                faixa     = audio[inicio_ms:fim_ms]

                self.atualizar_progresso(i + 1, num_trechos)
                self.status_label.config(
                    text=f"🔄 Processando trecho {i+1}/{num_trechos}...", style='Info.TLabel')
                self.log(f"🔄 Trecho {i+1}/{num_trechos}: {self.ms_to_hhmmss(inicio_ms)} - {self.ms_to_hhmmss(fim_ms)}", "info")
                self.log("🔇 Removendo silêncios...", "info")

                partes = split_on_silence(faixa,
                                          min_silence_len=LIMITE_SILENCIO_MS,
                                          silence_thresh=VOLUME_SILENCIO_DBFS,
                                          keep_silence=KEEP_SILENCE_MS)
                if partes:
                    faixa_limpa = sum(partes, AudioSegment.silent(duration=0))
                    self.log(f"✅ Silêncios removidos: {len(partes)} segmentos", "success")
                else:
                    self.log("⚠️ Nenhum silêncio detectado", "warning")
                    faixa_limpa = faixa

                temp_wav = f"temp_trecho_{i:03d}.wav"
                faixa_limpa.export(temp_wav, format="wav")

                self.log(f"🎤 Transcrevendo trecho {i+1}...", "info")
                texto_trecho = []
                try:
                    with sr.AudioFile(temp_wav) as source:
                        while True:
                            audio_data = recognizer.record(source, duration=GOOGLE_SEGMENTO_S)
                            if len(audio_data.frame_data) == 0:
                                break
                            try:
                                parte_txt = recognizer.recognize_google(
                                    audio_data, language=idioma_codigo)
                                if parte_txt.strip():
                                    texto_trecho.append(parte_txt.strip())
                            except sr.UnknownValueError:
                                self.log("⚠️ Google não entendeu um subtrecho", "warning")
                                continue
                            except sr.RequestError as e:
                                self.log(f"❌ Erro na API Google: {e}", "error")
                                break
                finally:
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)

                texto_concatenado = " ".join(texto_trecho).strip()
                self.total_chars += len(texto_concatenado)
                self.chars_label.config(text=f"{self.total_chars:,}")

                transcription_file = f"{self.audio_file}_transcription.txt"
                with open(transcription_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Trecho {i+1} ({self.ms_to_hhmmss(inicio_ms)} - {self.ms_to_hhmmss(fim_ms)})\n")
                    f.write(f"{'='*60}\n")
                    f.write(texto_concatenado + "\n" if texto_concatenado else "[Sem transcrição válida]\n")

                self.log(f"✅ Trecho {i+1} concluído ({len(texto_concatenado)} caracteres)", "success")

            self.log("🎉 Processamento concluído!", "success")
            self.log(f"💾 Arquivo salvo: {transcription_file}", "success")
            self.finalizar_processamento(True, transcription_file)

        except Exception as e:
            self.log(f"❌ ERRO: {str(e)}", "error")
            self.finalizar_processamento(False, str(e))

    def finalizar_processamento(self, sucesso: bool, info: str):
        if sucesso:
            self.atualizar_progresso(self.total_trechos, self.total_trechos)
        self.btn_select.config(state=tk.NORMAL)
        self.btn_process.config(state=tk.NORMAL)
        self.processing = False
        if sucesso:
            self.status_label.config(text="✅ Transcrição concluída com sucesso!", style='Success.TLabel')
            messagebox.showinfo("✅ Sucesso",
                f"Transcrição concluída!\n\n"
                f"📊 Total de caracteres: {self.total_chars:,}\n"
                f"⏱️ Tempo total: {self.tempo_label.cget('text')}\n\n"
                f"Arquivo salvo em:\n{info}")
        else:
            self.status_label.config(text="❌ Erro no processamento", style='Error.TLabel')
            messagebox.showerror("❌ Erro", f"Ocorreu um erro:\n\n{info}")


def main():
    root = tk.Tk()
    app = TranscritorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

