import subprocess
import os

# Use o caminho real onde o ffmpeg.exe está guardado
ffmpeg_path = r"[PATH TO FFMEP EXECUTABLE]" 
input_file = r"[PATH TO FILE]"
output_file = "video_comprimido.mp4"

# Garante que o Python "enxergue" a pasta do ffmpeg para achar as DLLs
env = os.environ.copy()
env["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

command = [ffmpeg_path, "-i", input_file, "-c:v", "libx264", "-crf", "23", output_file]

try:
    subprocess.run(command, check=True, env=env)
    print("Sucesso!")
except subprocess.CalledProcessError as e:
    print(f"Erro: {e}")

