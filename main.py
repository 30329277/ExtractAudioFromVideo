import tkinter as tk
from tkinter import filedialog, messagebox
from moviepy.editor import AudioFileClip
import os
import deepspeech
import numpy as np
import wave
import threading
import logging
import unicodedata
from concurrent.futures import ThreadPoolExecutor

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_audio_chunk(model, chunk):
    try:
        chunk_text = model.stt(chunk)
        return chunk_text
    except Exception as e:
        logging.error(f"Error in transcribe_audio_chunk: {e}")
        return ""

def transcribe_audio(audio_filename, num_threads=4):
    try:
        # 加载 DeepSpeech 模型
        # model_file_path = 'models\\deepspeech-0.9.3-models-zh-CN.pbmm'
        model_file_path = 'models\\deepspeech-0.9.3-models.pbmm'
        model = deepspeech.Model(model_file_path)
        logging.info(f"DeepSpeech model loaded from {model_file_path}")

        # 可选：加载 scorer 文件以提高准确性
        # scorer_file_path = 'models\\deepspeech-0.9.3-models-zh-CN.scorer'
        scorer_file_path = 'models\\deepspeech-0.9.3-models.scorer'
        model.enableExternalScorer(scorer_file_path)
        logging.info(f"Scorer enabled from {scorer_file_path}")

        # 读取 WAV 文件
        with wave.open(audio_filename, 'r') as wav:
            rate = wav.getframerate()
            frames = wav.getnframes()
            buffer = wav.readframes(frames)
            data16 = np.frombuffer(buffer, dtype=np.int16)
            logging.info(f"Audio file read: {audio_filename}, Frame Rate: {rate}, Frames: {frames}")

        # 分块处理音频
        chunk_size = 16000 * 60  # 每块 1 分钟
        text = ""
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(0, len(data16), chunk_size):
                chunk = data16[i:i + chunk_size]
                logging.info(f"Processing chunk: {i} to {i + chunk_size}")
                future = executor.submit(transcribe_audio_chunk, model, chunk)
                futures.append(future)

            for future in futures:
                chunk_text = future.result()
                text += chunk_text
                logging.info(f"Chunk transcription: {chunk_text}")

        logging.info(f"Transcription completed: {text}")
        return text
    except Exception as e:
        logging.error(f"Error in transcribe_audio: {e}")
        return None

def open_folder():
    folder_selected = filedialog.askdirectory()
    if not folder_selected:
        return
    for filename in os.listdir(folder_selected):
        if filename.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            video_path = os.path.join(folder_selected, filename)
            audio_button = tk.Button(window, text=f"Get the audio from {filename}",
                                     command=lambda path=video_path: process_video(path))
            audio_button.pack()

def process_video(video_path):
    thread = threading.Thread(target=get_audio, args=(video_path,))
    thread.start()

def get_audio(video_path):
    try:
        # 提取音频
        audio_clip = AudioFileClip(video_path)
        audio_filename = os.path.splitext(video_path)[0] + ".wav"
        audio_clip.write_audiofile(audio_filename, codec='pcm_s16le')  # 使用 PCM 编码保存为 WAV 文件
        logging.info(f"Audio extracted to {audio_filename}")

        # 转换成文本
        text = transcribe_audio(audio_filename)  # 确保transcribe_audio函数已定义
        if text is not None:
            # 过滤无效字符
            valid_text = ''.join(c for c in text if unicodedata.category(c) != 'Cs')
            with open(os.path.splitext(video_path)[0] + ".txt", 'w', encoding='utf-8') as f:
                f.write(valid_text)
            logging.info(f"Text written to {os.path.splitext(video_path)[0] + '.txt'}")
            messagebox.showinfo("Success", "Audio extracted and converted to text successfully!")
        else:
            messagebox.showerror("Error", "Failed to transcribe audio.")
    except Exception as e:
        logging.error(f"Error in get_audio: {e}")
        messagebox.showerror("Error", str(e))

# 创建主窗口
window = tk.Tk()
window.title("Video Handler")
window.geometry("300x500")  # 设置窗口大小

# 添加按钮
open_button = tk.Button(window, text="Open a local folder", command=open_folder)
open_button.pack()

# 运行主循环
window.mainloop()