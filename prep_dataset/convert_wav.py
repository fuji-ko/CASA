"""mp4ファイルを16kHzモノラルのwavファイルに変換"""
import os
import glob
from pydub import AudioSegment
import sys

input_dir = "/work/abelab5/k_fuji/CASA/data/Voices-AWS/reading/videos"
output_dir = "/work/abelab5/k_fuji/CASA/data/Voices-AWS/reading/audios"

video_list = glob.glob(os.path.join(input_dir, "*.mp4"))
for input_file in video_list:
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000).set_channels(1) # 16kHz, モノラルに変換
    audio_name = os.path.basename(input_file).split(".")[0] + ".wav"
    audio.export(os.path.join(output_dir, audio_name), format="wav")
    print(f"converted: {audio_name}")
print("finished")