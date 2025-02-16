from pathlib import Path

from paddlespeech.cli.asr.infer import ASRExecutor

asr = ASRExecutor()

audio_path = Path("/home/ymt/项目文件/SL008-YJ001-PFJC/yolov5/hege.wav")

result = asr(audio_file=audio_path,sample_rate=10000,force_yes=True)

print(result)
