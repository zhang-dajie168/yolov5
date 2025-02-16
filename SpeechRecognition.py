import deepspeech
import wave

# # 定义模型和语音文件的路径
# model_path = 'path/to/your/deepspeech-model.pbmm'
# scorer_path = 'path/to/your/deepspeech-scorer.scorer'
audio_file = 'path/to/your/audio/file.wav'

# 初始化 DeepSpeech 模型
model = deepspeech.Model()

model.enableExternalScorer()

# 打开语音文件
with wave.open(audio_file, 'rb') as audio:
    audio_data = audio.readframes(audio.getnframes())
    sample_rate = audio.getframerate()

# 进行语音转文本
text = model.stt(audio_data)

# 输出识别结果
print("识别结果:", text)
