# Chạy AudioInteraction trên máy local có GPU và kết nối agent

Mục tiêu là để GPU local giữ model trong RAM/VRAM, còn voice agent chỉ gửi một
file audio hoàn chỉnh tới endpoint HTTP. Cách này thay thế vòng lặp Kaggle đang
poll `GET /audio/latest` rồi khởi tạo lại model ở mỗi audio.

## 1. Cài đặt (Windows PowerShell)

```powershell
conda create -n audiointeraction python=3.12 -y
conda activate audiointeraction
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
$env:PYTHONPATH = (Get-Location).Path
python download.py
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Chọn bản PyTorch CUDA phù hợp driver NVIDIA của máy; `torch.cuda.is_available()`
phải in `True`. Weights phải nằm tại `./checkpoints`.

## 2. Kiểm tra offline

```powershell
python infer_offline.py --input "D:\audio\chunk.wav" --json
```

Dòng cuối là JSON để agent parse, ví dụ `reply_text` và `environment`.
`environment` là tag suy luận từ câu trả lời của model, không phải bộ phân loại
độc lập; luôn kiểm tra `analysis_valid` trước khi dùng để kích hoạt workflow.

## 3. Khởi động HTTP bridge giữ model trên GPU

```powershell
python agent_server.py --host 0.0.0.0 --port 8000
```

Server load weights đúng một lần khi boot. Kiểm tra từ cùng máy:

```powershell
curl http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/analyze -F "audio=@D:\audio\chunk.wav"
```

## 4. Sửa phần agent để gọi local GPU

Khi agent đã kết thúc một utterance/chunk WAV, gửi multipart `POST` tới
`http://<IP-may-GPU>:8000/analyze`, field bắt buộc là `audio`. Ví dụ Python:

```python
with open(wav_path, "rb") as f:
    result = requests.post(
        "http://192.168.1.50:8000/analyze",
        files={"audio": ("utterance.wav", f, "audio/wav")},
        timeout=180,
    ).json()

if result["environment"]["analysis_valid"]:
    print(result["reply_text"])
```

Nếu agent và GPU không cùng mạng LAN, expose port 8000 bằng một tunnel có xác
thực (ví dụ Cloudflare Tunnel/ngrok) và dùng URL HTTPS của tunnel. Không nên
expose endpoint không authentication ra Internet: audio có thể chứa dữ liệu
riêng tư. Hiện bridge xử lý tuần tự một request vì một GPU/KV cache chỉ phục vụ
an toàn một inference tại một thời điểm.

Không gửi mỗi frame 400 ms vào `/analyze`: đây là endpoint offline theo
utterance. Với streaming microphone thật thời gian thực, chạy `python
web/server.py` và dùng frontend WebUI có sẵn; việc nối trực tiếp LiveKit frame
vào streaming protocol cần adapter riêng vì model cần duy trì KV cache theo
session.
