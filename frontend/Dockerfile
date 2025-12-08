FROM python:3.11-slim

WORKDIR /app

# 複製依賴
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 預設啟動命令：跑 Discord bot
CMD ["python", "bot.py"]
