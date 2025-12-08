FROM python:3.11-slim

WORKDIR /app

# 複製依賴（從 frontend/ 目錄）
COPY frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼（從 frontend/ 目錄）
COPY frontend/ .

# 預設啟動命令
CMD ["python", "bot.py"]
