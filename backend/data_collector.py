import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ===== 設定區 (請修改這裡) =====
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = '../service_account.json'  # 指向上一層目錄的 json
DRIVE_FOLDER_ID = '1A4Fqe5wNN26CytRihxjgjJQNgM4__fro'  # <--- 這裡要改！！

# 想要抓取的標的
PAIRS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "MATIC-USD",
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"
]
INTERVAL = "15m"  # 收集 15分K
LOOKBACK = "5d"  # 每次抓最近 5 天


def authenticate_gdrive():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service


def fetch_data():
    all_data = {}
    print(f"\n🔄 [{datetime.now()}] 開始抓取數據...")

    for pair in PAIRS:
        try:
            # 下載數據
            df = yf.download(pair, period=LOOKBACK, interval=INTERVAL, progress=False, auto_adjust=False)

            if len(df) > 0:
                # 清理格式
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).lower() for c in df.columns]
                df.reset_index(inplace=True)

                # 暫存檔名
                filename = f"{pair.replace('-', '_')}_{INTERVAL}.csv"
                df.to_csv(filename, index=False)
                all_data[pair] = filename
                print(f"  ✅ {pair}: {len(df)} 筆")
            else:
                print(f"  ⚠️ {pair}: 無數據")

        except Exception as e:
            print(f"  ❌ {pair} 失敗: {e}")

    return all_data


def upload_to_drive(service, file_map):
    print("\n☁️ 正在上傳到 Google Drive...")

    # 檢查雲端已有的檔案
    results = service.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false",
        fields="files(id, name)").execute()
    existing_files = {f['name']: f['id'] for f in results.get('files', [])}

    for pair, filename in file_map.items():
        file_metadata = {'name': filename, 'parents': [DRIVE_FOLDER_ID]}
        media = MediaFileUpload(filename, mimetype='text/csv')

        try:
            if filename in existing_files:
                # 更新
                file_id = existing_files[filename]
                service.files().update(
                    fileId=file_id,
                    media_body=media
                ).execute()
                print(f"  🔄 更新: {filename}")
            else:
                # 新增
                service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"  ➕ 新增: {filename}")
        except Exception as e:
            print(f"  ❌ 上傳失敗 {filename}: {e}")
        finally:
            # 刪除本地暫存檔
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    # 檢查 json 是否存在
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"❌ 錯誤: 找不到 {SERVICE_ACCOUNT_FILE}，請確認檔案位置。")
        exit(1)

    data_files = fetch_data()

    if data_files:
        try:
            drive_service = authenticate_gdrive()
            upload_to_drive(drive_service, data_files)
            print("\n🎉 全部完成！")
        except Exception as e:
            print(f"❌ Drive 連線失敗: {e}")
