# backend/data_collector.py
import os
import shutil
import pandas as pd
import yfinance as yf
from datetime import datetime
from huggingface_hub import HfApi
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# ===== 設定區 =====
HF_DATA_REPO = "zongowo111/crypto-data"  # <--- 請確認這裡跟你在 HF 建立的名字一樣
HF_TOKEN = os.getenv("HF_TOKEN")

# 想要抓取的標的
PAIRS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "MATIC-USD",
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"
]

INTERVAL = "15m"  # 15分K
LOOKBACK = "5d"  # 抓最近 5 天 (覆蓋舊的保證數據連續)
TEMP_DIR = "temp_crypto_data"  # 暫存資料夾名稱


def fetch_and_upload():
    print(f"\n🔄 [{datetime.now()}] 開始抓取數據...")

    # 1. 建立乾淨的暫存資料夾
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    file_count = 0

    # 2. 抓取所有幣種數據
    for pair in PAIRS:
        try:
            # yfinance 下載
            df = yf.download(pair, period=LOOKBACK, interval=INTERVAL, progress=False, auto_adjust=False)

            if len(df) > 0:
                # 簡單清理格式
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).lower() for c in df.columns]
                df.reset_index(inplace=True)

                # 存入暫存資料夾
                filename = f"{pair.replace('-', '_')}_{INTERVAL}.csv"
                filepath = os.path.join(TEMP_DIR, filename)
                df.to_csv(filepath, index=False)

                file_count += 1
                print(f"  ✅ {pair}: {len(df)} 筆 -> {filename}")
            else:
                print(f"  ⚠️ {pair}: 無數據")

        except Exception as e:
            print(f"  ❌ {pair} 失敗: {e}")

    # 3. 一次性批量上傳 (Bulk Upload)
    if file_count > 0:
        print(f"\n☁️ 準備上傳 {file_count} 個檔案到 Hugging Face Dataset...")
        try:
            api = HfApi(token=HF_TOKEN)

            api.upload_folder(
                folder_path=TEMP_DIR,  # 上傳整個資料夾
                repo_id=HF_DATA_REPO,
                repo_type="dataset",  # 指定是 dataset
                path_in_repo=".",  # 放在 repo 根目錄
                commit_message=f"Auto-update data {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            print("🎉 上傳成功！(Single Commit)")

        except Exception as e:
            print(f"❌ 上傳失敗: {e}")
    else:
        print("⚠️ 沒有數據被抓取，跳過上傳。")

    # 4. 清理殘留檔案
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print("🧹 暫存檔已清理")


if __name__ == "__main__":
    if not HF_TOKEN:
        print("❌ 錯誤: 未找到 HF_TOKEN，請檢查 .env 檔案。")
    else:
        fetch_and_upload()
