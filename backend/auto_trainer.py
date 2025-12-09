import os
import glob
import pickle
import shutil
import pandas as pd
import numpy as np
from huggingface_hub import snapshot_download, HfApi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from strategy import TradingStrategy  # 確保 strategy.py 在同一目錄或 Python 路徑中

# ===== 設定 =====
HF_DATA_REPO = "zongowo111/crypto-data"   # 資料來源 (Dataset)
HF_MODEL_REPO = "zongowo111/crypto-trading-bot" # 模型去處 (Model)
HF_TOKEN = os.getenv("HF_TOKEN")

TEMP_DATA_DIR = "./temp_dataset"
MODEL_DIR = "./models"

def train_and_upload():
    print("🚀 [Step 1] 下載數據集...")
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
    
    try:
        # 從 HF Dataset 下載所有 CSV
        snapshot_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            local_dir=TEMP_DATA_DIR,
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"❌ 下載數據集失敗: {e}")
        return
    
    print("🚀 [Step 2] 開始訓練...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # 搜尋所有 CSV 檔案
    csv_files = glob.glob(f"{TEMP_DATA_DIR}/*.csv")
    print(f"找到 {len(csv_files)} 個數據文件")

    if not csv_files:
        print("⚠️ 未找到任何 CSV 文件，跳過訓練。")
        return

    strategy = TradingStrategy()
    
    trained_count = 0
    for csv_path in csv_files:
        try:
            # 解析檔名: BTC_USD_15m.csv
            filename = os.path.basename(csv_path)
            pair_tf = filename.replace(".csv", "") # BTC_USD_15m
            
            print(f"訓練中: {pair_tf}...")
            
            df = pd.read_csv(csv_path)
            
            # 特徵工程 (必須與 Bot 一致!)
            df = strategy.calculate_features(df)
            if len(df) < 100:
                print(f"⚠️ {pair_tf} 數據不足 (<100)，跳過")
                continue
            
            # 建立目標 (Target)
            # 簡單邏輯：未來3根K線漲幅 > 1.5%
            threshold = 0.015
            future_returns = df['close'].shift(-3) / df['close'] - 1
            conditions = [
                (future_returns > threshold),
                (future_returns < -threshold)
            ]
            df['target'] = np.select(conditions, [1, -1], default=0)
            
            # 清理 NaN
            df.dropna(inplace=True)
            
            # 準備訓練數據
            feature_cols = strategy.get_feature_columns()
            
            # 檢查特徵是否存在
            if not all(col in df.columns for col in feature_cols):
                print(f"⚠️ {pair_tf} 特徵缺失，跳過")
                continue

            X = df[feature_cols]
            y = df['target']
            
            if len(np.unique(y)) < 2:
                print(f"⚠️ {pair_tf} 只有單一類別，跳過")
                continue
            
            # 訓練
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            try:
                smote = SMOTE()
                X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
            except:
                X_res, y_res = X_train_scaled, y_train
                
            model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
            model.fit(X_res, y_res)
            
            # 保存
            with open(f"{MODEL_DIR}/model_{pair_tf}.pkl", 'wb') as f:
                pickle.dump(model, f)
            with open(f"{MODEL_DIR}/scaler_{pair_tf}.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            
            trained_count += 1
            
        except Exception as e:
            print(f"❌ 訓練失敗 {csv_path}: {e}")

    if trained_count > 0:
        print(f"🚀 [Step 3] 上傳 {trained_count} 個新模型到 Hugging Face...")
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(
                folder_path=MODEL_DIR,
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                path_in_repo=".",
                commit_message="Auto-retrained models from GitHub Actions"
            )
            print("🎉 訓練與更新完成！")
        except Exception as e:
            print(f"❌ 上傳失敗: {e}")
    else:
        print("⚠️ 沒有模型被成功訓練，跳過上傳。")

if __name__ == "__main__":
    if not HF_TOKEN:
        print("❌ 錯誤: 未設定 HF_TOKEN 環境變數")
    else:
        train_and_upload()