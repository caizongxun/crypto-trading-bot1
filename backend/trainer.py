# backend/trainer.py
# 本地執行版本 - 使用 Binance API
# 在 PyCharm 或本地終端執行

import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

import ccxt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from huggingface_hub import HfApi
from dotenv import load_dotenv
import time

print("✅ 依賴導入完成")

# ================================================================================
# 配置
# ================================================================================

load_dotenv()

CONFIG = {
    "TRADING_PAIRS": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT"
    ],
    "TIMEFRAMES": ["15m", "1h", "4h"],
    "LOOKBACK_DAYS": 365,
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    
    "HF_REPO_ID": os.getenv("HF_REPO_ID", "zongowo111/crypto-trading-bot"),
    "HF_TOKEN": os.getenv("HF_TOKEN", "your_hf_token_here"),
    
    "RF_N_ESTIMATORS": 100,
    "RF_MAX_DEPTH": 15,
    "RF_MIN_SAMPLES_SPLIT": 10,
}

print("✅ 配置完成")
print(f"交易對: {CONFIG['TRADING_PAIRS']}")
print(f"時間框架: {CONFIG['TIMEFRAMES']}")
print(f"使用交易所: Binance")

# ================================================================================
# 特徵工程
# ================================================================================

class TradingStrategy:
    """特徵工程類"""
    
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, 
                 macd_signal=9, bb_period=20, bb_std=2):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_rsi(self, data, period=14):
        """計算 RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """計算 MACD"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """計算布林帶"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def calculate_atr(self, high, low, close, period=14):
        """計算 ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """計算 Stochastic"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_features(self, df):
        """計算所有技術指標"""
        features = df.copy()
        
        features['rsi'] = self.calculate_rsi(features['close'], self.rsi_period)
        
        features['macd'], features['macd_signal'], features['macd_hist'] = self.calculate_macd(
            features['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal
        )
        
        features['bb_upper'], features['bb_mid'], features['bb_lower'] = self.calculate_bollinger_bands(
            features['close'], period=self.bb_period, std_dev=self.bb_std
        )
        
        features['atr'] = self.calculate_atr(
            features['high'], features['low'], features['close'], period=14
        )
        
        features['stoch_k'], features['stoch_d'] = self.calculate_stochastic(
            features['high'], features['low'], features['close'], k_period=14, d_period=3
        )
        
        features['sma_20'] = features['close'].rolling(window=20).mean()
        features['sma_50'] = features['close'].rolling(window=50).mean()
        features['sma_200'] = features['close'].rolling(window=200).mean()
        
        features['roc'] = features['close'].pct_change(periods=12)
        
        features['volume_sma'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / (features['volume_sma'] + 1e-9)
        
        features['high_low_ratio'] = (features['high'] - features['low']) / (features['close'] + 1e-9)
        
        features = features.dropna()
        
        return features
    
    def get_feature_columns(self):
        """返回特徵列"""
        return [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_mid', 'bb_lower', 'atr',
            'stoch_k', 'stoch_d',
            'sma_20', 'sma_50', 'sma_200',
            'roc', 'volume_ratio', 'high_low_ratio'
        ]


class TargetGenerator:
    """目標標籤生成器"""
    
    @staticmethod
    def generate_labels(df, future_periods=24, threshold=0.02):
        """生成標籤"""
        labels = []
        
        for i in range(len(df) - future_periods):
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + future_periods]['close']
            price_change = (future_price - current_price) / current_price
            
            if price_change > threshold:
                labels.append(1)
            elif price_change < -threshold:
                labels.append(-1)
            else:
                labels.append(0)
        
        labels.extend([0] * future_periods)
        return np.array(labels)

print("✅ 特徵工程類定義完成")

# ================================================================================
# 數據抓取
# ================================================================================

class DataFetcher:
    """從 Binance 抓取數據"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
    
    def fetch_historical_data(self, symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
        """抓取數據"""
        print(f"\n📊 抓取 {symbol} {timeframe}...")
        
        timeframe_map = {"15m": 96, "1h": 24, "4h": 6, "1d": 1}
        limit = days * timeframe_map.get(timeframe, 24)
        
        all_candles = []
        batch_size = 1000
        
        for i in range(0, limit, batch_size):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, limit=min(batch_size, limit - i)
                )
                all_candles.extend(ohlcv)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️ 抓取失敗: {str(e)[:50]}")
                break
        
        if len(all_candles) == 0:
            print(f"❌ {symbol} 無數據")
            return pd.DataFrame()
        
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        
        print(f"✅ 共 {len(df)} 根 K線")
        
        return df

# ================================================================================
# 訓練
# ================================================================================

def train_model_for_pair_timeframe(pair: str, timeframe: str) -> Dict:
    """訓練模型"""
    
    print(f"\n{'='*70}")
    print(f"🤖 訓練: {pair} {timeframe}")
    print(f"{'='*70}")
    
    try:
        # 1. 抓取數據
        fetcher = DataFetcher()
        df_raw = fetcher.fetch_historical_data(pair, timeframe, CONFIG["LOOKBACK_DAYS"])
        
        if len(df_raw) < 100:
            return {"pair": pair, "timeframe": timeframe, "success": False}
        
        # 2. 計算特徵
        print("🔧 計算指標...")
        strategy = TradingStrategy()
        df_features = strategy.calculate_features(df_raw)
        
        if len(df_features) < 50:
            return {"pair": pair, "timeframe": timeframe, "success": False}
        
        # 3. 生成標籤
        print("🏷️ 生成標籤...")
        label_generator = TargetGenerator()
        timeframe_periods = {"15m": 96, "1h": 24, "4h": 6}
        future_periods = timeframe_periods.get(timeframe, 24)
        df_features['label'] = label_generator.generate_labels(
            df_features, future_periods=future_periods, threshold=0.02
        )
        
        # 4. 預處理
        print("🔄 準備數據...")
        feature_columns = strategy.get_feature_columns()
        X = df_features[feature_columns].values
        y = df_features['label'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=CONFIG["TEST_SIZE"],
            random_state=CONFIG["RANDOM_STATE"], stratify=y
        )
        
        # 5. 訓練
        print("🤖 訓練模型...")
        model = RandomForestClassifier(
            n_estimators=CONFIG["RF_N_ESTIMATORS"],
            max_depth=CONFIG["RF_MAX_DEPTH"],
            min_samples_split=CONFIG["RF_MIN_SAMPLES_SPLIT"],
            random_state=CONFIG["RANDOM_STATE"],
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 6. 評估
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"訓練準確率: {train_accuracy:.4f}")
        print(f"測試準確率: {test_accuracy:.4f}")
        
        # 7. 保存
        print("💾 保存...")
        os.makedirs('./models', exist_ok=True)
        
        pair_clean = pair.replace('/', '_')
        model_filename = f"model_{pair_clean}_{timeframe}.pkl"
        scaler_filename = f"scaler_{pair_clean}_{timeframe}.pkl"
        
        with open(f'./models/{model_filename}', 'wb') as f:
            pickle.dump(model, f)
        
        with open(f'./models/{scaler_filename}', 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✅ 完成")
        
        return {
            "pair": pair,
            "timeframe": timeframe,
            "success": True,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "model_path": f'./models/{model_filename}',
            "scaler_path": f'./models/{scaler_filename}',
            "model_filename": model_filename,
            "scaler_filename": scaler_filename,
        }
    
    except Exception as e:
        print(f"❌ 失敗: {e}")
        return {"pair": pair, "timeframe": timeframe, "success": False}

# ================================================================================
# 批量訓練
# ================================================================================

print(f"\n{'='*70}")
print(f"🚀 開始批量訓練")
print(f"{'='*70}")

all_results = []

for pair in CONFIG["TRADING_PAIRS"]:
    for timeframe in CONFIG["TIMEFRAMES"]:
        result = train_model_for_pair_timeframe(pair, timeframe)
        all_results.append(result)

# ================================================================================
# 上傳到 Hugging Face
# ================================================================================

print(f"\n{'='*70}")
print(f"🚀 上傳到 Hugging Face")
print(f"{'='*70}")

try:
    api = HfApi(token=CONFIG["HF_TOKEN"])
    upload_count = 0
    
    for result in all_results:
        if not result["success"]:
            continue
        
        pair = result["pair"]
        timeframe = result["timeframe"]
        
        print(f"\n上傳 {pair} {timeframe}...")
        
        try:
            api.upload_file(
                path_or_fileobj=result["model_path"],
                path_in_repo=result["model_filename"],
                repo_id=CONFIG["HF_REPO_ID"],
                repo_type="model"
            )
            
            api.upload_file(
                path_or_fileobj=result["scaler_path"],
                path_in_repo=result["scaler_filename"],
                repo_id=CONFIG["HF_REPO_ID"],
                repo_type="model"
            )
            
            upload_count += 1
            print(f"✅ {pair} {timeframe} 上傳成功")
        
        except Exception as e:
            print(f"⚠️ 上傳失敗: {str(e)[:50]}")
    
    print(f"\n✅ 共上傳 {upload_count} 個模型")

except Exception as e:
    print(f"❌ 上傳失敗: {e}")

# ================================================================================
# 總結
# ================================================================================

print(f"\n{'='*70}")
print(f"✅ 訓練完成！")
print(f"{'='*70}")

success_count = sum(1 for r in all_results if r["success"])
total_count = len(all_results)

print(f"""
📊 訓練摘要:
├─ 成功: {success_count}/{total_count}
├─ 失敗: {total_count - success_count}/{total_count}

✨ 準備好部署了！
""")
