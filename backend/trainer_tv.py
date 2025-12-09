#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiingo Trainer - 专门用于 15m 数据的替代方案
使用 Tiingo API (免费层级支持 15m 数据)
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')
import time
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from huggingface_hub import HfApi
from dotenv import load_dotenv

print("✅ 依赖导入完成")

# ================================================================================
# 配置 - Tiingo API Key 写死在这里
# ================================================================================

load_dotenv()

# 请替换为你的 Tiingo API Key
# 注册地址: https://www.tiingo.com/ (免费，只需邮箱)
TIINGO_API_KEY = "ea4e4b47a135ae917494a1d0dbb6d97d43082f46"

CONFIG = {
    "TRADING_PAIRS": [
        "AAPL",          # 苹果
        "GOOGL",         # 谷歌
        "MSFT",          # 微软
        "AMZN",          # 亚马逊
        "TSLA",          # 特斯拉
        "NVDA",          # 英伟达
        "META",          # Meta
        # 注意: Tiingo 加密货币需要不同的 Endpoint，这里只用于美股
    ],
    "TIMEFRAMES": ["15min", "1hour", "4hour"],  # Tiingo 格式
    "LOOKBACK_DAYS": 365,
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "HF_REPO_ID": os.getenv("HF_REPO_ID", "zongowo111/crypto-trading-bot"),
    "HF_TOKEN": os.getenv("HF_TOKEN", ""),
    "RF_N_ESTIMATORS": 100,
    "RF_MAX_DEPTH": 15,
    "RF_MIN_SAMPLES_SPLIT": 10,
    "API_KEY": TIINGO_API_KEY,
}

print("✅ 配置完成")
print(f"交易对: {CONFIG['TRADING_PAIRS']}")
print(f"时间框架: {CONFIG['TIMEFRAMES']}")

# ================================================================================
# 特征工程 (与之前相同)
# ================================================================================

class TradingStrategy:
    """特征工程类"""
    
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26,
                 macd_signal=9, bb_period=20, bb_std=2):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_rsi(self, data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def calculate_atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_features(self, df):
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
        return [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_mid', 'bb_lower', 'atr',
            'stoch_k', 'stoch_d',
            'sma_20', 'sma_50', 'sma_200',
            'roc', 'volume_ratio', 'high_low_ratio'
        ]

class TargetGenerator:
    @staticmethod
    def generate_labels(df, future_periods=24, threshold=0.02):
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

print("✅ 特征工程类定义完成")

# ================================================================================
# Tiingo 数据获取
# ================================================================================

class TiingoDataFetcher:
    """使用 Tiingo API 获取数据"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }
    
    def fetch_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """获取 Tiingo 历史数据"""
        print(f"   📥 从 Tiingo 获取 {symbol} {timeframe}...", end=" ")
        
        # Tiingo 限制: 50 请求/小时, 1000 请求/天 (免费)
        # 所以我们需要慢一点
        time.sleep(1)
        
        try:
            # 转换时间框架
            resample_freq = None
            if timeframe == "15min":
                resample_freq = "15min"
            elif timeframe == "1hour":
                resample_freq = "1hour"
            elif timeframe == "4hour":
                resample_freq = "4hour"
            
            # Tiingo IEX Endpoint (支持 intraday)
            url = f"https://api.tiingo.com/iex/{symbol}/prices"
            
            params = {
                'startDate': (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'), # Tiingo 限制 intraday 只能拿近期
                'resampleFreq': resample_freq,
                'columns': 'open,high,low,close,volume'
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ API 错误: {response.status_code} {response.text[:50]}")
                return pd.DataFrame()
            
            data = response.json()
            
            if not data:
                print(f"❌ 无数据")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'date': 'timestamp'}, inplace=True)
            df.sort_values('timestamp', inplace=True)
            
            print(f"✅ {len(df)} 根 K 线")
            return df
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            return pd.DataFrame()

# ================================================================================
# 训练与主程序
# ================================================================================

def train_model_for_pair_timeframe(pair: str, timeframe: str, data_fetcher) -> Dict:
    """训练模型"""
    print(f"\n{'='*70}")
    print(f"🤖 训练: {pair} {timeframe}")
    print(f"{'='*70}")
    
    try:
        # 1. 获取数据
        df_raw = data_fetcher.fetch_historical_data(pair, timeframe)
        
        if len(df_raw) < 50: # Tiingo 数据可能较少，降低门槛
            print(f"❌ 数据不足（得到 {len(df_raw)} 根）")
            return {"pair": pair, "timeframe": timeframe, "success": False}
        
        # 2. 计算特征
        print("🔧 计算指标...")
        strategy = TradingStrategy()
        df_features = strategy.calculate_features(df_raw)
        
        if len(df_features) < 30:
            print(f"❌ 特征计算后数据不足")
            return {"pair": pair, "timeframe": timeframe, "success": False}
        
        # 3. 生成标签
        label_generator = TargetGenerator()
        df_features['label'] = label_generator.generate_labels(df_features)
        
        # 4. 训练
        feature_columns = strategy.get_feature_columns()
        X = df_features[feature_columns].values
        y = df_features['label'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=CONFIG["TEST_SIZE"],
            random_state=CONFIG["RANDOM_STATE"], stratify=y
        )
        
        print("🤖 训练模型...")
        model = RandomForestClassifier(
            n_estimators=CONFIG["RF_N_ESTIMATORS"],
            max_depth=CONFIG["RF_MAX_DEPTH"], 
            random_state=CONFIG["RANDOM_STATE"],
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"📈 训练准确率: {train_acc:.2%} | 测试准确率: {test_acc:.2%}")
        
        # 保存
        os.makedirs('./models', exist_ok=True)
        pair_clean = pair.replace('/', '_')
        tf_clean = timeframe.replace('min', 'm').replace('hour', 'h') # 统一文件名格式
        
        model_filename = f"model_{pair_clean}_{tf_clean}.pkl"
        scaler_filename = f"scaler_{pair_clean}_{tf_clean}.pkl"
        
        with open(f'./models/{model_filename}', 'wb') as f: pickle.dump(model, f)
        with open(f'./models/{scaler_filename}', 'wb') as f: pickle.dump(scaler, f)
        
        return {
            "pair": pair, "timeframe": timeframe, "success": True,
            "model_path": f'./models/{model_filename}',
            "scaler_path": f'./models/{scaler_filename}',
            "model_filename": model_filename,
            "scaler_filename": scaler_filename
        }
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        return {"pair": pair, "timeframe": timeframe, "success": False, "error": str(e)}

def upload_models_to_hf(results):
    if not CONFIG["HF_TOKEN"]: return
    print(f"\n📤 上传到 Hugging Face...")
    api = HfApi(token=CONFIG["HF_TOKEN"])
    for r in results:
        if r.get("success"):
            try:
                print(f"   {r['model_filename']}...", end=" ")
                api.upload_file(path_or_fileobj=r["model_path"], path_in_repo=r["model_filename"], repo_id=CONFIG["HF_REPO_ID"], repo_type="model")
                api.upload_file(path_or_fileobj=r["scaler_path"], path_in_repo=r["scaler_filename"], repo_id=CONFIG["HF_REPO_ID"], repo_type="model")
                print("✅")
            except: print("❌")

def main():
    print(f"\n🚀 开始训练 - Tiingo 版本（支持 15m）")
    if CONFIG["API_KEY"] == "your_tiingo_api_key_here":
        print("❌ 请先设置 CONFIG 中的 TIINGO_API_KEY")
        return

    data_fetcher = TiingoDataFetcher(CONFIG["API_KEY"])
    all_results = []
    
    for pair in CONFIG["TRADING_PAIRS"]:
        for timeframe in CONFIG["TIMEFRAMES"]:
            result = train_model_for_pair_timeframe(pair, timeframe, data_fetcher)
            all_results.append(result)
            
    upload_models_to_hf(all_results)
    print("\n✅ 所有任务完成！")

if __name__ == "__main__":
    main()
