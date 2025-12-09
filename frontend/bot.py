#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Trading Bot - Discord Bot 版本（完整版）
支持多交易對、多時間框架、使用訓練的 ML 模型生成真實信號
"""

import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
import discord
from discord.ext import commands, tasks
from flask import Flask
import threading
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 导入你的 strategy.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
try:
    from strategy import TradingStrategy, TargetGenerator
except ImportError:
    logging.warning("⚠️ Cannot import strategy.py, using dummy strategy")

# ===== 配置日誌 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== 載入環境變數 =====
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

# 驗證環境變數
if not all([DISCORD_TOKEN, DISCORD_CHANNEL_ID, HF_REPO_ID, HF_TOKEN]):
    logger.error("❌ 錯誤：缺少必要的環境變數")
    logger.error(" 需要: DISCORD_TOKEN, DISCORD_CHANNEL_ID, HF_REPO_ID, HF_TOKEN")
    exit(1)

try:
    DISCORD_CHANNEL_ID = int(DISCORD_CHANNEL_ID)
except ValueError:
    logger.error("❌ 錯誤：DISCORD_CHANNEL_ID 必須是數字")
    exit(1)

# ===== Discord Bot 配置 =====
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

# 禁用內建 help 指令，避免衝突
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# ===== 配置 - 從 .env 讀取 =====
def get_env_list(key, default):
    val = os.getenv(key)
    if not val:
        return default
    return [x.strip() for x in val.split(',')]

CONFIG = {
    "trading_pairs": get_env_list("TRADING_PAIRS", [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "BTC-USD"
    ]),
    # 這裡包含所有可能的時間框架 (美股 1d, 加密 4h)
    "timeframes": ["15m", "1h", "4h", "1d"],
    "model_dir": "./models",
    "hf_repo_id": HF_REPO_ID,
    "hf_token": HF_TOKEN,
    "discord_channel_id": DISCORD_CHANNEL_ID,
}

# ===== 交易信號參數配置（可調整）=====
SIGNAL_CONFIG = {
    "model_confidence_threshold": 0.55,  # 模型預測概率閾值
    "min_samples": 100,  # 最少需要多少根 K 線來計算特徵
    "signal_type": "buy",  # "buy", "sell", 或 "both"
}

# ===== 模型管理 =====
class ModelManager:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_dir = CONFIG["model_dir"]
        Path(self.model_dir).mkdir(exist_ok=True)

    def get_model_filename(self, pair, timeframe):
        """生成模型文件名"""
        pair_clean = pair.replace('/', '_').replace('^', '').replace('=', '_').replace('-', '_')
        return f"model_{pair_clean}_{timeframe}.pkl"
    
    def get_scaler_filename(self, pair, timeframe):
        """生成 scaler 文件名"""
        pair_clean = pair.replace('/', '_').replace('^', '').replace('=', '_').replace('-', '_')
        return f"scaler_{pair_clean}_{timeframe}.pkl"

    def download_all_models(self):
        """下載所有模型"""
        logger.info("📥 開始下載所有模型...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error("❌ 缺少 huggingface-hub 包")
            return False

        total = len(CONFIG["trading_pairs"]) * len(CONFIG["timeframes"])
        downloaded = 0

        for pair in CONFIG["trading_pairs"]:
            for timeframe in CONFIG["timeframes"]:
                
                # 這裡也要加上判斷，避免嘗試下載不存在的組合
                is_crypto = '/' in pair
                if is_crypto and timeframe == '1d':
                    continue
                if not is_crypto and timeframe == '4h':
                    continue

                model_filename = self.get_model_filename(pair, timeframe)
                scaler_filename = self.get_scaler_filename(pair, timeframe)
                
                model_path = os.path.join(self.model_dir, model_filename)
                scaler_path = os.path.join(self.model_dir, scaler_filename)

                try:
                    # 下載模型
                    if not os.path.exists(model_path):
                        hf_hub_download(
                            repo_id=CONFIG["hf_repo_id"],
                            filename=model_filename,
                            local_dir=self.model_dir,
                            token=CONFIG["hf_token"]
                        )
                    
                    # 下載 scaler
                    if not os.path.exists(scaler_path):
                        hf_hub_download(
                            repo_id=CONFIG["hf_repo_id"],
                            filename=scaler_filename,
                            local_dir=self.model_dir,
                            token=CONFIG["hf_token"]
                        )
                    
                    downloaded += 1

                except Exception as e:
                    pass

        logger.info(f"📊 下載完成，本地共有 {downloaded} 組模型")
        return downloaded > 0

    def load_model(self, pair, timeframe):
        """載入模型和 scaler"""
        model_filename = self.get_model_filename(pair, timeframe)
        scaler_filename = self.get_scaler_filename(pair, timeframe)
        
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)

        if not os.path.exists(model_path):
            return None, None

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            
            return model, scaler

        except Exception as e:
            logger.error(f"❌ 載入模型失敗 {pair} {timeframe}: {str(e)}")
            return None, None

# ===== 交易信號邏輯 =====
class SignalGenerator:
    def __init__(self):
        self.strategy = TradingStrategy()
    
    def generate_signal(self, pair, timeframe, model, scaler, historical_data_df):
        """生成交易信號"""
        
        if model is None:
            return {
                "pair": pair,
                "timeframe": timeframe,
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "No model available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # 計算特徵
            if len(historical_data_df) < SIGNAL_CONFIG["min_samples"]:
                return {
                    "pair": pair,
                    "timeframe": timeframe,
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reason": f"Insufficient data ({len(historical_data_df)})",
                    "timestamp": datetime.now().isoformat()
                }
            
            features_df = self.strategy.calculate_features(historical_data_df)
            
            if len(features_df) == 0:
                return {
                    "pair": pair,
                    "timeframe": timeframe,
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reason": "Feature calc failed",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 取最後一行數據
            feature_columns = self.strategy.get_feature_columns()
            # 確保特徵欄位對齊
            current_features = [c for c in features_df.columns if c in feature_columns]
            if not current_features:
                 return {"pair": pair, "action": "HOLD", "confidence": 0.0, "reason": "Features mismatch"}

            latest_features = features_df[feature_columns].iloc[-1].values.reshape(1, -1)
            
            # 標準化特徵
            if scaler is not None:
                latest_features = scaler.transform(latest_features)
            
            # 模型預測
            prediction = model.predict(latest_features)[0]  # 0, 1, -1
            
            # 獲取預測概率
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(latest_features)[0]
                confidence = np.max(probas)
            else:
                confidence = 0.5
            
            # 決定是否發送信號
            action = "HOLD"
            if prediction == 1 and confidence >= SIGNAL_CONFIG["model_confidence_threshold"]:
                if SIGNAL_CONFIG["signal_type"] in ["buy", "both"]:
                    action = "BUY"
            elif prediction == -1 and confidence >= SIGNAL_CONFIG["model_confidence_threshold"]:
                if SIGNAL_CONFIG["signal_type"] in ["sell", "both"]:
                    action = "SELL"
            
            current_price = historical_data_df['close'].iloc[-1]
            
            return {
                "pair": pair,
                "timeframe": timeframe,
                "action": action,
                "prediction": prediction,
                "confidence": float(confidence),
                "current_price": float(current_price),
                "reason": "Model prediction",
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Signal generation error for {pair} {timeframe}: {str(e)}")
            return {
                "pair": pair,
                "timeframe": timeframe,
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "Error",
                "timestamp": datetime.now().isoformat()
            }

# ===== 數據獲取（Binance + yfinance） =====
class DataFetcher:
    """獲取歷史數據"""
    
    @staticmethod
    def get_sample_data(pair, timeframe, n_bars=300):
        """根據交易對自動選擇數據源"""
        is_crypto_pair = '/' in pair  # 例如 BTC/USDT
        
        if is_crypto_pair:
            return DataFetcher._fetch_binance_data(pair, timeframe, n_bars)
        else:
            return DataFetcher._fetch_yfinance_data(pair, timeframe, n_bars)

    @staticmethod
    def _fetch_binance_data(pair, timeframe, n_bars):
        try:
            import ccxt
            exchange = ccxt.binance()
            
            # 轉換時間框架字串
            tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
            ccxt_tf = tf_map.get(timeframe, "1h")
            
            ohlcv = exchange.fetch_ohlcv(pair, ccxt_tf, limit=n_bars)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.warning(f"⚠️ Binance fetch failed for {pair}: {e}")
            return DataFetcher._generate_dummy_data(n_bars)

    @staticmethod
    def _fetch_yfinance_data(pair, timeframe, n_bars):
        try:
            import yfinance as yf
            
            # 調整 yfinance 參數
            if timeframe == "15m":
                period = "5d"
                interval = "15m"
            elif timeframe == "1h":
                period = "60d"
                interval = "1h"
            else:  # 1d
                period = "730d"
                interval = "1d"
            
            df = yf.download(
                pair, 
                period=period, 
                interval=interval, 
                progress=False, 
                auto_adjust=False,
                multi_level_index=False
            )
            
            if len(df) < SIGNAL_CONFIG["min_samples"]:
                return DataFetcher._generate_dummy_data(n_bars)
            
            # 處理 MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 統一欄位名稱
            df.columns = [str(c).capitalize() for c in df.columns]
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                return DataFetcher._generate_dummy_data(n_bars)
                
            df = df[required_cols].tail(n_bars)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ yfinance fetch failed for {pair}: {e}")
            return DataFetcher._generate_dummy_data(n_bars)
    
    @staticmethod
    def _generate_dummy_data(n_bars=200):
        """生成模擬數據（最後手段）"""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(n_bars)) + 100
        data = {
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars),
        }
        return pd.DataFrame(data)

# ===== 全局管理器 =====
model_manager = ModelManager()
signal_generator = SignalGenerator()
data_fetcher = DataFetcher()

# ===== Flask Server =====
app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'ok'}, 200

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ===== Discord Bot 事件 =====
@bot.event
async def on_ready():
    logger.info(f"✅ Bot connected as {bot.user}")
    
    # 啟動 Flask
    threading.Thread(target=run_flask, daemon=True).start()

    # 載入模型
    if not model_manager.models:
        model_manager.download_all_models()
        for pair in CONFIG['trading_pairs']:
            for timeframe in CONFIG['timeframes']:
                
                # 🟢 邏輯檢查：避免載入錯誤的模型
                is_crypto = '/' in pair
                if is_crypto and timeframe == '1d':
                    continue
                if not is_crypto and timeframe == '4h':
                    continue

                model, scaler = model_manager.load_model(pair, timeframe)
                if model:
                    model_manager.models[f"{pair}_{timeframe}"] = model
                    model_manager.scalers[f"{pair}_{timeframe}"] = scaler
        
        logger.info(f"✅ Loaded {len(model_manager.models)} models")

    # 啟動循環
    if not trading_loop.is_running():
        trading_loop.start()

# ===== 交易循環 =====
@tasks.loop(minutes=15)
async def trading_loop():
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if not channel:
        return

    logger.info("🔄 Checking signals...")
    
    for pair in CONFIG['trading_pairs']:
        for timeframe in CONFIG['timeframes']:
            
            # 🟢 邏輯檢查：區分美股與加密
            is_crypto = '/' in pair
            if is_crypto and timeframe == '1d':
                continue
            if not is_crypto and timeframe == '4h':
                continue

            model_key = f"{pair}_{timeframe}"
            model = model_manager.models.get(model_key)
            scaler = model_manager.scalers.get(model_key)
            
            if not model:
                continue
                
            df = data_fetcher.get_sample_data(pair, timeframe)
            signal = signal_generator.generate_signal(pair, timeframe, model, scaler, df)
            
            if signal["action"] != "HOLD":
                await send_signal(channel, signal)

async def send_signal(channel, signal):
    color = discord.Color.green() if signal['action'] == 'BUY' else discord.Color.red()
    embed = discord.Embed(
        title=f"🚀 {signal['action']} - {signal['pair']}",
        description=f"Timeframe: {signal['timeframe']}",
        color=color,
        timestamp=datetime.now()
    )
    embed.add_field(name="Confidence", value=f"{signal['confidence']:.1%}")
    embed.add_field(name="Price", value=f"${signal['current_price']:.2f}")
    await channel.send(embed=embed)

# ===== 指令區 =====
@bot.command(name="commands")
async def cmd_commands(ctx):
    """顯示指令列表"""
    msg = """
    **Bot Commands**
    `!status` - 查看狀態
    `!signal <pair> <tf>` - 查詢信號
    `!reload` - 重載模型
    `!config` - 查看配置
    """
    await ctx.send(msg)

@bot.command(name="status")
async def cmd_status(ctx):
    await ctx.send(f"✅ Bot is running. Loaded {len(model_manager.models)} models.")

@bot.command(name="reload")
async def cmd_reload(ctx):
    await ctx.send("🔄 Reloading models...")
    model_manager.models.clear()
    model_manager.scalers.clear()
    model_manager.download_all_models()
    # Re-load
    for pair in CONFIG['trading_pairs']:
        for timeframe in CONFIG['timeframes']:
            
            is_crypto = '/' in pair
            if is_crypto and timeframe == '1d':
                continue
            if not is_crypto and timeframe == '4h':
                continue

            model, scaler = model_manager.load_model(pair, timeframe)
            if model:
                model_manager.models[f"{pair}_{timeframe}"] = model
                model_manager.scalers[f"{pair}_{timeframe}"] = scaler
    await ctx.send(f"✅ Reloaded. Total models: {len(model_manager.models)}")

@bot.command(name="signal")
async def cmd_signal(ctx, pair=None, timeframe=None):
    if not pair or not timeframe:
        await ctx.send("Usage: !signal <pair> <timeframe>")
        return
    
    model_key = f"{pair}_{timeframe}"
    model = model_manager.models.get(model_key)
    scaler = model_manager.scalers.get(model_key)
    
    if not model:
        await ctx.send(f"❌ No model for {pair} {timeframe}")
        return

    df = data_fetcher.get_sample_data(pair, timeframe)
    signal = signal_generator.generate_signal(pair, timeframe, model, scaler, df)
    await send_signal(ctx.channel, signal)

# ===== 啟動 =====
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
