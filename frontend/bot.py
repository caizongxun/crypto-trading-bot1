#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Trading Bot - Discord Bot 版本（完整版）
支持多交易對、多時間框架、使用训练的 ML 模型生成真实信号
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

bot = commands.Bot(command_prefix="!", intents=intents)

# ===== 配置 - 根據你的 trainer_av.py 更新 =====
CONFIG = {
    "trading_pairs": [
        # 原本的 10 個加密貨幣（完全保留）
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT",
        # 額外加上的美股（對應你新訓練的模型）
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "BTC-USD",
    ],
    "timeframes": ["15m", "1h", "4h"],
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
                model_filename = self.get_model_filename(pair, timeframe)
                scaler_filename = self.get_scaler_filename(pair, timeframe)
                
                model_path = os.path.join(self.model_dir, model_filename)
                scaler_path = os.path.join(self.model_dir, scaler_filename)

                try:
                    # 下載模型
                    if not os.path.exists(model_path):
                        logger.info(f"📥 下載模型：{pair} {timeframe}...")
                        hf_hub_download(
                            repo_id=CONFIG["hf_repo_id"],
                            filename=model_filename,
                            local_dir=self.model_dir,
                            token=CONFIG["hf_token"]
                        )
                    
                    # 下載 scaler
                    if not os.path.exists(scaler_path):
                        logger.info(f"📥 下載 scaler：{pair} {timeframe}...")
                        hf_hub_download(
                            repo_id=CONFIG["hf_repo_id"],
                            filename=scaler_filename,
                            local_dir=self.model_dir,
                            token=CONFIG["hf_token"]
                        )
                    
                    logger.info(f"✅ 下載完成：{pair} {timeframe}")
                    downloaded += 1

                except Exception as e:
                    logger.warning(f"⚠️ 下載失敗 {pair} {timeframe}: {str(e)[:100]}")

        logger.info(f"📊 下載完成：{downloaded}/{total}")
        return downloaded > 0

    def load_model(self, pair, timeframe):
        """載入模型和 scaler"""
        model_filename = self.get_model_filename(pair, timeframe)
        scaler_filename = self.get_scaler_filename(pair, timeframe)
        
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)

        if not os.path.exists(model_path):
            logger.warning(f"⚠️ 模型不存在：{model_path}")
            return None, None

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            
            logger.info(f"✅ 模型已載入：{pair} {timeframe}")
            return model, scaler

        except Exception as e:
            logger.error(f"❌ 載入模型失敗 {pair} {timeframe}: {str(e)}")
            return None, None

# ===== 交易信號邏輯 =====
class SignalGenerator:
    def __init__(self):
        self.strategy = TradingStrategy()
    
    def generate_signal(self, pair, timeframe, model, scaler, historical_data_df):
        """
        生成交易信號
        
        Args:
            pair: 交易對 (如 "BTC-USD")
            timeframe: 時間框架 (如 "1h")
            model: 訓練的 RandomForest 模型
            scaler: 標準化器 (StandardScaler)
            historical_data_df: 歷史數據 DataFrame (包含 open, high, low, close, volume)
        
        Returns:
            signal dict with action, confidence, etc.
        """
        
        if model is None:
            logger.warning(f"⚠️ No model available for {pair} {timeframe}")
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
                    "reason": f"Insufficient data ({len(historical_data_df)} < {SIGNAL_CONFIG['min_samples']})",
                    "timestamp": datetime.now().isoformat()
                }
            
            features_df = self.strategy.calculate_features(historical_data_df)
            
            if len(features_df) == 0:
                return {
                    "pair": pair,
                    "timeframe": timeframe,
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reason": "Feature calculation failed",
                    "timestamp": datetime.now().isoformat()
                }
            
            # 取最後一行數據
            feature_columns = self.strategy.get_feature_columns()
            latest_features = features_df[feature_columns].iloc[-1].values.reshape(1, -1)
            
            # 標準化特徵
            if scaler is not None:
                latest_features = scaler.transform(latest_features)
            
            # 模型預測
            prediction = model.predict(latest_features)[0]  # 0, 1, -1
            
            # 獲取預測概率（如果可用）
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
                "reason": f"Error: {str(e)[:50]}",
                "timestamp": datetime.now().isoformat()
            }

# ===== 數據獲取（模擬/真實） =====
class DataFetcher:
    """獲取歷史數據用於特徵計算"""
    
    @staticmethod
    def get_sample_data(pair, timeframe, n_bars=200):
        """
        獲取樣本數據進行特徵計算
        
        在實際應用中，這裡應該連接到真實的數據源
        （如 Alpha Vantage, yfinance, Binance API 等）
        
        目前返回模擬數據以供測試
        """
        try:
            # 嘗試使用 yfinance 獲取真實數據
            import yfinance as yf
            
            # 根據不同交易對選擇時間範圍
            if timeframe == "15m":
                period = "5d"
                interval = "15m"
            elif timeframe == "1h":
                period = "60d"
                interval = "1h"
            else:  # 4h
                period = "730d"
                interval = "1d"
            
            # yfinance 可能不支持所有交易對，特別是美股需要特定格式
            ticker = pair
            if pair == "BTC-USD":
                ticker = "BTC-USD"
            
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if len(df) < SIGNAL_CONFIG["min_samples"]:
                logger.warning(f"⚠️ yfinance 數據不足 {pair}")
                return DataFetcher._generate_dummy_data(n_bars)
            
            # 確保列名正確
            df.columns = ['open', 'high', 'low', 'close', 'volume', 'Adj Close'] if len(df.columns) > 5 else ['open', 'high', 'low', 'close', 'volume']
            df = df[['open', 'high', 'low', 'close', 'volume']].tail(n_bars)
            df = df.reset_index(drop=True)
            
            return df
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch real data for {pair}: {str(e)[:50]}")
            return DataFetcher._generate_dummy_data(n_bars)
    
    @staticmethod
    def _generate_dummy_data(n_bars=200):
        """生成模擬數據用於測試"""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(n_bars)) + 100
        
        data = {
            'open': prices - np.abs(np.random.randn(n_bars)) * 2,
            'high': prices + np.abs(np.random.randn(n_bars)) * 2,
            'low': prices - np.abs(np.random.randn(n_bars)) * 2,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_bars),
        }
        
        return pd.DataFrame(data)

# ===== 全局管理器 =====
model_manager = ModelManager()
signal_generator = SignalGenerator()
data_fetcher = DataFetcher()

# ===== Flask Server =====
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok', 'bot': 'running'}, 200

@app.route('/status', methods=['GET'])
def status():
    return {
        'status': 'running',
        'bot_name': bot.user.name if bot.user else 'Not Connected',
        'models_loaded': len(model_manager.models),
        'timestamp': datetime.now().isoformat()
    }, 200

def run_flask():
    """在後台運行 Flask"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ===== Discord Bot 事件 =====
@bot.event
async def on_ready():
    """Bot 連接成功"""
    logger.info(f"✅ Bot connected as {bot.user}")
    logger.info(f" Bot ID: {bot.user.id}")
    logger.info("✅ 配置驗證成功")
    logger.info(f" 交易對: {CONFIG['trading_pairs']}")
    logger.info(f" 時間框架: {CONFIG['timeframes']}")
    logger.info(f" 交易對數量: {len(CONFIG['trading_pairs']) * len(CONFIG['timeframes'])}")

    # 啟動 Flask
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("✅ Flask server started in background")

    # 首次載入所有模型
    if not model_manager.models:
        logger.info("📦 First check: downloading all models...")
        model_manager.download_all_models()
        
        for pair in CONFIG['trading_pairs']:
            for timeframe in CONFIG['timeframes']:
                model, scaler = model_manager.load_model(pair, timeframe)
                if model:
                    model_manager.models[f"{pair}_{timeframe}"] = model
                    model_manager.scalers[f"{pair}_{timeframe}"] = scaler
        
        logger.info(f"✅ Model initialization completed - {len(model_manager.models)} models loaded")

    # 啟動交易循環
    if not trading_loop.is_running():
        trading_loop.start()
        logger.info("✅ Trading loop started")

@bot.event
async def on_error(event, *args, **kwargs):
    """錯誤處理"""
    logger.error(f"❌ Error in {event}: {args}, {kwargs}")

# ===== 交易循環 =====
@tasks.loop(minutes=15)
async def trading_loop():
    """定期檢查交易信號（每 15 分鐘）"""
    try:
        channel = bot.get_channel(DISCORD_CHANNEL_ID)
        if not channel:
            logger.error(f"❌ Cannot find channel {DISCORD_CHANNEL_ID}")
            return

        logger.info("🔄 Checking trading signals...")
        signals_found = 0

        for pair in CONFIG['trading_pairs']:
            for timeframe in CONFIG['timeframes']:
                model_key = f"{pair}_{timeframe}"
                model = model_manager.models.get(model_key)
                scaler = model_manager.scalers.get(model_key)
                
                # 獲取歷史數據
                df = data_fetcher.get_sample_data(pair, timeframe, n_bars=200)
                
                # 生成信號
                signal = signal_generator.generate_signal(pair, timeframe, model, scaler, df)
                
                if signal["action"] != "HOLD":
                    await send_signal(channel, signal)
                    signals_found += 1

        logger.info(f"✅ Signal check completed - found {signals_found} signals")

    except Exception as e:
        logger.error(f"❌ Error in trading loop: {str(e)}")

async def send_signal(channel, signal):
    """發送交易信號到 Discord"""
    color = discord.Color.green() if signal['action'] == 'BUY' else (discord.Color.red() if signal['action'] == 'SELL' else discord.Color.gray())
    
    embed = discord.Embed(
        title=f"🚀 {signal['action']} - {signal['pair']}",
        description=f"Timeframe: {signal['timeframe']}",
        color=color,
        timestamp=datetime.now()
    )

    embed.add_field(name="Action", value=signal['action'], inline=True)
    embed.add_field(name="Confidence", value=f"{signal['confidence']:.1%}", inline=True)
    
    if 'current_price' in signal:
        embed.add_field(name="Current Price", value=f"${signal['current_price']:.2f}", inline=True)
    
    embed.add_field(name="Reason", value=signal.get('reason', 'N/A'), inline=False)
    embed.add_field(name="Time", value=signal['timestamp'], inline=False)

    try:
        await channel.send(embed=embed)
        logger.info(f"✅ Signal sent: {signal['pair']} {signal['action']}")
    except Exception as e:
        logger.error(f"❌ Failed to send signal: {str(e)}")

# ===== Discord 指令 =====
@bot.command(name="status")
async def cmd_status(ctx):
    """查看 Bot 狀態"""
    embed = discord.Embed(title="🤖 Bot Status", color=discord.Color.blue())
    embed.add_field(name="Status", value="✅ Running", inline=False)
    embed.add_field(name="Trading Pairs", value=f"{len(CONFIG['trading_pairs'])}: {', '.join(CONFIG['trading_pairs'])}", inline=False)
    embed.add_field(name="Timeframes", value=", ".join(CONFIG['timeframes']), inline=False)
    embed.add_field(name="Models Loaded", value=len(model_manager.models), inline=True)
    embed.add_field(name="Total Models", value=len(CONFIG['trading_pairs']) * len(CONFIG['timeframes']), inline=True)
    await ctx.send(embed=embed)

@bot.command(name="reload")
async def cmd_reload(ctx):
    """重新載入所有模型"""
    await ctx.send("🔄 Reloading models...")
    model_manager.models.clear()
    model_manager.scalers.clear()
    model_manager.download_all_models()
    
    for pair in CONFIG['trading_pairs']:
        for timeframe in CONFIG['timeframes']:
            model, scaler = model_manager.load_model(pair, timeframe)
            if model:
                model_manager.models[f"{pair}_{timeframe}"] = model
                model_manager.scalers[f"{pair}_{timeframe}"] = scaler
    
    await ctx.send(f"✅ Loaded {len(model_manager.models)} models")

@bot.command(name="signal")
async def cmd_signal(ctx, pair=None, timeframe=None):
    """查看特定交易對的交易信號
    
    使用方法: !signal <交易對> <時間框架>
    例如: !signal BTC-USD 1h
    """
    if not pair or not timeframe:
        pairs_str = ", ".join(CONFIG['trading_pairs'])
        tf_str = ", ".join(CONFIG['timeframes'])
        await ctx.send(f"用法: !signal <交易對> <時間框架>\n\n可用交易對: {pairs_str}\n可用時間框架: {tf_str}")
        return
    
    model_key = f"{pair}_{timeframe}"
    
    if model_key not in model_manager.models:
        await ctx.send(f"❌ Model not found: {pair} {timeframe}")
        return
    
    model = model_manager.models[model_key]
    scaler = model_manager.scalers.get(model_key)
    
    # 獲取數據
    df = data_fetcher.get_sample_data(pair, timeframe, n_bars=200)
    
    # 生成信號
    signal = signal_generator.generate_signal(pair, timeframe, model, scaler, df)
    
    await send_signal(ctx.channel, signal)

@bot.command(name="signals")
async def cmd_signals(ctx):
    """查看所有交易對的信號"""
    await ctx.send("🔍 Checking all signals...\n")
    
    count = 0
    for pair in CONFIG['trading_pairs']:
        for timeframe in CONFIG['timeframes']:
            model_key = f"{pair}_{timeframe}"
            model = model_manager.models.get(model_key)
            scaler = model_manager.scalers.get(model_key)
            
            if not model:
                continue
            
            df = data_fetcher.get_sample_data(pair, timeframe, n_bars=200)
            signal = signal_generator.generate_signal(pair, timeframe, model, scaler, df)
            
            if signal["action"] != "HOLD":
                await send_signal(ctx.channel, signal)
                count += 1
    
    if count == 0:
        await ctx.send("❌ No signals found")

@bot.command(name="config")
async def cmd_config(ctx):
    """查看交易信號配置"""
    embed = discord.Embed(title="⚙️ Signal Configuration", color=discord.Color.orange())
    embed.add_field(name="Model Confidence Threshold", value=f"{SIGNAL_CONFIG['model_confidence_threshold']:.2%}", inline=True)
    embed.add_field(name="Signal Type", value=SIGNAL_CONFIG['signal_type'], inline=True)
    embed.add_field(name="Min Samples", value=SIGNAL_CONFIG['min_samples'], inline=True)
    await ctx.send(embed=embed)

@bot.command(name="set_threshold")
async def cmd_set_threshold(ctx, threshold: float):
    """設定模型信心度閾值 (0-1)
    
    例如: !set_threshold 0.55
    """
    if not (0 <= threshold <= 1):
        await ctx.send("❌ Threshold must be between 0 and 1")
        return
    
    SIGNAL_CONFIG['model_confidence_threshold'] = threshold
    await ctx.send(f"✅ Model Confidence Threshold set to {threshold:.2%}")

@bot.command(name="set_signal_type")
async def cmd_set_signal_type(ctx, signal_type: str):
    """設定信號類型
    
    buy = 只發送 BUY 信號
    sell = 只發送 SELL 信號
    both = 發送 BUY 和 SELL 信號
    
    例如: !set_signal_type both
    """
    if signal_type.lower() not in ["buy", "sell", "both"]:
        await ctx.send("❌ Signal type must be 'buy', 'sell', or 'both'")
        return
    
    SIGNAL_CONFIG['signal_type'] = signal_type.lower()
    await ctx.send(f"✅ Signal Type set to '{signal_type.lower()}'")

@bot.command(name="commands")
async def cmd_help(ctx):
    """顯示所有可用指令"""
    embed = discord.Embed(title="📖 Bot Commands", color=discord.Color.purple())
    embed.add_field(name="!status", value="查看 Bot 運行狀態", inline=False)
    embed.add_field(name="!signal <pair> <timeframe>", value="查看特定交易對的信號", inline=False)
    embed.add_field(name="!signals", value="查看所有符合條件的信號", inline=False)
    embed.add_field(name="!config", value="查看交易信號配置", inline=False)
    embed.add_field(name="!set_threshold <0-1>", value="設定模型信心度閾值", inline=False)
    embed.add_field(name="!set_signal_type <buy/sell/both>", value="設定信號類型", inline=False)
    embed.add_field(name="!reload", value="重新載入所有模型", inline=False)
    embed.add_field(name="!help", value="顯示此幫助信息", inline=False)
    await ctx.send(embed=embed)

# ===== 啟動 Bot =====
def main():
    logger.info("🚀 Starting Discord Bot...")
    logger.info(f" Channel ID: {DISCORD_CHANNEL_ID}")
    logger.info(f" Repository: {HF_REPO_ID}")
    logger.info(f" Total signals to monitor: {len(CONFIG['trading_pairs']) * len(CONFIG['timeframes'])}")

    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"❌ Bot startup failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
