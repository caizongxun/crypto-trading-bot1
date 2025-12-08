# frontend/bot.py
# Discord Bot + Flask Keep-alive + 多交易對支援 + 模型熱更新

import os
import json
import pickle
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import traceback

import discord
from discord.ext import commands, tasks
import pandas as pd
import numpy as np
from flask import Flask
from threading import Thread
import ccxt
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

from strategy import TradingStrategy
from config import Config

# 載入環境變數
load_dotenv()

# ========== 日誌設置 ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 配置 ==========
try:
    Config.validate()
    DISCORD_TOKEN = Config.DISCORD_TOKEN
    DISCORD_CHANNEL_ID = int(Config.DISCORD_CHANNEL_ID)
    HF_REPO_ID = Config.HF_REPO_ID
    HF_TOKEN = Config.HF_TOKEN
    
    # 解析多交易對和多時間框架
    TRADING_PAIRS = Config.TRADING_PAIRS if hasattr(Config, 'TRADING_PAIRS') else "BTC/USDT"
    TIMEFRAMES = Config.TIMEFRAMES if hasattr(Config, 'TIMEFRAMES') else "1h"
    
    # 如果是字符串，轉換為列表
    if isinstance(TRADING_PAIRS, str):
        TRADING_PAIRS = [p.strip() for p in TRADING_PAIRS.split(',')]
    if isinstance(TIMEFRAMES, str):
        TIMEFRAMES = [t.strip() for t in TIMEFRAMES.split(',')]
    
    logger.info(f"✅ 配置驗證成功")
    logger.info(f"   交易對: {TRADING_PAIRS}")
    logger.info(f"   時間框架: {TIMEFRAMES}")
    
except ValueError as e:
    logger.error(f"❌ 配置驗證失敗: {e}")
    raise

# 模型存儲路徑
MODEL_DIR = Config.MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== Flask Keep-alive (保持容器醒著) ==========
flask_app = Flask(__name__)

@flask_app.route("/")
def health_check():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}, 200

def run_flask():
    """後台執行 Flask 應用"""
    flask_app.run(host="0.0.0.0", port=Config.PORT, debug=False)

# ========== Binance API 封裝 ==========
class BinanceDataFetcher:
    """從 Binance 抓取實時數據"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """抓取最新 K 線數據"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol} {timeframe}: {e}")
            return None

# ========== 模型管理 ==========
class ModelManager:
    """管理多個交易對的模型下載、更新和推論"""
    
    def __init__(self, hf_repo_id: str, hf_token: str, model_dir: str = "/tmp/models"):
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.model_dir = model_dir
        self.models = {}  # {pair_timeframe: model}
        self.model_versions = {}  # {pair_timeframe: version}
        self.last_update_check = None
    
    def get_model_filename(self, pair: str, timeframe: str) -> str:
        """生成模型文件名"""
        # 將 / 替換為 _ (BTC/USDT -> BTC_USDT)
        pair_clean = pair.replace('/', '_')
        return f"model_{pair_clean}_{timeframe}.pkl"
    
    def download_model(self, pair: str, timeframe: str) -> bool:
        """從 Hugging Face 下載特定交易對的模型"""
        try:
            model_filename = self.get_model_filename(pair, timeframe)
            logger.info(f"Downloading model: {model_filename}")
            
            model_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=model_filename,
                token=self.hf_token,
                cache_dir=self.model_dir
            )
            
            # 載入模型
            with open(model_path, 'rb') as f:
                self.models[f"{pair}_{timeframe}"] = pickle.load(f)
            
            self.model_versions[f"{pair}_{timeframe}"] = datetime.now().isoformat()
            logger.info(f"✅ Model loaded: {pair} {timeframe}")
            return True
        
        except Exception as e:
            logger.warning(f"Model not found for {pair} {timeframe}: {e}")
            return False
    
    def check_for_updates(self) -> bool:
        """檢查是否有新模型 (每 24 小時一次)"""
        now = datetime.now()
        
        if self.last_update_check is None:
            logger.info("First check: downloading all models...")
            success = True
            for pair in TRADING_PAIRS:
                for timeframe in TIMEFRAMES:
                    if not self.download_model(pair, timeframe):
                        success = False
            self.last_update_check = now
            return success
        
        if (now - self.last_update_check) > timedelta(hours=24):
            logger.info("24-hour check: checking for model updates...")
            success = True
            for pair in TRADING_PAIRS:
                for timeframe in TIMEFRAMES:
                    if not self.download_model(pair, timeframe):
                        success = False
            self.last_update_check = now
            return success
        
        return True
    
    def predict(self, pair: str, timeframe: str, features: pd.Series) -> tuple:
        """
        進行推論
        
        Returns:
            (signal, confidence)
        """
        key = f"{pair}_{timeframe}"
        
        if key not in self.models:
            logger.warning(f"Model not loaded for {pair} {timeframe}")
            return 0, 0.0
        
        try:
            model = self.models[key]
            
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_mid', 'bb_lower', 'atr',
                'stoch_k', 'stoch_d',
                'sma_20', 'sma_50', 'sma_200',
                'roc', 'volume_ratio', 'high_low_ratio'
            ]
            
            X = features[feature_columns].values.reshape(1, -1)
            prediction = model.predict(X)[0]
            
            try:
                probabilities = model.predict_proba(X)[0]
                confidence = float(np.max(probabilities))
            except:
                confidence = 0.7
            
            return int(prediction), confidence
        
        except Exception as e:
            logger.error(f"Error during prediction for {pair} {timeframe}: {e}")
            return 0, 0.0

# ========== Discord Bot 設置 ==========
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="/", intents=intents)

# 全局狀態
class BotState:
    def __init__(self):
        self.model_manager = ModelManager(HF_REPO_ID, HF_TOKEN, MODEL_DIR)
        self.data_fetcher = BinanceDataFetcher()
        self.strategy = TradingStrategy()
        
        # 儲存最後的訊號 {pair_timeframe: {signal, price, confidence, time}}
        self.last_signals = {}
        self.is_running = False
        self.trading_params = {
            "threshold": 0.6,
            "enabled": True
        }

bot_state = BotState()

@bot.event
async def on_ready():
    """Bot 啟動事件"""
    logger.info(f"✅ Bot connected as {bot.user}")
    
    # 初始化模型
    bot_state.model_manager.check_for_updates()
    logger.info("✅ Model initialization completed")
    
    # 啟動背景任務
    if not trading_loop.is_running():
        trading_loop.start()
        logger.info("✅ Trading loop started")

@bot.command(name="status")
async def status_command(ctx):
    """查看當前狀態"""
    try:
        status_lines = [
            f"🤖 **Bot Status**",
            f"├─ 狀態: {'🟢 Running' if bot_state.is_running else '🔴 Stopped'}",
            f"├─ 監控交易對: {len(TRADING_PAIRS)} 個",
            f"├─ 監控時間框架: {len(TIMEFRAMES)} 個",
            f"├─ 信心度閾值: {bot_state.trading_params['threshold']}",
            f"└─ 交易已啟用: {'✅' if bot_state.trading_params['enabled'] else '❌'}",
            f"\n**最近訊號:**"
        ]
        
        if bot_state.last_signals:
            for pair_timeframe, signal_info in list(bot_state.last_signals.items())[-5:]:
                status_lines.append(
                    f"├─ {pair_timeframe}: {signal_info['signal']} @ {signal_info['price']:.2f}"
                )
        else:
            status_lines.append("├─ 暫無訊號")
        
        status_text = "\n".join(status_lines)
        
        embed = discord.Embed(
            title="Bot Status",
            description=status_text,
            color=discord.Color.green() if bot_state.is_running else discord.Color.red(),
            timestamp=datetime.now()
        )
        
        await ctx.send(embed=embed)
    
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")

@bot.command(name="set_threshold")
async def set_threshold(ctx, value: float):
    """設置信心度閾值 (0-1)"""
    try:
        if 0 <= value <= 1:
            bot_state.trading_params["threshold"] = value
            await ctx.send(f"✅ 信心度閾值已設置為 {value}")
        else:
            await ctx.send("❌ 值必須在 0-1 之間")
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")

@bot.command(name="toggle_trading")
async def toggle_trading(ctx):
    """啟用/禁用交易"""
    try:
        bot_state.trading_params["enabled"] = not bot_state.trading_params["enabled"]
        status = "✅ 已啟用" if bot_state.trading_params["enabled"] else "❌ 已禁用"
        await ctx.send(f"交易已{status}")
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")

@bot.command(name="check_model")
async def check_model(ctx):
    """手動檢查和更新模型"""
    try:
        await ctx.send("🔄 正在檢查模型更新...")
        
        success_count = 0
        for pair in TRADING_PAIRS:
            for timeframe in TIMEFRAMES:
                if bot_state.model_manager.download_model(pair, timeframe):
                    success_count += 1
        
        total = len(TRADING_PAIRS) * len(TIMEFRAMES)
        await ctx.send(f"✅ 已更新 {success_count}/{total} 個模型")
    
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")

@tasks.loop(minutes=15)  # 每 15 分鐘執行一次 (支援 15m K線)
async def trading_loop():
    """主交易循環 - 監控所有交易對和時間框架"""
    try:
        bot_state.is_running = True
        
        # 每 24 小時檢查一次模型
        bot_state.model_manager.check_for_updates()
        
        channel = bot.get_channel(DISCORD_CHANNEL_ID)
        if not channel:
            logger.error(f"Cannot find channel {DISCORD_CHANNEL_ID}")
            return
        
        # 遍歷所有交易對和時間框架
        for pair in TRADING_PAIRS:
            for timeframe in TIMEFRAMES:
                try:
                    # 1. 抓取數據
                    df = bot_state.data_fetcher.fetch_ohlcv(pair, timeframe, limit=200)
                    
                    if df is None or len(df) == 0:
                        continue
                    
                    # 2. 計算特徵
                    features_df = bot_state.strategy.calculate_features(df)
                    
                    if len(features_df) == 0:
                        continue
                    
                    features = features_df.iloc[-1]
                    
                    # 3. 進行推論
                    signal, confidence = bot_state.model_manager.predict(pair, timeframe, features)
                    
                    # 4. 記錄當前價格
                    current_price = df.iloc[-1]['close']
                    
                    # 5. 根據信心度閾值生成訊號
                    if bot_state.trading_params["enabled"] and confidence >= bot_state.trading_params["threshold"]:
                        
                        pair_timeframe = f"{pair}_{timeframe}"
                        
                        if signal == 1:
                            signal_name = "🟢 BUY"
                            color = discord.Color.green()
                        elif signal == -1:
                            signal_name = "🔴 SELL"
                            color = discord.Color.red()
                        else:
                            signal_name = "⚪ HOLD"
                            color = discord.Color.greyple()
                        
                        if signal != 0:  # 只發送 BUY 或 SELL，不發送 HOLD
                            message = (
                                f"{signal_name} **SIGNAL**\n"
                                f"├─ 交易對: {pair}\n"
                                f"├─ 時間框架: {timeframe}\n"
                                f"├─ 價格: ${current_price:.2f}\n"
                                f"├─ 信心度: {confidence:.2%}\n"
                                f"└─ 時間: {datetime.now().isoformat()}"
                            )
                            
                            # 記錄訊號
                            bot_state.last_signals[pair_timeframe] = {
                                "signal": signal_name,
                                "price": current_price,
                                "confidence": confidence,
                                "time": datetime.now()
                            }
                            
                            embed = discord.Embed(
                                title="Trading Signal",
                                description=message,
                                color=color,
                                timestamp=datetime.now()
                            )
                            
                            await channel.send(embed=embed)
                
                except Exception as e:
                    logger.error(f"Error processing {pair} {timeframe}: {e}")
                    continue
        
        bot_state.is_running = False
    
    except Exception as e:
        logger.error(f"Error in trading loop: {e}")
        traceback.print_exc()
        bot_state.is_running = False

# ========== 啟動 ==========
if __name__ == "__main__":
    # 啟動 Flask (後台線程)
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("✅ Flask server started in background")
    
    # 啟動 Discord Bot
    logger.info("🚀 Starting Discord Bot...")
    bot.run(DISCORD_TOKEN)
