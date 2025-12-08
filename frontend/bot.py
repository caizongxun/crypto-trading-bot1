#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Trading Bot - Discord Bot 版本（完整版）
支持多交易對、多時間框架、可調整交易邏輯
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
from pathlib import Path

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
    logger.error("   需要: DISCORD_TOKEN, DISCORD_CHANNEL_ID, HF_REPO_ID, HF_TOKEN")
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

# ===== 配置 =====
CONFIG = {
    "trading_pairs": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", 
                      "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT"],
    "timeframes": ["15m", "1h", "4h"],
    "model_dir": "./models",
    "hf_repo_id": HF_REPO_ID,
    "hf_token": HF_TOKEN,
    "discord_channel_id": DISCORD_CHANNEL_ID,
}

# ===== 交易信號參數配置（可調整）=====
SIGNAL_CONFIG = {
    "indicator_threshold": 0.65,  # 指標信號閾值（0-1）
    "model_confidence_threshold": 0.55,  # 模型信心度閾值（0-1）
    "indicator_weight": 0.4,  # 指標權重
    "model_weight": 0.6,  # 模型權重
    "signal_type": "both",  # "both" = 同時滿足, "either" = 滿足任一
}

# ===== 模型管理 =====
class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_dir = CONFIG["model_dir"]
        Path(self.model_dir).mkdir(exist_ok=True)
    
    def get_model_filename(self, pair, timeframe):
        """生成模型文件名"""
        pair_clean = pair.replace("/", "_")
        return f"model_{pair_clean}_{timeframe}.pkl"
    
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
                filename = self.get_model_filename(pair, timeframe)
                filepath = os.path.join(self.model_dir, filename)
                
                try:
                    if os.path.exists(filepath):
                        logger.info(f"✅ 模型已存在：{pair} {timeframe}")
                        downloaded += 1
                        continue
                    
                    logger.info(f"📥 下載模型：{pair} {timeframe}...")
                    hf_hub_download(
                        repo_id=CONFIG["hf_repo_id"],
                        filename=filename,
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
        """載入模型"""
        filename = self.get_model_filename(pair, timeframe)
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ 模型不存在：{filepath}")
            return None
        
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✅ 模型已載入：{pair} {timeframe}")
            return model
        except Exception as e:
            logger.error(f"❌ 載入模型失敗 {pair} {timeframe}: {str(e)}")
            return None

# ===== 交易信號邏輯 =====
class SignalGenerator:
    @staticmethod
    def get_indicator_signal(pair, timeframe):
        """
        獲取技術指標信號
        返回: (0-1 之間的信號值, "BUY" 或 "SELL")
        """
        # 這裡可以根據實際的技術指標計算
        # 示例：使用 RSI, MACD, Bollinger Bands 等
        # 暫時返回模擬值
        import random
        indicator_value = random.random()
        action = "BUY" if indicator_value > 0.5 else "SELL"
        return indicator_value, action
    
    @staticmethod
    def get_model_signal(model, pair, timeframe):
        """
        獲取模型信號
        返回: (0-1 之間的信心度, "BUY" 或 "SELL")
        """
        # 這裡可以用模型預測
        # 示例：model.predict_proba(features)
        # 暫時返回模擬值
        import random
        confidence = random.random()
        action = "BUY" if confidence > 0.5 else "SELL"
        return confidence, action
    
    @staticmethod
    def generate_signal(pair, timeframe, model=None):
        """
        生成綜合交易信號
        使用指標權重 + 模型權重
        """
        indicator_value, indicator_action = SignalGenerator.get_indicator_signal(pair, timeframe)
        
        if model:
            model_confidence, model_action = SignalGenerator.get_model_signal(model, pair, timeframe)
        else:
            model_confidence = 0.5
            model_action = "BUY"
        
        # 計算綜合信心度
        combined_confidence = (
            indicator_value * SIGNAL_CONFIG["indicator_weight"] +
            model_confidence * SIGNAL_CONFIG["model_weight"]
        )
        
        # 判斷是否應該發送信號
        indicator_ok = indicator_value >= SIGNAL_CONFIG["indicator_threshold"]
        model_ok = model_confidence >= SIGNAL_CONFIG["model_confidence_threshold"]
        
        if SIGNAL_CONFIG["signal_type"] == "both":
            should_signal = indicator_ok and model_ok
        else:  # either
            should_signal = indicator_ok or model_ok
        
        action = indicator_action if indicator_action == model_action else "HOLD"
        
        return {
            "pair": pair,
            "timeframe": timeframe,
            "action": action if should_signal else "HOLD",
            "confidence": combined_confidence,
            "indicator_value": indicator_value,
            "model_confidence": model_confidence,
            "should_signal": should_signal,
            "timestamp": datetime.now().isoformat()
        }

# ===== 全局模型管理器 =====
model_manager = ModelManager()
signal_generator = SignalGenerator()

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
        'timestamp': datetime.now().isoformat()
    }, 200

def run_flask():
    """在後台運行 Flask"""
    app.run(host='0.0.0.0', port=5000, debug=False)

# ===== Discord Bot 事件 =====

@bot.event
async def on_ready():
    """Bot 連接成功"""
    logger.info(f"✅ Bot connected as {bot.user}")
    logger.info(f"   Bot ID: {bot.user.id}")
    
    logger.info("✅ 配置驗證成功")
    logger.info(f"   交易對: {CONFIG['trading_pairs']}")
    logger.info(f"   時間框架: {CONFIG['timeframes']}")
    logger.info(f"   交易對數量: {len(CONFIG['trading_pairs']) * len(CONFIG['timeframes'])}")
    
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
                model = model_manager.load_model(pair, timeframe)
                if model:
                    model_manager.models[f"{pair}_{timeframe}"] = model
        logger.info("✅ Model initialization completed")
    
    # 啟動交易循環
    if not trading_loop.is_running():
        trading_loop.start()
    logger.info("✅ Trading loop started")

@bot.event
async def on_error(event, *args, **kwargs):
    """錯誤處理"""
    logger.error(f"❌ Error in {event}: {args}, {kwargs}")

# ===== 交易循環 =====

@tasks.loop(minutes=5)
async def trading_loop():
    """定期檢查交易信號（每 5 分鐘）"""
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
                
                signal = signal_generator.generate_signal(pair, timeframe, model)
                
                if signal["should_signal"] and signal["action"] != "HOLD":
                    await send_signal(channel, signal)
                    signals_found += 1
        
        logger.info(f"✅ Signal check completed - found {signals_found} signals")
    
    except Exception as e:
        logger.error(f"❌ Error in trading loop: {str(e)}")

async def send_signal(channel, signal):
    """發送交易信號到 Discord"""
    color = discord.Color.green() if signal['action'] == 'BUY' else discord.Color.red()
    
    embed = discord.Embed(
        title=f"🚀 Trading Signal - {signal['pair']}",
        description=f"Timeframe: {signal['timeframe']}",
        color=color,
        timestamp=datetime.now()
    )
    embed.add_field(name="Action", value=signal['action'], inline=True)
    embed.add_field(name="Confidence", value=f"{signal['confidence']:.1%}", inline=True)
    embed.add_field(name="Indicator Value", value=f"{signal['indicator_value']:.2%}", inline=True)
    embed.add_field(name="Model Confidence", value=f"{signal['model_confidence']:.2%}", inline=True)
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
    model_manager.download_all_models()
    for pair in CONFIG['trading_pairs']:
        for timeframe in CONFIG['timeframes']:
            model = model_manager.load_model(pair, timeframe)
            if model:
                model_manager.models[f"{pair}_{timeframe}"] = model
    await ctx.send(f"✅ Loaded {len(model_manager.models)} models")

@bot.command(name="signal")
async def cmd_signal(ctx, pair="BTC/USDT", timeframe="1h"):
    """查看特定交易對的交易信號
    使用方法: !signal <交易對> <時間框架>
    例如: !signal BTC/USDT 1h
    """
    model_key = f"{pair}_{timeframe}"
    model = model_manager.models.get(model_key)
    
    if model_key not in model_manager.models and not model:
        available_pairs = ", ".join(CONFIG['trading_pairs'])
        available_tf = ", ".join(CONFIG['timeframes'])
        await ctx.send(f"❌ Model not found: {pair} {timeframe}\n\nAvailable:\nPairs: {available_pairs}\nTimeframes: {available_tf}")
        return
    
    signal = signal_generator.generate_signal(pair, timeframe, model)
    await send_signal(ctx.channel, signal)

@bot.command(name="signals")
async def cmd_signals(ctx):
    """查看所有交易對的信號"""
    await ctx.send("🔍 Checking all signals...\n")
    
    for pair in CONFIG['trading_pairs']:
        for timeframe in CONFIG['timeframes']:
            model_key = f"{pair}_{timeframe}"
            model = model_manager.models.get(model_key)
            
            signal = signal_generator.generate_signal(pair, timeframe, model)
            if signal["should_signal"] and signal["action"] != "HOLD":
                await send_signal(ctx.channel, signal)

@bot.command(name="config")
async def cmd_config(ctx):
    """查看交易信號配置"""
    embed = discord.Embed(title="⚙️ Signal Configuration", color=discord.Color.orange())
    embed.add_field(name="Indicator Weight", value=f"{SIGNAL_CONFIG['indicator_weight']}", inline=True)
    embed.add_field(name="Model Weight", value=f"{SIGNAL_CONFIG['model_weight']}", inline=True)
    embed.add_field(name="Indicator Threshold", value=f"{SIGNAL_CONFIG['indicator_threshold']:.2%}", inline=True)
    embed.add_field(name="Model Confidence Threshold", value=f"{SIGNAL_CONFIG['model_confidence_threshold']:.2%}", inline=True)
    embed.add_field(name="Signal Type", value=SIGNAL_CONFIG['signal_type'], inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name="set_indicator_weight")
async def cmd_set_indicator_weight(ctx, weight: float):
    """設定指標權重 (0-1)
    例如: !set_indicator_weight 0.4
    """
    if not (0 <= weight <= 1):
        await ctx.send("❌ Weight must be between 0 and 1")
        return
    
    SIGNAL_CONFIG['indicator_weight'] = weight
    SIGNAL_CONFIG['model_weight'] = 1 - weight
    await ctx.send(f"✅ Indicator Weight set to {weight}, Model Weight set to {1-weight}")

@bot.command(name="set_indicator_threshold")
async def cmd_set_indicator_threshold(ctx, threshold: float):
    """設定指標閾值 (0-1)
    例如: !set_indicator_threshold 0.65
    """
    if not (0 <= threshold <= 1):
        await ctx.send("❌ Threshold must be between 0 and 1")
        return
    
    SIGNAL_CONFIG['indicator_threshold'] = threshold
    await ctx.send(f"✅ Indicator Threshold set to {threshold:.2%}")

@bot.command(name="set_model_threshold")
async def cmd_set_model_threshold(ctx, threshold: float):
    """設定模型信心度閾值 (0-1)
    例如: !set_model_threshold 0.55
    """
    if not (0 <= threshold <= 1):
        await ctx.send("❌ Threshold must be between 0 and 1")
        return
    
    SIGNAL_CONFIG['model_confidence_threshold'] = threshold
    await ctx.send(f"✅ Model Confidence Threshold set to {threshold:.2%}")

@bot.command(name="set_signal_type")
async def cmd_set_signal_type(ctx, signal_type: str):
    """設定信號類型
    both = 指標和模型同時滿足條件才發送
    either = 指標或模型滿足條件就發送
    例如: !set_signal_type both
    """
    if signal_type.lower() not in ["both", "either"]:
        await ctx.send("❌ Signal type must be 'both' or 'either'")
        return
    
    SIGNAL_CONFIG['signal_type'] = signal_type.lower()
    await ctx.send(f"✅ Signal Type set to '{signal_type.lower()}'")

@bot.command(name="help")
async def cmd_help(ctx):
    """顯示所有可用指令"""
    embed = discord.Embed(title="📖 Bot Commands", color=discord.Color.purple())
    embed.add_field(name="!status", value="查看 Bot 運行狀態", inline=False)
    embed.add_field(name="!signal [pair] [timeframe]", value="查看特定交易對的信號", inline=False)
    embed.add_field(name="!signals", value="查看所有符合條件的信號", inline=False)
    embed.add_field(name="!config", value="查看交易信號配置", inline=False)
    embed.add_field(name="!set_indicator_weight [0-1]", value="設定指標權重", inline=False)
    embed.add_field(name="!set_indicator_threshold [0-1]", value="設定指標閾值", inline=False)
    embed.add_field(name="!set_model_threshold [0-1]", value="設定模型信心度閾值", inline=False)
    embed.add_field(name="!set_signal_type [both/either]", value="設定信號類型", inline=False)
    embed.add_field(name="!reload", value="重新載入所有模型", inline=False)
    embed.add_field(name="!help", value="顯示此幫助信息", inline=False)
    
    await ctx.send(embed=embed)

# ===== 啟動 Bot =====

def main():
    logger.info("🚀 Starting Discord Bot...")
    logger.info(f"   Channel ID: {DISCORD_CHANNEL_ID}")
    logger.info(f"   Repository: {HF_REPO_ID}")
    logger.info(f"   Total signals to monitor: {len(CONFIG['trading_pairs']) * len(CONFIG['timeframes'])}")
    
    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"❌ Bot startup failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
