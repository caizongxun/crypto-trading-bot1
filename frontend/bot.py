#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Trading Bot - Discord Bot 版本
支持多交易對、多時間框架的交易信號發送
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
# 啟用所有 Privileged Intents（修正：之前缺少的 Intent）
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ===== 配置 =====
CONFIG = {
    "trading_pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "timeframes": ["15m", "1h", "4h", "1d"],
    "model_dir": "./models",
    "hf_repo_id": HF_REPO_ID,
    "hf_token": HF_TOKEN,
    "discord_channel_id": DISCORD_CHANNEL_ID,
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
        """下載所有模型（優化版本）"""
        logger.info("📥 開始下載所有模型...")
        
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error("❌ 缺少 huggingface-hub 包，請執行: pip install huggingface-hub")
            return False
        
        total = len(CONFIG["trading_pairs"]) * len(CONFIG["timeframes"])
        downloaded = 0
        
        for pair in CONFIG["trading_pairs"]:
            for timeframe in CONFIG["timeframes"]:
                filename = self.get_model_filename(pair, timeframe)
                filepath = os.path.join(self.model_dir, filename)
                
                try:
                    # 檢查本地是否已存在
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
            import pickle
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✅ 模型已載入：{pair} {timeframe}")
            return model
        except Exception as e:
            logger.error(f"❌ 載入模型失敗 {pair} {timeframe}: {str(e)}")
            return None

# ===== 全局模型管理器 =====
model_manager = ModelManager()

# ===== Flask Server（用於 Koyeb 健康檢查）=====
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
    
    # 驗證配置
    logger.info("✅ 配置驗證成功")
    logger.info(f"   交易對: {CONFIG['trading_pairs']}")
    logger.info(f"   時間框架: {CONFIG['timeframes']}")
    
    # 啟動 Flask
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("✅ Flask server started in background")
    
    # 首次載入所有模型
    if not model_manager.models:
        model_manager.download_all_models()
        for pair in CONFIG['trading_pairs']:
            for timeframe in CONFIG['timeframes']:
                model = model_manager.load_model(pair, timeframe)
                if model:
                    model_manager.models[f"{pair}_{timeframe}"] = model
        logger.info("✅ Model initialization completed")
    
    # 啟動交易循環
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
            logger.info("💡 請確認：")
            logger.info(f"   1. Channel ID 正確：{DISCORD_CHANNEL_ID}")
            logger.info(f"   2. Bot 有進入該伺服器")
            logger.info(f"   3. Bot 有發送訊息的權限")
            return
        
        # 模擬交易信號（你可以替換為實際的交易邏輯）
        logger.info("🔄 檢查交易信號...")
        
        # 示例：生成測試信號
        for pair in CONFIG['trading_pairs']:
            for timeframe in CONFIG['timeframes']:
                model_key = f"{pair}_{timeframe}"
                
                if model_key in model_manager.models:
                    # 這裡放你的交易邏輯
                    signal = {
                        "pair": pair,
                        "timeframe": timeframe,
                        "action": "BUY",  # 或 "SELL"
                        "confidence": 0.75,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # 可選：發送信號到 Discord
                    # await send_signal(channel, signal)
        
        logger.info("✅ Signal check completed")
    
    except Exception as e:
        logger.error(f"❌ Error in trading loop: {str(e)}")

async def send_signal(channel, signal):
    """發送交易信號到 Discord"""
    embed = discord.Embed(
        title=f"🚀 交易信號 - {signal['pair']}",
        description=f"時間框架: {signal['timeframe']}",
        color=discord.Color.green() if signal['action'] == 'BUY' else discord.Color.red(),
        timestamp=datetime.now()
    )
    embed.add_field(name="操作", value=signal['action'], inline=True)
    embed.add_field(name="信心度", value=f"{signal['confidence']:.1%}", inline=True)
    embed.add_field(name="時間", value=signal['timestamp'], inline=False)
    
    try:
        await channel.send(embed=embed)
        logger.info(f"✅ 信號已發送：{signal['pair']} {signal['action']}")
    except Exception as e:
        logger.error(f"❌ 發送信號失敗: {str(e)}")

# ===== Discord 指令 =====

@bot.command(name="status")
async def cmd_status(ctx):
    """查看 Bot 狀態"""
    embed = discord.Embed(title="🤖 Bot 狀態", color=discord.Color.blue())
    embed.add_field(name="狀態", value="✅ 運行中", inline=False)
    embed.add_field(name="交易對", value=", ".join(CONFIG['trading_pairs']), inline=False)
    embed.add_field(name="時間框架", value=", ".join(CONFIG['timeframes']), inline=False)
    embed.add_field(name="已載入模型", value=len(model_manager.models), inline=True)
    embed.add_field(name="總模型數", value=len(CONFIG['trading_pairs']) * len(CONFIG['timeframes']), inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name="reload")
async def cmd_reload(ctx):
    """重新載入所有模型"""
    await ctx.send("🔄 正在重新載入模型...")
    model_manager.models.clear()
    model_manager.download_all_models()
    for pair in CONFIG['trading_pairs']:
        for timeframe in CONFIG['timeframes']:
            model = model_manager.load_model(pair, timeframe)
            if model:
                model_manager.models[f"{pair}_{timeframe}"] = model
    await ctx.send(f"✅ 已載入 {len(model_manager.models)} 個模型")

# ===== 啟動 Bot =====

def main():
    logger.info("🚀 Starting Discord Bot...")
    logger.info(f"   Channel ID: {DISCORD_CHANNEL_ID}")
    logger.info(f"   Repository: {HF_REPO_ID}")
    
    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"❌ Bot 啟動失敗: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
