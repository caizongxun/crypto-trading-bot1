import os
import discord
import asyncio
import logging
import pickle
import warnings
import shutil
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# 引入策略 (特徵計算)
from strategy import TradingStrategy

# 忽略 scikit-learn 版本不一致的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")

# ==============================================================================
# 配置日誌
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoBot")

# ==============================================================================
# 環境變數
# ==============================================================================
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'file.env'))

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID')
HF_REPO_ID = os.getenv('HF_REPO_ID', 'zongowo111/crypto-trading-bot')
TRADING_PAIRS = os.getenv('TRADING_PAIRS', 'BTC/USDT,ETH/USDT,AAPL,TSLA').split(',')
TIMEFRAMES = os.getenv('TIMEFRAMES', '15m,1h,4h,1d').split(',')
ADMIN_ID = os.getenv('ADMIN_ID', '')  # 管理員 Discord ID (可選)

# ==============================================================================
# 機器人核心類別
# ==============================================================================
class CryptoBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        
        self.channel_id = int(CHANNEL_ID) if CHANNEL_ID else None
        self.trading_pairs = [p.strip() for p in TRADING_PAIRS]
        self.timeframes = [t.strip() for t in TIMEFRAMES]
        self.models = {}  # {(pair, timeframe): (model, scaler)}
        self.model_dir = "./models"
        self.latest_recommendations = [] # 快取最新的高信心訊號
        self.admin_id = ADMIN_ID

    async def on_ready(self):
        """機器人啟動時執行"""
        logger.info(f'✅ Bot connected as {self.user}')
        
        # 1. 下載並載入模型
        self.download_models()
        self.load_models()
        
        # 2. 啟動背景任務
        self.bg_task = self.loop.create_task(self.trading_loop())
        
        # 3. 啟動自動重載任務 (每6小時)
        self.reload_task = self.loop.create_task(self.auto_reload_loop())

    def download_models(self):
        """從 Hugging Face 下載模型"""
        logger.info(f"📥 Downloading models from {HF_REPO_ID}...")
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.git*", "*.md"]
            )
            logger.info("✅ Models downloaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to download models: {e}")

    def load_models(self):
        """載入本地模型到記憶體"""
        logger.info("📂 Loading models into memory...")
        self.models = {}
        loaded_count = 0
        
        if not os.path.exists(self.model_dir):
            logger.error("❌ Model directory not found!")
            return

        for filename in os.listdir(self.model_dir):
            if filename.endswith(".pkl") and filename.startswith("model_"):
                try:
                    # 解析檔名: model_BTC_USD_1h.pkl
                    parts = filename.replace("model_", "").replace(".pkl", "").rsplit("_", 1)
                    if len(parts) != 2: continue
                        
                    pair_name, timeframe = parts
                    scaler_filename = f"scaler_{pair_name}_{timeframe}.pkl"
                    
                    model_path = os.path.join(self.model_dir, filename)
                    scaler_path = os.path.join(self.model_dir, scaler_filename)
                    
                    if not os.path.exists(scaler_path):
                        logger.warning(f"⚠️ Scaler missing for {filename}")
                        continue

                    with open(model_path, 'rb') as f: model = pickle.load(f)
                    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
                        
                    key = f"{pair_name}_{timeframe}"
                    self.models[key] = (model, scaler)
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {filename}: {e}")
        
        logger.info(f"✅ Loaded {loaded_count} models.")

    def calculate_signal(self, pair, timeframe):
        """計算單一交易對的訊號詳細數據 (不發送通知)"""
        try:
            # --- 1. 數據抓取 ---
            # 轉換 ticker 格式給 yfinance
            # --- 1. 數據抓取 ---
            # 轉換 ticker 格式給 yfinance
            yf_ticker = pair

            # 如果是加密貨幣對 (含 / )
            if '/' in yf_ticker:
                yf_ticker = yf_ticker.replace('/', '-')

            # 處理 USDT -> USD (避免重複加減號)
            if 'USDT' in yf_ticker:
                if '-' in yf_ticker:
                    yf_ticker = yf_ticker.replace('USDT', 'USD')  # 例如 BTC-USDT -> BTC-USD
                else:
                    yf_ticker = yf_ticker.replace('USDT', '-USD')  # 例如 BTCUSDT -> BTC-USD

            # 防呆：如果不小心變成了 BTC--USD，修回來
            yf_ticker = yf_ticker.replace('--', '-')

            df = yf.download(
                yf_ticker, period="5d", interval=timeframe, 
                progress=False, auto_adjust=False, multi_level_index=False
            )
            
            if len(df) < 50: return None

            # 清理欄位
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in required): return None

            current_price = df['close'].iloc[-1]

            # --- 2. 模型匹配 ---
            # 轉換 pair 格式對應檔名 (BTC_USD_1h)
            file_pair = pair.replace('/', '_').replace('-', '_').replace('USDT', 'USD')
            model_key = f"{file_pair}_{timeframe}"
            
            if model_key not in self.models:
                # 嘗試另一種 (BTC_USDT)
                alt_key = f"{pair.replace('/', '_')}_{timeframe}"
                if alt_key in self.models:
                    model_key = alt_key
                else:
                    return None

            model, scaler = self.models[model_key]

            # --- 3. 特徵計算 ---
            strategy = TradingStrategy()
            df_features = strategy.calculate_features(df)
            if len(df_features) == 0: return None
            
            latest_features = df_features.iloc[[-1]][strategy.get_feature_columns()]

            # --- 4. 預測 ---
            X_scaled = scaler.transform(latest_features)
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            confidence = max(proba)
            
            # --- 5. 取得指標數值 (供顯示用) ---
            rsi = df_features['rsi'].iloc[-1]
            macd = df_features['macd'].iloc[-1]
            atr = df_features['atr'].iloc[-1]

            return {
                'pair': pair,
                'tf': timeframe,
                'action': "BUY" if prediction == 1 else ("SELL" if prediction == -1 else "HOLD"),
                'confidence': confidence,
                'price': current_price,
                'rsi': rsi,
                'macd': macd,
                'atr': atr,
                'model_key': model_key
            }
        except Exception as e:
            # logger.error(f"Calc failed for {pair}: {e}")
            return None

    async def check_signals(self):
        """定期檢查 (只發送高信心訊號)"""
        logger.info("🔄 Checking signals...")
        current_recs = []

        for pair in self.trading_pairs:
            for timeframe in self.timeframes:
                # 稍微讓出 CPU
                await asyncio.sleep(0)
                
                result = self.calculate_signal(pair, timeframe)
                if not result: continue

                # 門檻邏輯: 信心 > 60% 且非 HOLD
                if result['confidence'] > 0.6 and result['action'] != 'HOLD':
                    # 計算 TP/SL
                    sl = result['price'] - (result['atr'] * 2.0)
                    tp = result['price'] + (result['atr'] * 3.0)
                    
                    if result['action'] == 'SELL':
                        sl = result['price'] + (result['atr'] * 2.0)
                        tp = result['price'] - (result['atr'] * 3.0)

                    # 存入列表
                    rec_str = (
                        f"**{result['action']} {pair}** ({timeframe})\n"
                        f"💰 `${result['price']:.2f}` | 📊 `{result['confidence']:.1%}`\n"
                        f"🛑 `${sl:.2f}` | 🎯 `${tp:.2f}`"
                    )
                    current_recs.append(rec_str)

                    # 發送 Discord Embed
                    embed = discord.Embed(
                        title=f"🚨 {result['action']} Signal: {pair}", 
                        color=0x00ff00 if result['action'] == "BUY" else 0xff0000
                    )
                    embed.add_field(name="TF", value=timeframe, inline=True)
                    embed.add_field(name="Conf", value=f"{result['confidence']:.1%}", inline=True)
                    embed.add_field(name="Price", value=f"${result['price']:.2f}", inline=True)
                    embed.add_field(name="Strategy", value=f"🛑 SL: ${sl:.2f}\n🎯 TP: ${tp:.2f}", inline=False)
                    embed.set_footer(text=f"RSI: {result['rsi']:.1f}")
                    embed.timestamp = datetime.now()
                    
                    channel = self.get_channel(self.channel_id)
                    if channel:
                        await channel.send(embed=embed)

        self.latest_recommendations = current_recs
        logger.info(f"✅ Check done. {len(current_recs)} signals found.")

    async def auto_reload_loop(self):
        """每 6 小時自動從 HF 拉取新模型"""
        await self.wait_until_ready()
        while not self.is_closed():
            await asyncio.sleep(21600)  # 6小時
            logger.info("🔄 Auto-reloading models...")
            try:
                if os.path.exists(self.model_dir):
                    shutil.rmtree(self.model_dir)
                self.download_models()
                self.load_models()
            except Exception as e:
                logger.error(f"❌ Auto-reload failed: {e}")

    async def trading_loop(self):
        """背景迴圈"""
        await self.wait_until_ready()
        while not self.is_closed():
            await self.check_signals()
            await asyncio.sleep(900) # 每15分

    async def on_message(self, message):
        """訊息監聽"""
        if message.author == self.user: return

        # 指令 1: !recommend
        if message.content == "!recommend":
            if self.latest_recommendations:
                msg = "📊 **Current High-Confidence Setup:**\n\n" + "\n\n".join(self.latest_recommendations)
                await message.channel.send(msg)
            else:
                await message.channel.send("🤷‍♂️ No high-confidence signals (>60%) at the moment.")

        # 指令 2: !status (強制查看市場概況)
        elif message.content == "!status":
            status_msg = await message.channel.send("🔍 Scanning market... (This may take a moment)")
            results = []
            
            for pair in self.trading_pairs:
                for tf in self.timeframes:
                    await asyncio.sleep(0) 
                    res = self.calculate_signal(pair, tf)
                    if res: results.append(res)
            
            if not results:
                await status_msg.edit(content="❌ No data available.")
                return

            # 排序：信心高 -> 低
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            # 建立表格
            output = "📊 **Market Overview (Top 15)**\n```\n"
            output += f"{'Pair':<10} {'TF':<4} {'Act':<4} {'Conf':<5} {'RSI':<4} {'MACD'}\n"
            output += "-" * 45 + "\n"
            
            for r in results[:15]:
                macd_sign = "+" if r['macd'] > 0 else ""
                output += f"{r['pair']:<10} {r['tf']:<4} {r['action']:<4} {r['confidence']:.0%}  {r['rsi']:.0f}   {macd_sign}{r['macd']:.1f}\n"
            
            output += "```"
            await status_msg.edit(content=output)

        # 指令 3: !reload
        elif message.content == "!reload":
            if self.admin_id and str(message.author.id) != self.admin_id:
                await message.channel.send("⛔ Permission denied.")
                return
            await message.channel.send("🔄 Force reloading models...")
            try:
                if os.path.exists(self.model_dir): shutil.rmtree(self.model_dir)
                self.download_models()
                self.load_models()
                await message.channel.send(f"✅ Reload success! {len(self.models)} models loaded.")
            except Exception as e:
                await message.channel.send(f"❌ Reload failed: {e}")

        # 指令 4: !ping
        elif message.content == "!ping":
            await message.channel.send("🏓 Pong! System online.")

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("❌ DISCORD_TOKEN not found in .env")
        exit(1)
    client = CryptoBot()
    client.run(DISCORD_TOKEN)