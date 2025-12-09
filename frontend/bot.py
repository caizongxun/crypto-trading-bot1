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
        self.models = {}  # 存放載入的模型 {(pair, timeframe): (model, scaler)}
        self.model_dir = "./models"
        self.latest_recommendations = [] # 儲存最新的推薦訊號
        self.admin_id = ADMIN_ID

    async def on_ready(self):
        """機器人啟動時執行"""
        logger.info(f'✅ Bot connected as {self.user}')
        
        # 1. 下載並載入模型
        self.download_models()
        self.load_models()
        
        # 2. 啟動背景任務
        self.bg_task = self.loop.create_task(self.trading_loop())

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
                    # 解析檔名: model_BTC_USD_1h.pkl -> pair=BTC_USD, tf=1h
                    parts = filename.replace("model_", "").replace(".pkl", "").rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                        
                    pair_name, timeframe = parts
                    scaler_filename = f"scaler_{pair_name}_{timeframe}.pkl"
                    scaler_path = os.path.join(self.model_dir, scaler_filename)
                    model_path = os.path.join(self.model_dir, filename)
                    
                    if not os.path.exists(scaler_path):
                        logger.warning(f"⚠️ Scaler missing for {filename}")
                        continue

                    # 載入 pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                        
                    # 存入字典 (key 統一用底線格式，例如 BTC_USD_1h)
                    key = f"{pair_name}_{timeframe}"
                    self.models[key] = (model, scaler)
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {filename}: {e}")
        
        logger.info(f"✅ Loaded {loaded_count} models.")

    async def check_signals(self):
        """定期檢查交易訊號 (核心邏輯)"""
        logger.info("🔄 Checking signals...")
        current_recs = []

        for pair in self.trading_pairs:
            for timeframe in self.timeframes:
                try:
                    # -------------------------------------------------
                    # 1. 數據抓取 (Data Fetching)
                    # -------------------------------------------------
                    # 轉換 Ticker 格式: BTC/USDT -> BTC-USD (給 yfinance 用)
                    yf_ticker = pair
                    if '/' in pair:
                        yf_ticker = pair.replace('/', '-')
                    if 'USDT' in yf_ticker and '-' not in yf_ticker:
                         yf_ticker = yf_ticker.replace('USDT', '-USD')
                    if 'USDT' in yf_ticker: # 確保 BTC/USDT -> BTC-USD
                        yf_ticker = yf_ticker.replace('USDT', 'USD')

                    # 下載數據 (抓 5 天以確保 SMA200 足夠)
                    df = yf.download(
                        yf_ticker, 
                        period="5d", 
                        interval=timeframe, 
                        progress=False, 
                        auto_adjust=False,
                        multi_level_index=False
                    )

                    if len(df) < 50:
                        continue

                    # 清理欄位
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [str(c).lower() for c in df.columns]

                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if not all(col in df.columns for col in required_cols):
                        continue

                    current_price = df['close'].iloc[-1]

                    # -------------------------------------------------
                    # 2. 匹配模型 (Model Matching)
                    # -------------------------------------------------
                    # 將 pair 轉成檔名格式: BTC/USDT -> BTC_USD
                    file_pair = pair.replace('/', '_').replace('-', '_').replace('USDT', 'USD')
                    model_key = f"{file_pair}_{timeframe}"
                    
                    if model_key not in self.models:
                        # 嘗試另一種可能: BTC_USDT (如果你的檔名沒轉 USD)
                        file_pair_alt = pair.replace('/', '_').replace('-', '_')
                        model_key_alt = f"{file_pair_alt}_{timeframe}"
                        if model_key_alt in self.models:
                            model_key = model_key_alt
                        else:
                            continue

                    model, scaler = self.models[model_key]

                    # -------------------------------------------------
                    # 3. 特徵工程 (Feature Engineering)
                    # -------------------------------------------------
                    strategy = TradingStrategy()
                    df_features = strategy.calculate_features(df)
                    
                    if len(df_features) == 0:
                        continue

                    # 取最新一筆
                    latest_features = df_features.iloc[[-1]][strategy.get_feature_columns()]

                    # -------------------------------------------------
                    # 4. 預測與風控 (Prediction & Risk)
                    # -------------------------------------------------
                    X_scaled = scaler.transform(latest_features)
                    prediction = model.predict(X_scaled)[0]
                    proba = model.predict_proba(X_scaled)[0]
                    confidence = max(proba)

                    # 計算 ATR 止盈止損
                    atr = df_features['atr'].iloc[-1]
                    sl_price = current_price - (atr * 2.0)
                    tp_price = current_price + (atr * 3.0)
                    
                    if prediction == -1: # SELL
                        sl_price = current_price + (atr * 2.0)
                        tp_price = current_price - (atr * 3.0)

                    # -------------------------------------------------
                    # 5. 發送訊號 (Signal Dispatch)
                    # -------------------------------------------------
                    # 門檻: 信心 > 60% 且不是 HOLD (0)
                    if confidence > 0.6 and prediction != 0:
                        action = "BUY" if prediction == 1 else "SELL"
                        
                        # 存入推薦清單
                        rec_str = (
                            f"**{action} {pair}** ({timeframe})\n"
                            f"💰 `${current_price:.2f}` | 📊 `{confidence:.1%}`\n"
                            f"🛑 `${sl_price:.2f}` | 🎯 `${tp_price:.2f}`"
                        )
                        current_recs.append(rec_str)

                        # 發送 Discord Embed
                        embed = discord.Embed(
                            title=f"🚨 {action} Signal: {pair}",
                            color=0x00ff00 if action == "BUY" else 0xff0000
                        )
                        embed.add_field(name="Timeframe", value=timeframe, inline=True)
                        embed.add_field(name="Confidence", value=f"{confidence:.1%}", inline=True)
                        embed.add_field(name="Price", value=f"${current_price:.2f}", inline=True)
                        embed.add_field(name="Strategy", value=f"🛑 SL: ${sl_price:.2f}\n🎯 TP: ${tp_price:.2f}", inline=False)
                        embed.set_footer(text=f"Model: {model_key}")
                        embed.timestamp = datetime.now()
                        
                        channel = self.get_channel(self.channel_id)
                        if channel:
                            await channel.send(embed=embed)
                            
                except Exception as e:
                    # 捕捉單一錯誤，避免整個迴圈中斷
                    # logger.error(f"❌ Error {pair} {timeframe}: {e}")
                    pass

        self.latest_recommendations = current_recs
        logger.info(f"✅ Check done. {len(current_recs)} signals found.")

    async def trading_loop(self):
        """背景迴圈"""
        await self.wait_until_ready()
        while not self.is_closed():
            await self.check_signals()
            # 每 15 分鐘檢查一次
            await asyncio.sleep(900) 

    async def on_message(self, message):
        """訊息監聽"""
        if message.author == self.user:
            return

        # 1. !recommend 指令
        if message.content == "!recommend":
            if self.latest_recommendations:
                msg = "📊 **Current High-Confidence Setup:**\n\n"
                msg += "\n\n".join(self.latest_recommendations)
                await message.channel.send(msg)
            else:
                await message.channel.send("🤷‍♂️ No high-confidence signals at the moment.")

        # 2. !reload 指令 (強制更新模型)
        elif message.content == "!reload":
            # 簡單權限檢查 (如果有設定 ADMIN_ID)
            if self.admin_id and str(message.author.id) != self.admin_id:
                await message.channel.send("⛔ Permission denied.")
                return

            await message.channel.send("🔄 Force reloading models from Hugging Face...")
            try:
                # 刪除舊模型，強制重新下載
                if os.path.exists(self.model_dir):
                    shutil.rmtree(self.model_dir)
                
                self.download_models()
                self.load_models()
                await message.channel.send(f"✅ Reload success! {len(self.models)} models loaded.")
            except Exception as e:
                await message.channel.send(f"❌ Reload failed: {e}")

        # 3. !ping 指令
        elif message.content == "!ping":
            await message.channel.send("🏓 Pong! System online.")

# ==============================================================================
# 程式進入點
# ==============================================================================
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("❌ DISCORD_TOKEN not found in .env")
        exit(1)
        
    client = CryptoBot()
    client.run(DISCORD_TOKEN)