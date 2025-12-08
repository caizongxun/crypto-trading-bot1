# frontend/config.py
# 配置管理模塊

import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

class Config:
    """基礎配置"""
    
    # ========== Discord 配置 ==========
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
    
    # ========== Hugging Face 配置 ==========
    HF_REPO_ID = os.getenv("HF_REPO_ID")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # ========== Binance 配置 ==========
    BINANCE_SYMBOL = os.getenv("BINANCE_SYMBOL", "BTC/USDT")
    TIMEFRAME = os.getenv("TIMEFRAME", "1h")
    
    # ========== Flask 配置 ==========
    PORT = int(os.getenv("PORT", 5000))
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    
    # ========== 日誌配置 ==========
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # ========== 內部配置 ==========
    MODEL_DIR = "/tmp/models"
    MODEL_CHECK_INTERVAL_HOURS = 24  # 每 24 小時檢查一次模型更新
    TRADING_LOOP_INTERVAL_MINUTES = 60  # 每 60 分鐘執行一次交易循環
    
    @staticmethod
    def validate():
        """驗證必要的配置"""
        required = [
            "DISCORD_TOKEN",
            "HF_REPO_ID",
            "HF_TOKEN"
        ]
        
        missing = [key for key in required if not getattr(Config, key)]
        
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")
        
        return True


class DevelopmentConfig(Config):
    """開發環境配置"""
    FLASK_ENV = "development"
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """生產環境配置 (Koyeb)"""
    FLASK_ENV = "production"
    LOG_LEVEL = "INFO"


def get_config():
    """根據環境返回配置"""
    env = os.getenv("FLASK_ENV", "production")
    
    if env == "development":
        return DevelopmentConfig
    else:
        return ProductionConfig


if __name__ == "__main__":
    # 驗證配置
    try:
        Config.validate()
        print("✅ 配置驗證成功")
        print(f"HF Repository: {Config.HF_REPO_ID}")
        print(f"Binance Symbol: {Config.BINANCE_SYMBOL}")
    except ValueError as e:
        print(f"❌ 配置驗證失敗: {e}")
