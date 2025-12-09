# backend/strategy.py
# 特徵工程 (Feature Engineering) - 使用 pandas 計算技術指標
# 不依賴 pandas-ta，避免相容性問題
# 與 frontend/strategy.py 完全相同，確保訓練和推論一致

import pandas as pd
import numpy as np

class TradingStrategy:
    """
    統一的特徵工程類，確保訓練和推論的指標計算完全一致
    使用純 pandas 實現所有指標計算
    """
    
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, 
                 macd_signal=9, bb_period=20, bb_std=2):
        """初始化策略參數"""
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_rsi(self, data, period=14):
        """計算相對強弱指數 (RSI)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """計算 MACD 指標"""
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
        """計算平均真實波幅 (ATR)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """計算隨機指標 (Stochastic)"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def calculate_features(self, df):
        """
        計算所有技術指標
        
        Args:
            df: pandas DataFrame 包含 'close', 'high', 'low', 'volume' 列
        
        Returns:
            特徵 DataFrame
        """
        features = df.copy()
        
        # 1. RSI (Relative Strength Index)
        features['rsi'] = self.calculate_rsi(features['close'], self.rsi_period)
        
        # 2. MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = self.calculate_macd(
            features['close'], 
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal
        )
        
        # 3. Bollinger Bands
        features['bb_upper'], features['bb_mid'], features['bb_lower'] = self.calculate_bollinger_bands(
            features['close'],
            period=self.bb_period,
            std_dev=self.bb_std
        )
        
        # 4. ATR (Average True Range)
        features['atr'] = self.calculate_atr(
            features['high'],
            features['low'],
            features['close'],
            period=14
        )
        
        # 5. Stochastic Oscillator
        features['stoch_k'], features['stoch_d'] = self.calculate_stochastic(
            features['high'],
            features['low'],
            features['close'],
            k_period=14,
            d_period=3
        )
        
        # 6. 簡單移動平均線 (SMA)
        features['sma_20'] = features['close'].rolling(window=20).mean()
        features['sma_50'] = features['close'].rolling(window=50).mean()
        features['sma_200'] = features['close'].rolling(window=200).mean()
        
        # 7. 價格變化率 (ROC - Rate of Change)
        features['roc'] = features['close'].pct_change(periods=12)
        
        # 8. 成交量變化
        features['volume_sma'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma']
        
        # 9. High-Low Ratio
        features['high_low_ratio'] = (features['high'] - features['low']) / features['close']
        
        # 移除 NaN (前面指標會產生的 NaN)
        features = features.dropna()
        
        return features
    
    def get_feature_columns(self):
        """返回所有特徵列名稱 (用於模型訓練)"""
        return [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_mid', 'bb_lower', 'atr',
            'stoch_k', 'stoch_d',
            'sma_20', 'sma_50', 'sma_200',
            'roc', 'volume_ratio', 'high_low_ratio'
        ]
    
    def prepare_for_inference(self, df_recent):
        """
        準備最新數據進行推論 (只需要最後一行)
        
        Args:
            df_recent: 最近 200 根 K 線 (DataFrame)
        
        Returns:
            準備好的特徵 (Series)
        """
        features = self.calculate_features(df_recent)
        if len(features) == 0:
            return None
        return features.iloc[-1]


class TargetGenerator:
    """目標標籤生成器 (用於訓練)"""
    
    @staticmethod
    def generate_labels(df, future_periods=24, threshold=0.02):
        """
        基於未來價格生成買/賣標籤
        
        Args:
            df: DataFrame 包含 'close' 列
            future_periods: 預測未來多少根 K 線 (預設 24 個 1小時 K 線 = 1 天)
            threshold: 漲幅閾值 (2% = 0.02)
        
        Returns:
            標籤列 (0 = Hold, 1 = Buy Signal, -1 = Sell Signal)
        """
        labels = []
        
        for i in range(len(df) - future_periods):
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + future_periods]['close']
            
            price_change = (future_price - current_price) / current_price
            
            if price_change > threshold:
                labels.append(1)  # 上漲 => Buy
            elif price_change < -threshold:
                labels.append(-1)  # 下跌 => Sell
            else:
                labels.append(0)  # 橫盤 => Hold
        
        # 填充最後的數據點
        labels.extend([0] * future_periods)
        
        return np.array(labels)
