# backend/upload_models.py
# 上傳本地訓練好的模型到 Hugging Face

import os
import glob
from huggingface_hub import HfApi
from dotenv import load_dotenv

print("🚀 開始上傳模型到 Hugging Face")

# 讀取環境變數
load_dotenv()

HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_REPO_ID or not HF_TOKEN:
    print("❌ 錯誤: 未找到 HF_REPO_ID 或 HF_TOKEN")
    print("請檢查 .env 文件")
    exit(1)

print(f"📁 倉庫: {HF_REPO_ID}")
print(f"🔑 Token: {HF_TOKEN[:20]}...")

# 初始化 API
api = HfApi(token=HF_TOKEN)

# 找到所有模型文件
model_dir = "./models"
if not os.path.exists(model_dir):
    print(f"❌ 錯誤: {model_dir} 文件夾不存在")
    exit(1)

# 獲取所有 .pkl 文件
pkl_files = glob.glob(f"{model_dir}/*.pkl")
json_files = glob.glob(f"{model_dir}/*.json")

all_files = pkl_files + json_files

if not all_files:
    print(f"❌ 錯誤: 在 {model_dir} 中找不到任何文件")
    exit(1)

print(f"\n📦 找到 {len(all_files)} 個文件")
print("=" * 70)

# 上傳每個文件
success_count = 0
failed_files = []

for file_path in all_files:
    filename = os.path.basename(file_path)
    
    print(f"\n📤 上傳: {filename}")
    
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        print(f"✅ 成功: {filename}")
        success_count += 1
    
    except Exception as e:
        print(f"❌ 失敗: {filename}")
        print(f"   錯誤: {str(e)[:100]}")
        failed_files.append(filename)

# 總結
print(f"\n{'='*70}")
print(f"✅ 上傳完成！")
print(f"{'='*70}")
print(f"成功: {success_count}/{len(all_files)}")

if failed_files:
    print(f"\n❌ 失敗的文件:")
    for f in failed_files:
        print(f"   - {f}")
else:
    print(f"\n🎉 所有文件上傳成功！")
    print(f"📍 查看: https://huggingface.co/{HF_REPO_ID}")
