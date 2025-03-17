#!/bin/bash

# 设置通义千问API Key
export DASHSCOPE_API_KEY='sk-xxxxxx'
# GLM
export ANTHROPIC_API_KEY='xxxxxxxx'
## kimi
export MOONSHOT_API_KEY='sk-xxxxxxx'
## 百度
export BCE_API_KEY='bce-xxxxxxxxx'
## 豆包
export DOUBAO_API_KEY='xxxxxx'
##文心一言
export WENXIN_API_KEY='xxxxx'
export WENXIN_API_SECRET='xxxxxx'
## deepseek
export DEEPSEEK_API_KEYS='sk-xxxxxxxxxxxx'
## 腾讯混元
export TENCENT_API_KEY='sk-xxxxxx'
export TENCENT_SECRET_ID="xxxxxxx"
export TENCENT_SECRET_KEY="xxxxxxxxx"
## 百川
export BAICHUAN_API_KEY='sk-xxxxxx'



# 安装依赖
pip install -r requirements.txt

echo "环境设置完成！"
echo "现在你可以运行 python image_recognition_demo.py 来识别图片了" 
