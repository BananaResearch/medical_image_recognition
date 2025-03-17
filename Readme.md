# 项目名称
 
**项目简介**：调用不同模型进行图片识别。
 
## 快速开始和配置
 
- 首先你要先获取调用模型所需的api_key。具体方法可以通过询问 LLM 来获取。
- 将获得的api_key 更新到 setup_demo文件中
- 将图片上传到云平台，获取一个线上的图片地址
- 在prompt_detail模块中更新一下图片地址和本地图片地址，请保持两个地址图片一致
- 修改 image_recognition_demo 模块中最后的main
```bash
if __name__ == "__main__": 
    ####本地图片地址#######
    image_path_local = "/Users/mac/Downloads/xdt-xlbq01.jpeg"
    ###在线图片地址
    image_path = "https://public-ai-demo.oss-cn-beijing.aliyuncs.com/%E5%8C%BB%E7%96%97demo/xdt-xlbq01.jpeg"
    ### 配置 识别和评分 prompt，请保持一致
    prompt_sys_content = prompt_detail.ping_sys_promote
    shibie_prompt = prompt_detail.ping_prompt_check
    #配置循环多少次
    n = 2
```

- 最后执行以下命令

```bash
# 使用pip安装
bash setup_demo.sh