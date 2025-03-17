import os
import base64
from io import BytesIO
import requests
import json
import types
from zhipuai import ZhipuAI
from openai import OpenAI
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
import prompt_detail


## 选择评测的文件对应的prompt
# prompt_sys_content = shibie_prompt

def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded 
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content
def get_wenxin_access_token():
    api_key = os.getenv('WENXIN_API_KEY')
    api_secret = os.getenv('WENXIN_API_SECRET')
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={api_secret}"
    
    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    return json.loads(response.text)['access_token']

def encode_image_to_base64(image_path):
    """
    将图片转换为base64编码
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def recognize_image_qianwen(image_path):
    """
    使用通义千问在线API识别图片内容
    :param image_path: 图片文件路径
    :return: 识别结果
    """
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-plus",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {"role": "user","content": [
                {"type": "text","text": "请根据图片内容，识别图片内容，并详细解读一下图中内容，并从识别的报告中分析发现的问题给出初步诊断结果和治疗措施，要求结构化输出.过滤掉重复项。"},
                {"type": "image_url",
                "image_url": {"url": image_path}}
                ]},
            {"role": "system","content": prompt_sys_content}
            ]
        )
    print("result识别结果=================>", completion.choices[0].message.content)
    return completion.choices[0].message.content

def recognize_image_wenxin11(image_path):
    """
    使用文心模型识别图片内容
    :param image_path: 图片文件路径
    :return: 识别结果
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 设置API配置
    access_token = get_wenxin_access_token()
    base_url = f"http://aip.baidubce.com/rest/2.0/ocr/v1/medical_report_detection?access_token={access_token}"
    
    # 将图片转换为base64
    base64_image = get_file_content_as_base64(image_path)
    
    # 构建消息
    params = {"image":str(base64_image)}
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    print(params)
    
    try:
        # 调用API
        response = requests.post(base_url, data=params, headers=headers)
        print(response.text)
        
    except Exception as e:
        print(f"API调用详细错误: {str(e)}")
        raise

def recognize_image_glm(image_path):
    """
    使用智谱AI GLM-4V模型识别图片内容
    :param image_path: 图片文件路径
    :return: 识别结果
    """
 
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("未设置 ANTHROPIC_API_KEY 环境变量，请先设置API密钥")
    image_path = encode_image_to_base64(image_path)
    
    try:
        # 创建客户端
        client = ZhipuAI(api_key=api_key)        
        # 构建消息
        messages = [
            {"role": "system","content": prompt_sys_content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请根据图片内容，识别图片内容，并详细解读一下图中内容，并从解读中分析发现的问题给出初步诊断结果和治疗措施，要求结构化输出.过滤掉重复项。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_path
                        }
                    }
                ]
            }
        ]
        
        # 调用API
        response = client.chat.completions.create(
            model="glm-4v",
            messages=messages
        )
        print("result识别结果=================>", response.choices[0].message.content)
        return response.choices[0].message.content
            
    except Exception as e:
        print(f"\nGLM-4V API调用错误: {str(e)}")


def recognize_image_moonshot(image_path):
    """
    使用moonshot-v1-128k-vision-preview模型识别图片
    :param image_path: 图片文件路径
    :return: 模型识别结果
    """
    api_key = os.getenv('MOONSHOT_API_KEY')
    if not api_key:
        raise ValueError("未设置 MOONSHOT_API_KEY 环境变量，请先设置API密钥")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
    try:
        client = OpenAI(
            api_key=os.environ.get("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.cn/v1",
        )
        
        # 构建消息
        messages = [
           {"role": "system","content": prompt_sys_content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请根据图片内容，详细解读一下图中内容，并从解读中分析发现的问题给出初步诊断结果和治疗措施，要求结构化输出。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_to_base64(image_path)}"
                        }
                    }
                ]
            }
        ]
        
        # 调用API
        response = client.chat.completions.create(
            model="moonshot-v1-128k-vision-preview",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9
        )
        print("result识别结果=================>", response.choices[0].message.content)
        return response.choices[0].message.content
            
    except Exception as e:
        print(f"\nMoonshot Vision API调用错误: {str(e)}")


def recognize_image_deepSeek_VL2(image_path):
    """
    使用DeepSeek VL2模型识别图片内容
    :param image_path: 图片文件路径
    :return: 识别结果
    """
    api_key = os.getenv('BCE_API_KEY')

    if not api_key:
        raise ValueError("未设置 BCE_API_KEY 环境变量，请先设置API密钥")

    url = "https://qianfan.baidubce.com/v2/chat/completions"

    payload = json.dumps({
        "model": "deepseek-vl2",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请根据图片内容，识别图片内容，并详细解读一下图中内容，并从解读中分析发现的问题给出初步诊断结果和治疗措施，要求结构化输出.过滤掉重复项。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_path
                        }
                    }
                ]
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)
    print("result识别结果=================>", result)
    return result.get('choices')[0].get('message').get('content')


def recognize_image_doubao(image_path):
    """
    使用豆包模型识别图片内容
    :param image_path: 图片文件路径
    :return: 识别结果
    """
    api_key = os.getenv('DOUBAO_API_KEY')
    if not api_key:
        raise ValueError("未设置 DOUBAO_API_KEY 环境变量，请先设置API密钥")


    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    payload = json.dumps({
    "model": "doubao-1.5-vision-pro-32k-250115",
    "messages": [
        {"role": "system","content": prompt_sys_content},
        {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请根据图片内容，识别图片内容，并详细解读一下图中内容，并从解读中分析发现的问题给出初步诊断结果和治疗措施，要求结构化输出.过滤掉重复项。"
            },
            {
            "type": "image_url",
            "image_url": {
                    "url": image_path
                }
            }
        ]
        }
    ]
    })
    headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)
    print("result识别结果=================>", result)
    # print("result识别结果=================>", result.get('choices')[0].get('message').get('content'))
    return result.get('choices')[0].get('message').get('content')

    
def recognize_image_hunyuan(image_path):
    """
    使用腾讯混元大模型识别图片内容（直接调用 HTTP API）
    :param image_path: 图片文件路径
    :return: 识别结果
    """
    try:
        # 实例化一个认证对象，入参需要传入腾讯云账户 SecretId 和 SecretKey，此处还需注意密钥对的保密
        # 代码泄露可能会导致 SecretId 和 SecretKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议采用更安全的方式来使用密钥，请参见：https://cloud.tencent.com/document/product/1278/85305
        # 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取
        cred = credential.Credential(os.getenv("TENCENT_SECRET_ID"), os.getenv("TENCENT_SECRET_KEY"))
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = "hunyuan.tencentcloudapi.com"

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = hunyuan_client.HunyuanClient(cred, "", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.ChatCompletionsRequest()
        params = {
            "Model": "hunyuan-vision",
            "Messages": [
                {
                    "Role": "system",
                    "Contents": [
                        {
                            "Type": "text",
                            "Text": prompt_sys_content
                        }
                    ]
                },
                {
                    "Role": "user",
                    "Contents": [
                        {
                            "Type": "image",
                            "Text": "请根据图片内容，识别图片内容，并详细解读一下图中内容，并从解读中分析发现的问题给出初步诊断结果和治疗措施，要求结构化输出.过滤掉重复项。",
                            "ImageUrl": {
                                "Url": image_path
                            }
                        }
                    ]
                }
            ]
        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个ChatCompletionsResponse的实例，与请求对象对应
        resp = client.ChatCompletions(req)
        # 输出json格式的字符串回包
        if isinstance(resp, types.GeneratorType):  # 流式响应
            for event in resp:
                print(event)
        else:  # 非流式响应
            print("result识别结果=================>", resp.Choices[0].Message.Content)
            return resp.Choices[0].Message.Content
    except TencentCloudSDKException as err:
        print(err)    

def recognize_image_wenxin(image_path):
    """
    使用模型识别图片内容
    :param image_path: 图片文件路径
    :return: 识别结果
    """
    url = "https://api.bianshi.ai/v3/chat/completions"

    payload = json.dumps({
    "model": "bianshi-medical",
    "messages": [
        {
        "role": "user",
        "content": "舌苔厚腻可能是什么体质？",
        "image": encode_image_to_base64(image_path)
        },
        {"role": "system","content": prompt_sys_content}
    ],
    "temperature": 0.3
    })
    headers = {
    'Authorization': 'Bearer $API_KEY',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

def pingfen(result):
    import re
    ##### 配置对应的评分prompt ############
    ping_promote = shibie_prompt
    if result == "" or result == None or len(result) == 0:
        return 0
    else:
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEYS"), base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": ping_promote},
                {"role": "user", "content": result}
            ],
            stream=False
        )
        print("response 分析=================>", response)

        if re.findall(r'```json(.*?)```', response.choices[0].message.content, re.DOTALL):
            code_blocks = re.findall(r'```json(.*?)```', response.choices[0].message.content, re.DOTALL)
            cleaned_code_blocks = [code.strip() for code in code_blocks]
            json_data = json.loads(cleaned_code_blocks[0])
            score = json_data.get("score")
            print("评分为++++++++++++++>", score)
            return score
        else:
            return 0
    
if __name__ == "__main__": 
    ####本地图片地址#######
    image_path_local = "/Users/mac/Downloads/xdt-xlbq01.jpeg"
    ###在线图片地址
    image_path = "https://public-ai-demo.oss-cn-beijing.aliyuncs.com/%E5%8C%BB%E7%96%97demo/xdt-xlbq01.jpeg"
    ### 配置 识别和评分 prompt，请保持一致
    prompt_sys_content = prompt_detail.ping_sys_promote
    shibie_prompt = prompt_detail.ping_prompt_check

    a,b,c,d,e,f = 0,0,0,0,0,0
    #配置循环多少次
    n = 2
    for i in range(1,n):
        result = recognize_image_qianwen(image_path)
        a += pingfen(result)
        print("recognize_qianwen++++++++++++++>", a/(i))
        result = recognize_image_deepSeek_VL2(image_path)
        b += pingfen(result)
        print("recognize_deepseek_vl2++++++++++++++>", b/(i))
        result = recognize_image_glm(image_path_local)
        c += pingfen(result)
        print("recognize_glm++++++++++++++>", c/(i))
        result = recognize_image_moonshot(image_path_local)
        d += pingfen(result)
        print("recognize_moonshot++++++++++++++>", d/(i))
        result = recognize_image_doubao(image_path)
        e += pingfen(result)
        print("recognize_doubao++++++++++++++>", e/(i))
        result = recognize_image_hunyuan(image_path)
        f += pingfen(result)
        print("recognize_image_hunyuan++++++++++++++>", f/(i))
        print(result)
        
    # except Exception as e:
    #     print(f"\n发生错误: {str(e)}")

      