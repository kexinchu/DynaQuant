# Qwen3-235B-A22B 模型服务测试程序

这是一个用于测试 Qwen3-235B-A22B 模型服务的完整程序，可以读取测试数据文件，发送请求给模型，并将结果记录到 JSONL 文件中。

## 功能特性

- 支持 TXT 和 JSONL 格式的测试数据文件
- 自动处理不同格式的输入数据
- 记录完整的请求和响应信息
- 支持自定义生成参数（温度、top-p、最大token数等）
- 详细的日志记录和错误处理
- 生成测试结果摘要统计
- 支持请求间隔控制，避免服务过载

## 文件结构

```
.
├── test_qwen_service.py      # 主测试程序
├── test_data.txt            # TXT格式测试数据示例
├── test_data.jsonl          # JSONL格式测试数据示例
├── README_test_program.md   # 本说明文件
└── Qwen3-235B-A22B.sh      # 模型服务启动脚本
```

## 环境要求

- Python 3.7+
- requests 库
- 已启动的 Qwen3-235B-A22B 模型服务

## 安装依赖

```bash
pip install requests
```

## 启动模型服务

首先，使用提供的脚本启动模型服务：

```bash
bash Qwen3-235B-A22B.sh
```

服务将在 `http://127.0.0.1:8080` 上启动。

## 使用方法

### 基本用法

```bash
# 使用TXT文件测试
python test_qwen_service.py -i test_data.txt -o results.jsonl

# 使用JSONL文件测试
python test_qwen_service.py -i test_data.jsonl -o results.jsonl

# 指定自定义参数
python test_qwen_service.py -i test_data.txt -o results.jsonl --max-tokens 256 --temperature 0.8
```

### 命令行参数

| 参数 | 短参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `--input` | `-i` | 输入文件路径（必需） | - |
| `--output` | `-o` | 输出文件路径 | `test_results.jsonl` |
| `--host` | - | 服务主机地址 | `127.0.0.1` |
| `--port` | - | 服务端口 | `8080` |
| `--max-tokens` | - | 最大生成token数 | `512` |
| `--temperature` | - | 生成温度 | `0.7` |
| `--top-p` | - | top-p采样参数 | `0.9` |
| `--delay` | - | 请求间隔时间（秒） | `1.0` |

### 测试数据格式

#### TXT 格式
每行一个测试问题，程序会自动转换为标准的消息格式。

#### JSONL 格式
每行一个JSON对象，支持以下字段：
- `id`: 唯一标识符
- `type`: 数据类型
- `content`: 问题内容
- `category`: 问题类别
- `difficulty`: 难度等级

程序会自动从这些字段中提取内容，如果字段不存在，会使用整个对象作为内容。

## 输出结果格式

程序会生成一个JSONL文件，每行包含一个完整的测试结果记录：

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "request_id": "txt_0001",
  "request_type": "txt",
  "user_request": {
    "content": "请介绍一下人工智能的发展历史",
    "messages": [{"role": "user", "content": "请介绍一下人工智能的发展历史"}],
    "parameters": {
      "max_tokens": 512,
      "temperature": 0.7,
      "top_p": 0.9
    }
  },
  "model_response": {
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1704096000,
    "model": "qwen3-235b-a22b",
    "choices": [{"message": {"content": "人工智能的发展历史..."}}],
    "usage": {"prompt_tokens": 15, "completion_tokens": 200},
    "request_time": 2.5
  },
  "processing_info": {
    "request_number": 1,
    "total_requests": 15
  }
}
```

## 示例用法

### 1. 快速测试（使用默认参数）

```bash
python test_qwen_service.py -i test_data.txt
```

### 2. 自定义生成参数

```bash
python test_qwen_service.py \
  -i test_data.jsonl \
  -o custom_results.jsonl \
  --max-tokens 1024 \
  --temperature 0.5 \
  --top-p 0.8
```

### 3. 调整请求间隔

```bash
python test_qwen_service.py \
  -i test_data.txt \
  --delay 2.0  # 每次请求间隔2秒
```

### 4. 指定不同的服务地址

```bash
python test_qwen_service.py \
  -i test_data.txt \
  --host 192.168.1.100 \
  --port 9000
```

## 错误处理

程序包含完善的错误处理机制：

- 自动重试失败的请求
- 详细的错误日志记录
- 跳过格式错误的数据行
- 连接超时处理
- 服务不可用检测

## 性能优化建议

1. **请求间隔**: 根据服务性能调整 `--delay` 参数
2. **批量处理**: 对于大量数据，可以分批处理
3. **并发控制**: 避免同时发送过多请求
4. **内存管理**: 大文件处理时注意内存使用

## 故障排除

### 常见问题

1. **连接失败**
   - 检查服务是否已启动
   - 确认主机地址和端口号
   - 检查防火墙设置

2. **请求超时**
   - 增加超时时间
   - 检查网络连接
   - 调整请求间隔

3. **内存不足**
   - 分批处理数据
   - 减少并发请求数
   - 检查系统资源

### 日志分析

程序会输出详细的日志信息，包括：
- 连接状态
- 请求进度
- 错误详情
- 性能统计

## 扩展功能

程序设计为模块化结构，可以轻松扩展：

- 添加新的数据格式支持
- 实现并发请求处理
- 集成其他模型服务
- 添加结果分析功能

## 许可证

本程序遵循项目整体许可证。

## 贡献

欢迎提交问题报告和改进建议！
