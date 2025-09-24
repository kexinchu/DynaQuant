# Coze API 结果解析程序

这是一个用于读取模型输出文件，发送请求到Coze API进行分析，并提取结构化数据的完整程序。

## 功能特性

- **文件读取**: 支持读取JSONL格式的模型输出文件
- **API集成**: 集成Coze API，支持流式响应解析
- **数据提取**: 自动提取dimension_name和score等关键信息
- **结果分析**: 计算总体评分和统计信息
- **批量处理**: 支持批量处理大量记录
- **数据导出**: 导出结构化的分析结果和摘要报告

## 文件结构

```
.
├── coze_api_processor.py    # 主程序
├── run_coze_analysis.py     # 简化运行脚本
├── example_usage.py         # 使用示例
├── coze_config.json         # API配置文件
├── README_coze_analysis.md  # 本说明文件
└── test_results.jsonl       # 模型输出文件（需要先运行测试程序生成）
```

## 环境要求

- Python 3.7+
- requests 库
- 有效的Coze API密钥和工作流ID

## 安装依赖

```bash
pip install requests
```

## 配置说明

### 1. API配置

编辑 `coze_config.json` 文件，填入你的API信息：

```json
{
  "api_key": "your_api_key_here",
  "workflow_id": "your_workflow_id_here",
  "base_url": "https://api.coze.cn/v1/workflow/stream_run",
  "default_delay": 2.0,
  "timeout": 600
}
```

### 2. 工作流配置

确保你的Coze工作流接受以下参数：
- `content`: 用户请求内容
- `answer`: 模型回答内容

## 使用方法

### 方法1: 使用简化脚本（推荐）

```bash
# 基本用法
python run_coze_analysis.py test_results.jsonl

# 指定输出文件
python run_coze_analysis.py test_results.jsonl coze_results.jsonl summary.json
```

### 方法2: 直接使用主程序

```bash
python coze_api_processor.py \
  --input test_results.jsonl \
  --output coze_results.jsonl \
  --summary summary_report.json \
  --api-key your_api_key \
  --workflow-id your_workflow_id \
  --delay 2.0
```

### 方法3: 运行示例程序

```bash
python example_usage.py
```

## 命令行参数

| 参数 | 短参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `--input` | `-i` | 输入文件路径（必需） | - |
| `--output` | `-o` | 输出文件路径 | `coze_results.jsonl` |
| `--summary` | `-s` | 摘要报告文件路径 | `summary_report.json` |
| `--api-key` | - | Coze API密钥（必需） | - |
| `--workflow-id` | - | 工作流ID（必需） | - |
| `--delay` | - | 请求间隔时间（秒） | `2.0` |

## 输入文件格式

程序期望的输入文件格式（JSONL）应该包含以下结构：

```json
{
  "request_id": "unique_id",
  "user_request": {
    "content": "用户的问题或请求",
    "messages": [{"role": "user", "content": "..."}]
  },
  "model_response": {
    "choices": [
      {
        "message": {
          "content": "模型的回答内容"
        }
      }
    ]
  }
}
```

## 输出结果格式

### 1. 主要结果文件 (coze_results.jsonl)

每条记录包含：

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "record_id": "txt_0001",
  "original_record": {
    "content": "原始用户请求",
    "answer": "原始模型回答",
    "request_info": {...},
    "model_response": {...}
  },
  "coze_api_result": {
    "dimensions": [
      {
        "dimension_name": "上下文理解",
        "score": 1,
        "reasoning_content": "推理过程说明"
      }
    ],
    "overall_score": {
      "total_dimensions": 6,
      "total_score": 6,
      "average_score": 1.0,
      "max_score": 1,
      "min_score": 1,
      "score_distribution": {"1": 6}
    },
    "raw_api_response": {...}
  }
}
```

### 2. 摘要报告文件 (summary_report.json)

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "total_records": 15,
  "summary": [
    {
      "record_id": "txt_0001",
      "content_preview": "请介绍一下人工智能的发展历史...",
      "answer_preview": "人工智能的发展历史可以追溯到...",
      "dimensions_count": 6,
      "average_score": 1.0,
      "total_score": 6
    }
  ]
}
```

## 支持的维度

根据你的API示例，程序会提取以下维度的评分：

1. **上下文理解** - 评估回答是否与上下文连贯
2. **工具调用** - 检查是否正确使用工具
3. **幻觉** - 评估是否存在虚构信息
4. **重复答案** - 检查是否重复之前的回答
5. **明知故问** - 评估是否提出不必要的问题
6. **回答矛盾** - 检查回答是否存在逻辑矛盾

## 使用流程

### 完整工作流程

1. **准备阶段**
   ```bash
   # 确保有模型输出文件
   ls test_results.jsonl
   ```

2. **配置API**
   ```bash
   # 编辑配置文件
   vim coze_config.json
   ```

3. **运行分析**
   ```bash
   # 使用简化脚本
   python run_coze_analysis.py test_results.jsonl
   ```

4. **查看结果**
   ```bash
   # 查看主要结果
   head -5 coze_results.jsonl
   
   # 查看摘要报告
   cat summary_report.json
   ```

### 示例工作流程

```bash
# 1. 运行模型测试（如果还没有结果文件）
python test_qwen_service.py -i test_data.txt

# 2. 运行Coze API分析
python run_coze_analysis.py test_results.jsonl

# 3. 查看分析结果
ls -la coze_results.jsonl summary_report.json
```

## 错误处理

程序包含完善的错误处理机制：

- **API错误**: 自动记录失败的请求
- **数据解析**: 跳过格式错误的记录
- **网络问题**: 超时处理和重试机制
- **日志记录**: 详细的错误日志

## 性能优化

1. **请求间隔**: 调整 `--delay` 参数避免API限流
2. **批量处理**: 程序自动批量处理，无需手动分批
3. **内存管理**: 流式处理，支持大文件
4. **错误恢复**: 单条记录失败不影响整体处理

## 故障排除

### 常见问题

1. **API认证失败**
   - 检查API密钥是否正确
   - 确认API密钥是否有效

2. **工作流ID错误**
   - 验证工作流ID是否存在
   - 检查工作流是否已发布

3. **请求超时**
   - 增加超时时间
   - 检查网络连接

4. **数据格式错误**
   - 检查输入文件格式
   - 验证JSON结构

### 调试技巧

1. **启用详细日志**
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **测试单个请求**
   ```bash
   python example_usage.py
   ```

3. **检查API响应**
   ```bash
   curl -X POST 'https://api.coze.cn/v1/workflow/stream_run' \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"workflow_id": "YOUR_WORKFLOW_ID", "parameters": {...}}'
   ```

## 扩展功能

程序设计为模块化结构，可以轻松扩展：

- **新API集成**: 添加其他评估API
- **自定义评分**: 实现自定义评分算法
- **结果可视化**: 添加图表和统计图表
- **数据库存储**: 集成数据库存储结果

## 许可证

本程序遵循项目整体许可证。

## 贡献

欢迎提交问题报告和改进建议！
