# 🤖 AI项目四大模块

这是一个综合性的AI项目，包含四个主要功能模块。目前已完成第一个模块：**网络日志自动化分析工具**。

## 📊 模块1：网络日志自动化分析工具 ✅ 已完成

### 🌟 功能特点

- **多格式日志解析**: 支持Apache、Nginx、系统日志、防火墙日志等格式
- **智能异常检测**: 使用Isolation Forest算法自动识别异常行为
- **交互式仪表板**: 基于Streamlit的实时可视化界面
- **批量处理**: 支持目录扫描和批量日志文件处理
- **实时监控**: 提供流式异常检测功能

### 🚀 快速开始

1. **环境设置**
```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖（已完成）
pip install -r requirements.txt
```

2. **运行测试**
```bash
python test_log_analyzer.py
```

3. **启动Web仪表板**
```bash
streamlit run app.py
```

### 📁 项目结构

```
modules/log_analyzer/
├── __init__.py          # 模块初始化
├── parser.py            # 日志解析器
├── anomaly_detector.py  # 异常检测引擎
└── dashboard.py         # Web仪表板

config/
└── log_analyzer_config.py  # 配置管理

data/logs/
└── sample_apache.log    # 样本数据

app.py                   # 主程序入口
test_log_analyzer.py     # 测试脚本
```

### 🔧 核心组件

#### 1. 日志解析器 (LogParser)
- 自动检测日志格式
- 支持多种时间戳格式
- 智能编码检测
- 批量文件处理

#### 2. 异常检测器 (LogAnomalyDetector)
- 基于机器学习的异常检测
- 多维度特征提取
- 实时异常评分
- 详细异常报告

#### 3. 可视化仪表板 (LogAnalyzerDashboard)
- 实时数据可视化
- 交互式图表展示
- 异常情况告警
- 数据导出功能

### 📈 测试结果

最新测试显示：
- ✅ 成功解析 10 条日志记录
- ✅ 检测到 1 个异常行为
- ✅ 支持 4 种日志格式
- ✅ 提取 12 个分析特征

### 🎯 下一步计划

#### 模块2：AI聊天机器人集成 (进行中)
- [ ] 设计聊天机器人API接口
- [ ] 实现与内部消息系统的连接器
- [ ] 集成LLM API
- [ ] 添加对话上下文管理

#### 模块3：HuggingFace模型本地部署
- [ ] 创建模型下载和管理系统
- [ ] 实现FastAPI服务器用于模型推理
- [ ] 添加模型版本管理
- [ ] 实现负载均衡和缓存机制

#### 模块4：视频检测系统
- [ ] 设计视频处理管道
- [ ] 实现本地检测功能
- [ ] 集成云端检测API
- [ ] 添加实时流处理能力

### 🛠 技术栈

- **后端**: Python 3.12, FastAPI, SQLAlchemy
- **机器学习**: scikit-learn, pandas, numpy
- **可视化**: Streamlit, Plotly, matplotlib
- **数据处理**: pandas, regex, chardet
- **开发工具**: black, flake8, pytest

### 📞 支持与反馈

如有问题或建议，请通过以下方式联系：
- 创建Issue报告问题
- 提交Pull Request贡献代码
- 参与项目讨论

---

*项目由AI团队开发 © 2025*