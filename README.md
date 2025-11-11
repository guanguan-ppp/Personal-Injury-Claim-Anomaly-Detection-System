# 人伤理赔费用异常检测系统

基于 GAN 的医疗理赔费用异常检测系统，使用 FastAPI 提供 RESTful API 服务。

## 项目概述

本项目使用生成对抗网络（GAN）来检测医疗理赔费用中的异常情况。系统能够自动识别超出正常范围的理赔金额，为保险审核提供辅助决策。

## 技术栈

- **后端框架**: FastAPI
- **机器学习**: PyTorch
- **数据预处理**: scikit-learn, pandas, numpy
- **容器化**: Docker
- **API 文档**: 自动生成的 Swagger UI

## 项目结构

```
claim_gan_project/
├── Dockerfile              # 容器构建文件
├── .dockerignore          # Docker 忽略文件
├── requirements.txt       # Python 依赖列表
├── app.py                # FastAPI 主应用
├── load_and_predict.py   # 模型加载和预测逻辑
├── model_config.json     # 模型配置文件
├── README.md             # 项目说明文档
└── saved_claim_gan_model/ # 训练好的模型文件
    ├── model_weights.pth  # 模型权重
    └── model_config.json  # 模型配置（副本）
```

## 快速开始

### 前提条件

- Docker 20.0+
- 或者 Python 3.9+
- 至少 2GB 可用内存

### 使用 Docker 运行（推荐）

1. **构建 Docker 镜像**
   ```bash
   docker build -t claim-gan-detector:latest .
   ```

2. **运行容器**
   ```bash
   docker run -d -p 8000:8000 --name gan-detector claim-gan-detector:latest
   ```

3. **验证服务**
   ```bash
   curl http://localhost:8000/health
   ```

### 本地开发运行

1. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行应用**
   ```bash
   python app.py
   ```

## API 文档

服务启动后访问：http://localhost:8000/docs

### 主要端点

#### 1. 健康检查
```http
GET /health
```
响应：
```json
{"status": "healthy", "model_loaded": true}
```

#### 2. 单条预测
```http
POST /predict
Content-Type: application/json

{
  "amount": 50000
}
```

响应示例：
```json
{
  "amount": 50000.0,
  "discriminator_score": 0.1234,
  "threshold": 0.4100,
  "is_anomaly": true,
  "is_reasonable": false,
  "explanation": "鉴别器分数: 0.1234 < 阈值 0.4100",
  "recommendation": "建议进一步审核",
  "boundary_violation": false,
  "z_score": 5.89
}
```

#### 3. 批量预测
```http
POST /predict/batch
Content-Type: application/json

{
  "amounts": [50, 200, 50000, 100000]
}
```

#### 4. 模型信息
```http
GET /model/info
```

## 模型说明

### 检测逻辑

1. **边界检查**: 基于训练数据的统计范围（Z-score > 10 或低于最小值）
2. **GAN 鉴别器评分**: 使用训练好的鉴别器对标准化后的费用进行评分
3. **阈值判断**: 分数低于阈值判定为异常

### 训练数据统计
- 数据范围: -302,267,712.53 ~ 343,074,483.90
- 均值: 234,074.76
- 标准差: 8,476,638.36
- 异常阈值: 0.4100

## 开发指南

### 添加新功能

1. **修改模型逻辑**: 编辑 `load_and_predict.py`
2. **添加 API 端点**: 在 `app.py` 中定义新的路由
3. **更新依赖**: 修改 `requirements.txt`

### 测试新功能

```bash
# 使用 curl 测试
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"amount": 1000}'

# 或使用 Python 脚本
python test_api.py
```

## 部署说明

### 生产环境配置

1. **使用 Docker Compose**:
   ```bash
   docker-compose up -d
   ```

2. **资源限制**:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 2G
         cpus: '1.0'
   ```

3. **健康检查**:
   ```bash
   docker logs gan-detector --tail 50
   docker stats gan-detector
   ```

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_DIR` | `saved_claim_gan_model` | 模型文件目录 |
| `PYTHONPATH` | `/app` | Python 路径 |

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查 `saved_claim_gan_model` 目录是否存在且包含模型文件
   - 验证文件权限

2. **CPU 兼容性问题**
   - 确保使用 PyTorch 2.0.1 版本
   - 设置环境变量 `OMP_NUM_THREADS=1`

3. **内存不足**
   - 增加 Docker 内存分配
   - 使用 `docker system prune` 清理缓存

### 日志查看

```bash
# 查看实时日志
docker logs -f gan-detector

# 查看错误日志
docker logs gan-detector | grep -i error

# 进入容器调试
docker exec -it gan-detector /bin/bash
```

## 模型训练

如果需要重新训练模型：

1. 准备训练数据（Excel 文件包含 `fee_all` 列）
2. 运行训练脚本：
   ```bash
   python train.py
   ```
3. 模型将保存到 `saved_claim_gan_model` 目录

## 性能指标

- **响应时间**: < 100ms（单次预测）
- **并发支持**: 100+ QPS
- **内存占用**: ~1.2GB（包含 PyTorch 运行时）

## 维护说明

### 定期任务

1. **监控模型性能**
2. **更新训练数据**
3. **检查依赖安全更新**
4. **备份模型文件**

### 版本升级

1. 测试新版本兼容性
2. 更新 `requirements.txt`
3. 重新构建 Docker 镜像
4. 部署到测试环境验证

## 联系方式

- 项目维护者: [关芝玉]
- 问题反馈: [GitHub Issues 或guanzhiyu777@163.com]
- 文档更新: 请维护此 README 文件



---

**注意**: 首次部署时请确保 `saved_claim_gan_model` 目录包含训练好的模型文件 (`model_weights.pth` 和 `model_config.json`)。
