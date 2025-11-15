#!/bin/bash

echo "🚀 开始打包医疗费用预测系统Docker镜像..."

# 检查必要文件
if [ ! -f "Dockerfile.no-mirror-final" ]; then
    echo "错误: 缺少Dockerfile"
    exit 1
fi

# 构建镜像（如果尚未构建）
if ! docker images | grep -q "medical-prediction-api"; then
    echo "构建Docker镜像..."
    docker build -t medical-prediction-api -f Dockerfile.no-mirror-final .
fi

# 创建部署目录
rm -rf docker-deployment-package
mkdir -p docker-deployment-package

echo "打包Docker镜像..."
docker save -o docker-deployment-package/medical-prediction-api.tar medical-prediction-api

echo "复制部署文件..."
cp Dockerfile.no-mirror-final docker-deployment-package/
cp requirements.txt docker-deployment-package/
cp app.py docker-deployment-package/
cp Model_fusion2.py docker-deployment-package/ 2>/dev/null || echo "Model_fusion2.py 不存在，跳过"

echo "创建部署脚本..."
cat > docker-deployment-package/deploy.sh << 'SCRIPT'
#!/bin/bash
echo "部署医疗费用预测系统..."
docker load -i medical-prediction-api.tar
mkdir -p data logs
docker run -d -p 8000:8000 -v \$(pwd)/data:/app/data --name medical-prediction medical-prediction-api
echo "服务已启动: http://localhost:8000/docs"
SCRIPT

chmod +x docker-deployment-package/deploy.sh

echo "创建压缩包..."
tar -czf medical-prediction-docker-package.tar.gz docker-deployment-package/

echo "✅ 打包完成!"
echo "📦 部署包: medical-prediction-docker-package.tar.gz"
echo "📁 内容目录: docker-deployment-package/"
echo ""
echo "在其他机器上部署:"
echo "  tar -xzf medical-prediction-docker-package.tar.gz"
echo "  cd docker-deployment-package"
echo "  ./deploy.sh"
