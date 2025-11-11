from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from load_and_predict import ImprovedClaimGANAnomalyDetector
import uvicorn
import os

# 创建 FastAPI 应用
app = FastAPI(
    title="人伤理赔费用异常检测API",
    description="基于GAN的理赔费用异常检测服务",
    version="1.0.0"
)

# 全局模型实例
detector = None

# 请求数据模型
class ClaimRequest(BaseModel):
    amount: float

class BatchClaimRequest(BaseModel):
    amounts: list[float]

# 响应数据模型
class ClaimResponse(BaseModel):
    amount: float
    discriminator_score: float = None
    threshold: float = None
    is_anomaly: bool
    is_reasonable: bool
    explanation: str = None
    recommendation: str = None
    boundary_violation: bool = None
    z_score: float = None
    error: str = None

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global detector
    try:
        detector = ImprovedClaimGANAnomalyDetector()
        model_dir = os.getenv('MODEL_DIR', 'saved_claim_gan_model')
        detector.load_model(model_dir)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e

@app.get("/")
async def root():
    return {
        "message": "人伤理赔费用异常检测API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    if detector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {"status": "healthy", "model_loaded": detector is not None}

@app.post("/predict", response_model=ClaimResponse)
async def predict_single(request: ClaimRequest):
    """单条预测端点"""
    if detector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        result = detector.predict(request.amount)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchClaimRequest):
    """批量预测端点"""
    if detector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        results = detector.batch_predict(request.amounts)
        return {
            "predictions": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")

@app.get("/model/info")
async def model_info():
    """获取模型信息"""
    if detector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "threshold": detector.threshold,
        "train_min": detector.train_min,
        "train_max": detector.train_max,
        "train_mean": detector.train_mean,
        "train_std": detector.train_std,
        "latent_dim": detector.latent_dim,
        "hidden_dim": detector.hidden_dim
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
