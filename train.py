import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
import json
import os

warnings.filterwarnings('ignore')


class ImprovedClaimGANAnomalyDetector:
    def __init__(self, latent_dim=10, hidden_dim=128):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.scaler = StandardScaler()

        # 改进的网络结构
        self.generator = ImprovedGenerator(latent_dim, hidden_dim)
        self.discriminator = ImprovedDiscriminator(hidden_dim)

        # 使用不同的学习率
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()

    def fit(self, claims_data, epochs=3000, batch_size=64):
        # 数据预处理
        if isinstance(claims_data, pd.DataFrame):
            self.original_data = claims_data.values.flatten()
        else:
            self.original_data = np.array(claims_data).flatten()

        print(f"训练数据量: {len(self.original_data)}")
        print(f"费用范围: {self.original_data.min():.2f} ~ {self.original_data.max():.2f}")
        print(f"平均费用: {self.original_data.mean():.2f}")
        print(f"标准差: {self.original_data.std():.2f}")

        # 数据标准化
        self.data = self.scaler.fit_transform(self.original_data.reshape(-1, 1))
        self.data_tensor = torch.FloatTensor(self.data)

        # 训练历史
        self.d_losses = []
        self.g_losses = []

        self.generator.train()
        self.discriminator.train()

        for epoch in range(epochs):
            # 训练鉴别器
            d_losses_batch = []
            for _ in range(1):  # 减少鉴别器训练次数
                # 真实数据
                idx = torch.randint(0, len(self.data), (batch_size,))
                real_data = self.data_tensor[idx]
                real_labels = torch.ones(batch_size, 1) * 0.9  # 标签平滑

                # 生成假数据
                z = torch.randn(batch_size, self.latent_dim)
                fake_data = self.generator(z)
                fake_labels = torch.zeros(batch_size, 1)

                # 鉴别器前向传播
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data.detach())

                # 鉴别器损失
                d_loss_real = self.criterion(real_output, real_labels)
                d_loss_fake = self.criterion(fake_output, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / 2

                # 反向传播和优化
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                d_losses_batch.append(d_loss.item())

            # 训练生成器
            g_losses_batch = []
            for _ in range(1):
                z = torch.randn(batch_size, self.latent_dim)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data)

                # 生成器希望鉴别器将假数据判断为真
                g_loss = self.criterion(fake_output, torch.ones(batch_size, 1))

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                g_losses_batch.append(g_loss.item())

            self.d_losses.append(np.mean(d_losses_batch))
            self.g_losses.append(np.mean(g_losses_batch))

            if epoch % 500 == 0:
                print(f'Epoch [{epoch}/{epochs}], D Loss: {self.d_losses[-1]:.4f}, G Loss: {self.g_losses[-1]:.4f}')

        # 计算更合理的阈值
        self._calculate_improved_threshold()
        print("GAN训练完成!")

    def _calculate_improved_threshold(self):
        """改进的阈值计算方法"""
        self.discriminator.eval()
        with torch.no_grad():
            # 计算训练数据的鉴别器分数
            scores = self.discriminator(self.data_tensor).numpy().flatten()

            # 使用均值和标准差来设置阈值
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # 设置阈值为均值减去2倍标准差
            self.threshold = mean_score - 2 * std_score

            print(f"训练数据分数 - 均值: {mean_score:.4f}, 标准差: {std_score:.4f}")
            print(f"异常检测阈值: {self.threshold:.4f} (均值 - 2*标准差)")

            # 保存训练数据的统计信息用于边界检测
            self.train_min = float(self.original_data.min())
            self.train_max = float(self.original_data.max())
            self.train_mean = float(self.original_data.mean())
            self.train_std = float(self.original_data.std())

    def predict(self, claim_amount):
        """
        改进的预测方法，包含边界检查
        """
        # 边界检查：如果费用远超出训练数据范围，直接判断为异常
        z_score = abs(claim_amount - self.train_mean) / self.train_std

        if z_score > 10:  # 如果Z-score大于10，直接判断为异常
            return {
                'amount': claim_amount,
                'discriminator_score': 0.0,
                'threshold': self.threshold,
                'is_anomaly': True,
                'explanation': f"费用严重超出训练数据范围 (Z-score: {z_score:.2f} > 10)",
                'boundary_violation': True
            }

        # 正常流程
        try:
            standardized_amount = self.scaler.transform([[claim_amount]])
            input_tensor = torch.FloatTensor(standardized_amount)

            self.discriminator.eval()
            with torch.no_grad():
                score = self.discriminator(input_tensor).item()

            is_anomaly = score < self.threshold

            result = {
                'amount': claim_amount,
                'discriminator_score': score,
                'threshold': self.threshold,
                'is_anomaly': is_anomaly,
                'explanation': f"鉴别器分数: {score:.4f} {'<' if is_anomaly else '>='} 阈值 {self.threshold:.4f}",
                'boundary_violation': False,
                'z_score': z_score
            }

            return result

        except Exception as e:
            return {
                'amount': claim_amount,
                'error': str(e),
                'is_anomaly': True
            }

    def save_model(self, model_dir='saved_model'):
        """保存整个模型到指定目录"""
        # 创建目录
        os.makedirs(model_dir, exist_ok=True)

        # 保存模型权重
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, os.path.join(model_dir, 'model_weights.pth'))

        # 保存scaler参数 - 修复JSON序列化问题
        scaler_params = {
            'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else [],
            'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else [],
            'var': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') and self.scaler.var_ is not None else [],
            'n_samples_seen': int(self.scaler.n_samples_seen_) if hasattr(self.scaler, 'n_samples_seen_') else 0
        }

        # 修复：确保所有数值都是Python原生类型，而不是numpy类型
        def convert_to_python_types(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_python_types(value) for key, value in obj.items()}
            else:
                return obj

        # 保存配置信息 - 确保所有值都是Python原生类型
        config = {
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'threshold': float(self.threshold),  # 确保是Python float
            'train_min': float(self.train_min),
            'train_max': float(self.train_max),
            'train_mean': float(self.train_mean),
            'train_std': float(self.train_std),
            'scaler_params': convert_to_python_types(scaler_params)
        }

        with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        print(f"模型已保存到目录: {model_dir}")
        print(f"保存的文件: model_weights.pth, model_config.json")

    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.plot(self.g_losses, label='Generator Loss')
        plt.title('Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.show()


class ImprovedGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(ImprovedGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ImprovedDiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(ImprovedDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 训练并保存模型
if __name__ == "__main__":
    # 使用你的真实数据范围
    np.random.seed(42)

    # 模拟你的数据范围 (10-500)
    n_samples = 2000
    normal_claims = np.random.uniform(10, 500, n_samples)

    #######
    file_path = 'fee_all1.xlsx'  # 请确保文件路径正确

    # 读取Excel文件（不指定工作表名或使用正确的工作表名）
    df = pd.read_excel(file_path)

    # 检查列名，确认fee_all列存在
    print("所有列名:", df.columns.tolist())

    # 提取名为"fee_all"的列并转换为NumPy数组
    if 'fee_all' in df.columns:
        fee_all_array = df['fee_all'].to_numpy()


    # 创建改进的检测器
    detector = ImprovedClaimGANAnomalyDetector(latent_dim=10, hidden_dim=128)

    # 训练模型
    detector.fit(fee_all_array, epochs=3000, batch_size=64)

    # 绘制训练历史
    detector.plot_training_history()

    # 测试费用
    test_amounts = [50, 200, 400, 50000, 100000, 10, 5, 1000]

    print("\n改进模型的检测结果:")
    print("=" * 80)
    for amount in test_amounts:
        result = detector.predict(amount)
        status = "异常" if result['is_anomaly'] else "正常"

        if 'boundary_violation' in result and result['boundary_violation']:
            print(f"金额: {amount:8.2f} | 状态: {status:4} | {result['explanation']}")
        elif 'error' in result:
            print(f"金额: {amount:8.2f} | 错误: {result['error']}")
        else:
            print(f"金额: {amount:8.2f} | 状态: {status:4} | {result['explanation']}")

    # 保存模型
    detector.save_model('saved_claim_gan_model')
    print("\n模型保存完成！现在你可以运行 load_and_predict.py 来使用训练好的模型了。")
