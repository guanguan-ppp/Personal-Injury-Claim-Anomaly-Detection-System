import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import os


class ImprovedClaimGANAnomalyDetector:
    def __init__(self, latent_dim=10, hidden_dim=128):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.scaler = StandardScaler()

        # 初始化网络结构（必须与训练时相同）
        self.generator = ImprovedGenerator(latent_dim, hidden_dim)
        self.discriminator = ImprovedDiscriminator(hidden_dim)

    def load_model(self, model_dir='saved_claim_gan_model'):
        """从指定目录加载模型"""
        try:
            # 检查文件是否存在
            weights_path = os.path.join(model_dir, 'model_weights.pth')  # 修复：使用正确的文件名
            config_path = os.path.join(model_dir, 'model_config.json')

            print(f"尝试加载权重文件: {weights_path}")
            print(f"尝试加载配置文件: {config_path}")

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"权重文件不存在: {weights_path}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")

            # 加载模型权重
            checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

            # 加载配置信息
            with open(config_path, 'r') as f:
                config = json.load(f)

            # 恢复scaler参数
            scaler_params = config['scaler_params']
            self.scaler.mean_ = np.array(scaler_params['mean'])
            self.scaler.scale_ = np.array(scaler_params['scale'])
            if scaler_params['var'] is not None and len(scaler_params['var']) > 0:
                self.scaler.var_ = np.array(scaler_params['var'])
            if scaler_params['n_samples_seen'] is not None:
                self.scaler.n_samples_seen_ = scaler_params['n_samples_seen']

            # 恢复其他参数
            self.threshold = config['threshold']
            self.train_min = config['train_min']
            self.train_max = config['train_max']
            self.train_mean = config['train_mean']
            self.train_std = config['train_std']

            print("=" * 60)
            print("模型加载成功!")
            print(f"训练数据范围: {self.train_min:.2f} ~ {self.train_max:.2f}")
            print(f"训练数据均值: {self.train_mean:.2f}, 标准差: {self.train_std:.2f}")
            print(f"异常检测阈值: {self.threshold:.4f}")
            print("=" * 60)

        except Exception as e:
            print(f"模型加载失败: {e}")
            # 显示当前目录内容，帮助调试
            if os.path.exists(model_dir):
                print(f"目录 {model_dir} 中的文件:")
                for file in os.listdir(model_dir):
                    print(f"  - {file}")
            else:
                print(f"目录 {model_dir} 不存在")
            raise

    def predict(self, claim_amount):
        """
        使用训练好的模型预测费用是否合理
        """
        # 边界检查：如果费用远超出训练数据范围，直接判断为异常
        z_score = abs(claim_amount - self.train_mean) / self.train_std

        if z_score > 10:  # 如果Z-score大于10，直接判断为异常
            return {
                'amount': claim_amount,
                'discriminator_score': 0.0,
                'threshold': self.threshold,
                'is_anomaly': True,
                'is_reasonable': False,
                'explanation': f"费用严重超出训练数据范围 (Z-score: {z_score:.2f} > 10)",
                'recommendation': '建议人工审核：费用异常偏高',
                'boundary_violation': True
            }

        # 检查是否低于训练数据范围
        if claim_amount < self.train_min:
            return {
                'amount': claim_amount,
                'discriminator_score': 0.0,
                'threshold': self.threshold,
                'is_anomaly': True,
                'is_reasonable': False,
                'explanation': f"费用低于训练数据最小值 {self.train_min:.2f}",
                'recommendation': '建议人工审核：费用异常偏低',
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
                'is_reasonable': not is_anomaly,
                'explanation': f"鉴别器分数: {score:.4f} {'<' if is_anomaly else '>='} 阈值 {self.threshold:.4f}",
                'recommendation': '建议进一步审核' if is_anomaly else '费用合理',
                'boundary_violation': False,
                'z_score': z_score
            }

            return result

        except Exception as e:
            return {
                'amount': claim_amount,
                'error': str(e),
                'is_anomaly': True,
                'is_reasonable': False
            }

    def batch_predict(self, claim_amounts):
        """批量预测多个费用"""
        results = []
        for amount in claim_amounts:
            results.append(self.predict(amount))
        return results


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


def main():
    # 创建检测器实例
    detector = ImprovedClaimGANAnomalyDetector()

    # 加载训练好的模型 - 使用正确的路径
    try:
        detector.load_model('saved_claim_gan_model')  # 确保这个路径与文件1中保存的路径一致
    except Exception as e:
        print(f"无法加载模型: {e}")
        print("请先运行 train_and_save_model.py 来训练和保存模型")
        print("确保模型保存在 'saved_claim_gan_model' 目录中")
        return

    # 测试数据
    test_claims = [
        50,  # 正常范围
        200,  # 正常范围
        400,  # 正常范围
        50000,  # 异常（超出范围）
        100000,  # 异常（超出范围）
        10,  # 边界正常
        5,  # 异常（低于最小值）
        1000,  # 稍高但可能正常
        300,  # 正常范围
        600  # 稍高但可能正常
    ]

    print("\n" + "=" * 80)
    print("人伤理赔费用合理性检测结果")
    print("=" * 80)

    results = detector.batch_predict(test_claims)

    for result in results:
        if 'error' in result:
            print(f"金额: {result['amount']:8.2f} | 错误: {result['error']}")
        else:
            status = "异常" if result['is_anomaly'] else "正常"
            color = "\033[91m" if result['is_anomaly'] else "\033[92m"  # 红色异常，绿色正常
            reset = "\033[0m"

            if result.get('boundary_violation', False):
                print(f"金额: {result['amount']:8.2f} | "
                      f"状态: {color}{status:4}{reset} | "
                      f"{result['explanation']}")
            else:
                print(f"金额: {result['amount']:8.2f} | "
                      f"状态: {color}{status:4}{reset} | "
                      f"分数: {result['discriminator_score']:6.4f} | "
                      f"{result['recommendation']}")

    print("=" * 80)

    # 交互式检测
    print("\n交互式检测模式 (输入 'quit' 退出)")
    while True:
        try:
            user_input = input("\n请输入要检测的理赔费用: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            amount = float(user_input)
            result = detector.predict(amount)

            if 'error' in result:
                print(f"错误: {result['error']}")
            else:
                status = "异常" if result['is_anomaly'] else "正常"
                color = "\033[91m" if result['is_anomaly'] else "\033[92m"
                reset = "\033[0m"

                print(f"检测结果: {color}{status}{reset}")
                print(f"详细说明: {result['explanation']}")
                print(f"处理建议: {result['recommendation']}")

        except ValueError:
            print("请输入有效的数字!")
        except KeyboardInterrupt:
            print("\n程序退出")
            break


if __name__ == "__main__":
    main()
