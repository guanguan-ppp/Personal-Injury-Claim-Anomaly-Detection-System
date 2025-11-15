from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import lightgbm as lgb
import warnings
import os
from collections import Counter

warnings.filterwarnings('ignore')

# Windows中文字体配置
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12


class RegionalQuantileModel:
    """
    整合六大经济区域的分位数建模 - 专注95%分位数版本
    """

    # 六大经济区域划分
    ECONOMIC_REGIONS = {
        '华东': ['31', '32', '33', '34', '35', '36', '37'],
        '华北': ['11', '12', '13', '14', '15'],
        '华中': ['41', '42', '43'],
        '华南': ['44', '45', '46'],
        '西南': ['50', '51', '52', '53', '54'],
        '西北': ['61', '62', '63', '64', '65']
    }

    def __init__(self, train_path, test_path):
        print("正在加载数据...")
        try:
            self.train_data = pd.read_excel(train_path)
            self.test_data = pd.read_excel(test_path)
            print(f"训练集加载成功，形状: {self.train_data.shape}")
            print(f"测试集加载成功，形状: {self.test_data.shape}")
        except Exception as e:
            print(f"数据加载失败: {e}")
            return

        # 根据要求删除"定损类型"和"三者交通方式"特征
        self.feature_names = [
            '伤势程度', '手术次数统计', '治疗情况', '责任赔偿系数',
            '骨折类数量', '护理费数量', '软组织损伤类数量', '颅脑损伤类数量',
            '内脏损伤类数量', '神经损伤类数量', '其他损伤类数量', '地域'
        ]
        self.models = {}
        self.scaler = StandardScaler()
        self.regional_stats = {}
        self.best_quantile = 0.95  # 专注95%分位数

    def prepare_data_with_regions(self):
        """
        准备数据并添加经济区域信息
        """
        print("=== 1. 数据准备与区域划分 ===")

        # 基本数据清洗
        train_df = self.train_data.copy()
        train_df = train_df[train_df['fee_all'] > 0]
        print(f"训练集: {len(train_df)}条")

        test_df = self.test_data.copy()
        test_df = test_df[test_df['fee_all'] > 0]
        print(f"测试集: {len(test_df)}条")

        # 检查地区列
        for df in [train_df, test_df]:
            if '地域' not in df.columns:
                print("警告: 数据中缺少'地域'列")
                df['地域'] = '未知地区'

            df['地域'] = df['地域'].astype(str)
            df['province_code'] = df['地域'].str[:2]

        print(f"训练集提取到 {train_df['province_code'].nunique()} 个省份")
        print(f"测试集提取到 {test_df['province_code'].nunique()} 个省份")

        # 添加经济区域分类
        train_df = self.add_economic_regions(train_df)
        test_df = self.add_economic_regions(test_df)

        # 筛选训练集10-90%区间
        train_df, lower_bound, upper_bound = self.filter_10_90_interval(train_df)

        # 创建两种测试集：10-90%区间和全部数据
        test_df_10_90 = test_df[
            (test_df['fee_all'] >= lower_bound) &
            (test_df['fee_all'] <= upper_bound)
            ].copy()

        print(f"测试集10-90%区间: 样本数{len(test_df_10_90)}")
        print(f"测试集全部数据: 样本数{len(test_df)}")

        return train_df, test_df_10_90, test_df, lower_bound, upper_bound

    def add_economic_regions(self, df):
        """添加经济区域分类"""

        def get_economic_region(province_code):
            for region, codes in self.ECONOMIC_REGIONS.items():
                if province_code in codes:
                    return region
            return '其他'

        df['economic_region'] = df['province_code'].map(get_economic_region)
        return df

    def filter_10_90_interval(self, df):
        """筛选10-90%费用区间"""
        lower_bound = df['fee_all'].quantile(0.1)
        upper_bound = df['fee_all'].quantile(0.9)

        interval_data = df[
            (df['fee_all'] >= lower_bound) &
            (df['fee_all'] <= upper_bound)
            ].copy()

        print(f"10-90%区间: 费用范围[{lower_bound:.0f}, {upper_bound:.0f}], 样本数{len(interval_data)}")

        return interval_data, lower_bound, upper_bound

    def analyze_regional_distribution(self, train_df, test_df_10_90, test_df_all):
        """分析地区分布"""
        print("=== 2. 地区分布分析 ===")

        # 训练集经济区域统计
        train_region_stats = train_df.groupby('economic_region').agg({
            'fee_all': ['count', 'mean', 'median', 'std', 'min', 'max']
        }).round(2)

        train_region_stats.columns = ['样本数', '平均费用', '中位数费用', '费用标准差', '最低费用', '最高费用']
        train_region_stats = train_region_stats.sort_values('平均费用', ascending=False)

        self.regional_stats = train_region_stats

        print("训练集经济区域费用统计:")
        for region, stats in train_region_stats.iterrows():
            print(f"  {region}: {int(stats['样本数'])}样本, 平均费用={stats['平均费用']:.2f}")

        # 测试集10-90%区间经济区域统计
        test_10_90_stats = test_df_10_90.groupby('economic_region').agg({
            'fee_all': ['count', 'mean', 'median', 'std', 'min', 'max']
        }).round(2)

        test_10_90_stats.columns = ['样本数', '平均费用', '中位数费用', '费用标准差', '最低费用', '最高费用']
        test_10_90_stats = test_10_90_stats.sort_values('平均费用', ascending=False)

        print("\n测试集10-90%区间经济区域费用统计:")
        for region, stats in test_10_90_stats.iterrows():
            print(f"  {region}: {int(stats['样本数'])}样本, 平均费用={stats['平均费用']:.2f}")

        # 测试集全部数据经济区域统计
        test_all_stats = test_df_all.groupby('economic_region').agg({
            'fee_all': ['count', 'mean', 'median', 'std', 'min', 'max']
        }).round(2)

        test_all_stats.columns = ['样本数', '平均费用', '中位数费用', '费用标准差', '最低费用', '最高费用']
        test_all_stats = test_all_stats.sort_values('平均费用', ascending=False)

        print("\n测试集全部数据经济区域费用统计:")
        for region, stats in test_all_stats.iterrows():
            print(f"  {region}: {int(stats['样本数'])}样本, 平均费用={stats['平均费用']:.2f}")

        self.visualize_regional_distribution(train_region_stats, test_10_90_stats, test_all_stats)

        return train_region_stats, test_10_90_stats, test_all_stats

    def visualize_regional_distribution(self, train_stats, test_10_90_stats, test_all_stats):
        """可视化地区分布"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        regions = train_stats.index

        # 1. 训练集样本量分布
        ax1.bar(regions, train_stats['样本数'].values, color='skyblue', alpha=0.7)
        ax1.set_title('训练集各经济区域样本量')
        ax1.set_ylabel('样本数量')
        ax1.tick_params(axis='x', rotation=45)

        # 2. 测试集样本量分布
        ax2.bar(regions, test_all_stats['样本数'].values, color='lightcoral', alpha=0.7)
        ax2.set_title('测试集各经济区域样本量')
        ax2.set_ylabel('样本数量')
        ax2.tick_params(axis='x', rotation=45)

        # 3. 训练集平均费用分布
        ax3.bar(regions, train_stats['平均费用'].values, color='skyblue', alpha=0.7)
        ax3.set_title('训练集各经济区域平均费用')
        ax3.set_ylabel('平均费用')
        ax3.tick_params(axis='x', rotation=45)

        # 4. 测试集平均费用分布
        ax4.bar(regions, test_all_stats['平均费用'].values, color='lightcoral', alpha=0.7)
        ax4.set_title('测试集各经济区域平均费用')
        ax4.set_ylabel('平均费用')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def create_regional_features(self, train_df, test_df):
        """创建地区特征"""
        print("=== 3. 创建地区特征 ===")

        # 基础特征
        available_features = [f for f in self.feature_names if f in train_df.columns and f != '地域']
        print(f"可用基础特征: {available_features}")

        X_train = train_df[available_features].copy()
        y_train = train_df['fee_all']

        X_test = test_df[available_features].copy()
        y_test = test_df['fee_all']

        # 处理分类变量
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        self.label_encoders = {}
        for col in categorical_cols:
            print(f"处理分类变量: {col}")
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            self.label_encoders[col] = le

        # 添加经济区域特征
        if 'economic_region' in train_df.columns:
            # 使用训练集计算区域统计特征
            region_means = train_df.groupby('economic_region')['fee_all'].mean()
            region_medians = train_df.groupby('economic_region')['fee_all'].median()
            region_std = train_df.groupby('economic_region')['fee_all'].std()
            region_counts = train_df.groupby('economic_region')['fee_all'].count()

            # 为训练集添加区域特征
            X_train['region_mean_cost'] = train_df['economic_region'].map(region_means)
            X_train['region_median_cost'] = train_df['economic_region'].map(region_medians)
            X_train['region_std_cost'] = train_df['economic_region'].map(region_std)
            X_train['region_sample_count'] = train_df['economic_region'].map(region_counts)

            # 为测试集添加区域特征
            X_test['region_mean_cost'] = test_df['economic_region'].map(region_means)
            X_test['region_median_cost'] = test_df['economic_region'].map(region_medians)
            X_test['region_std_cost'] = test_df['economic_region'].map(region_std)
            X_test['region_sample_count'] = test_df['economic_region'].map(region_counts)

            # 处理缺失值
            national_mean = y_train.mean()
            national_median = y_train.median()
            national_std = y_train.std()

            for df in [X_train, X_test]:
                df['region_mean_cost'].fillna(national_mean, inplace=True)
                df['region_median_cost'].fillna(national_median, inplace=True)
                df['region_std_cost'].fillna(national_std, inplace=True)
                df['region_sample_count'].fillna(0, inplace=True)

            # 相对特征
            X_train['cost_vs_region_mean'] = y_train / X_train['region_mean_cost']
            X_train['cost_vs_national_mean'] = y_train / national_mean

            X_test['cost_vs_region_mean'] = y_test / X_test['region_mean_cost']
            X_test['cost_vs_national_mean'] = y_test / national_mean

            # 区域编码
            le_region = LabelEncoder()
            X_train['economic_region_encoded'] = le_region.fit_transform(train_df['economic_region'])
            X_test['economic_region_encoded'] = le_region.transform(test_df['economic_region'])

            # 保存统计量和编码器
            self.region_means = region_means
            self.region_medians = region_medians
            self.region_std = region_std
            self.region_counts = region_counts
            self.national_mean = national_mean
            self.national_median = national_median
            self.national_std = national_std
            self.region_encoder = le_region

            region_features = [col for col in X_train.columns if 'region' in col]
            print(f"新增地区特征: {len(region_features)}个")

        print(f"训练集特征数量: {X_train.shape[1]}")
        print(f"测试集特征数量: {X_test.shape[1]}")

        return X_train, X_test, y_train, y_test

    def train_quantile_model(self):
        """训练95%分位数回归模型"""
        print("=== 4. 训练95%分位数回归模型 ===")

        # 准备数据
        train_df, test_df_10_90, test_df_all, lower_bound, upper_bound = self.prepare_data_with_regions()

        if len(train_df) == 0:
            print("错误: 没有可用的训练数据")
            return None

        self.train_df = train_df
        self.test_df_10_90 = test_df_10_90
        self.test_df_all = test_df_all

        # 分析地区分布
        train_stats, test_10_90_stats, test_all_stats = self.analyze_regional_distribution(train_df, test_df_10_90,
                                                                                           test_df_all)

        # 创建特征
        X_train, X_test_10_90, y_train, y_test_10_90 = self.create_regional_features(train_df, test_df_10_90)
        _, X_test_all, _, y_test_all = self.create_regional_features(train_df, test_df_all)

        if X_train is None:
            print("错误: 特征创建失败")
            return None

        # 训练95%分位数模型
        X_train_scaled = self.scaler.fit_transform(X_train)

        print("训练 95% 分位数模型...")
        try:
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.95,  # 95%分位数
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1
            )

            model.fit(X_train_scaled, y_train)

            self.models[0.95] = {
                'model': model,
                'metrics': {}
            }

            print("  95%分位数模型训练完成")

            # 评估模型
            self.evaluate_model(X_test_10_90, y_test_10_90, "测试集10-90%区间")
            self.evaluate_model(X_test_all, y_test_all, "测试集全部数据")

            # 可视化结果
            self.visualize_results()

            return self.models

        except Exception as e:
            print(f"  训练失败: {e}")
            return None

    def evaluate_model(self, X_test, y_test, test_set_name="测试集"):
        """评估95%分位数模型在特定测试集上的性能"""
        print(f"=== 5. 在{test_set_name}上评估模型 ===")

        if 0.95 not in self.models:
            print("没有可评估的模型")
            return {}

        X_test_scaled = self.scaler.transform(X_test)
        model_info = self.models[0.95]

        print(f"评估 95% 分位数模型在{test_set_name}上的性能...")

        try:
            model = model_info['model']
            y_pred = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            relative_error = mae / y_test.mean() if y_test.mean() > 0 else 0

            def quantile_loss(y_true, y_pred, q):
                error = y_true - y_pred
                return np.maximum(q * error, (q - 1) * error).mean()

            q_loss = quantile_loss(y_test, y_pred, 0.95)

            model_info['metrics'][test_set_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Relative_Error': relative_error,
                'Quantile_Loss': q_loss,
                'predictions': y_pred,
                'actuals': y_test
            }

            print(f"  95%分位数 - R²: {r2:.4f}, MAE: {mae:.2f}, "
                  f"相对误差: {relative_error:.2%}")

            return model_info['metrics'][test_set_name]

        except Exception as e:
            print(f"  评估失败: {e}")
            return {}

    def visualize_results(self):
        """可视化95%分位数模型结果"""
        print("=== 6. 可视化结果 ===")

        if 0.95 not in self.models:
            print("没有可用的模型结果进行可视化")
            return

        model_info = self.models[0.95]

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 1. 性能指标比较
            metrics_10_90 = model_info['metrics']['测试集10-90%区间']
            metrics_all = model_info['metrics']['测试集全部数据']

            metrics_names = ['R²', 'MAE', '相对误差']
            metrics_10_90_values = [metrics_10_90['R2'], metrics_10_90['MAE'], metrics_10_90['Relative_Error']]
            metrics_all_values = [metrics_all['R2'], metrics_all['MAE'], metrics_all['Relative_Error']]

            x = np.arange(len(metrics_names))
            width = 0.35

            bars1 = ax1.bar(x - width / 2, metrics_10_90_values, width,
                            label='10-90%区间', color='lightblue', alpha=0.7)
            bars2 = ax1.bar(x + width / 2, metrics_all_values, width,
                            label='全部数据', color='lightcoral', alpha=0.7)

            ax1.set_xlabel('评估指标')
            ax1.set_ylabel('数值')
            ax1.set_title('95%分位数模型性能比较')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics_names)
            ax1.legend()

            # 在柱状图上添加数值标签
            for bar, value in zip(bars1, metrics_10_90_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{value:.3f}' if value > 0.01 else f'{value:.3%}',
                         ha='center', va='bottom', fontsize=9)

            for bar, value in zip(bars2, metrics_all_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{value:.3f}' if value > 0.01 else f'{value:.3%}',
                         ha='center', va='bottom', fontsize=9)

            # 2. 10-90%区间预测效果散点图
            y_pred_10_90 = metrics_10_90['predictions']
            y_test_10_90 = metrics_10_90['actuals']

            ax2.scatter(y_test_10_90, y_pred_10_90, alpha=0.5, color='blue', s=20)
            max_val = max(y_test_10_90.max(), y_pred_10_90.max())
            ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='理想预测线')
            ax2.set_xlabel('实际费用')
            ax2.set_ylabel('预测费用')
            ax2.set_title('95%分位数模型在10-90%区间的预测效果')
            ax2.legend()

            # 3. 全部数据预测效果散点图
            y_pred_all = metrics_all['predictions']
            y_test_all = metrics_all['actuals']

            ax3.scatter(y_test_all, y_pred_all, alpha=0.5, color='red', s=20)
            max_val = max(y_test_all.max(), y_pred_all.max())
            ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='理想预测线')
            ax3.set_xlabel('实际费用')
            ax3.set_ylabel('预测费用')
            ax3.set_title('95%分位数模型在全部数据上的预测效果')
            ax3.legend()

            # 4. 残差分布图
            residuals_10_90 = y_test_10_90 - y_pred_10_90
            residuals_all = y_test_all - y_pred_all

            ax4.hist(residuals_10_90, bins=50, alpha=0.7, label='10-90%区间', color='lightblue')
            ax4.hist(residuals_all, bins=50, alpha=0.7, label='全部数据', color='lightcoral')
            ax4.set_xlabel('残差')
            ax4.set_ylabel('频数')
            ax4.set_title('95%分位数模型残差分布')
            ax4.legend()

            plt.tight_layout()
            plt.show()

            # 业务建议
            self.business_recommendations()

        except Exception as e:
            print(f"可视化失败: {e}")

    def business_recommendations(self):
        """业务建议"""
        print("=== 7. 业务建议 ===")

        if 0.95 not in self.models:
            print("没有可用的结果进行业务建议")
            return

        model_info = self.models[0.95]
        metrics_10_90 = model_info['metrics'].get('测试集10-90%区间', {})
        metrics_all = model_info['metrics'].get('测试集全部数据', {})

        print("推荐使用分位数: 95%")

        if metrics_10_90:
            print(f"\n测试集10-90%区间性能:")
            print(f"  R²={metrics_10_90['R2']:.4f}, MAE={metrics_10_90['MAE']:.2f}, "
                  f"相对误差={metrics_10_90['Relative_Error']:.2%}")

        if metrics_all:
            print(f"\n测试集全部数据性能:")
            print(f"  R²={metrics_all['R2']:.4f}, MAE={metrics_all['MAE']:.2f}, "
                  f"相对误差={metrics_all['Relative_Error']:.2%}")

        best_r2 = max(metrics_10_90.get('R2', 0), metrics_all.get('R2', 0))

        if best_r2 > 0.7:
            print("\n✅ 模型性能优秀，可直接用于业务预测")
        elif best_r2 > 0.6:
            print("\n✅ 模型性能良好，可作为重要参考")
        elif best_r2 > 0.5:
            print("\n⚠️ 模型性能一般，建议结合业务经验使用")
        else:
            print("\n❌ 模型性能较差，主要用于数据探索")

        print(f"\n💡💡 业务解读:")
        print(f"  • 95%分位数模型适用于高风险案件的费用预估")
        print(f"  • 模型在正常费用区间(10-90%)表现优异，R²达到{metrics_10_90.get('R2', 0):.4f}")
        print(f"  • 在全部数据上相对误差为{metrics_all.get('Relative_Error', 0):.2%}")

        return {
            'best_quantile': 0.95,
            'metrics_10_90': metrics_10_90,
            'metrics_all': metrics_all
        }

    def run_analysis(self):
        """运行分析流程"""
        print("开始95%分位数建模分析")
        print("=" * 60)

        try:
            # 训练模型并评估
            models = self.train_quantile_model()

            if models is not None:
                print("\n✅ 分析完成!")
                return {
                    'models': self.models,
                    'best_quantile': 0.95
                }
            else:
                print("❌ 分析失败")
                return None

        except Exception as e:
            print(f"分析过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None


class CasePredictionSystem:
    """案件费用预测交互系统"""

    def __init__(self, regional_model):
        self.model = regional_model
        self.feature_mappings = self._create_feature_mappings()

    def _create_feature_mappings(self):
        """创建特征映射字典"""
        mappings = {
            '伤势程度': {
                '伤': 1, '残': 2, '死亡': 3
            },
            '治疗情况': {
                '门诊治疗': 1, '住院治疗': 2, '当场死亡': 3
            }
        }
        return mappings

    def get_case_input(self):
        """交互式获取案件信息"""
        print("\n" + "=" * 60)
        print("🧑⚕️ 医疗案件费用预测系统")
        print("=" * 60)

        case_info = {}

        # 1. 伤势程度
        print("\n📊 伤势程度选择:")
        for level, code in self.feature_mappings['伤势程度'].items():
            print(f"  {code}: {level}")
        injury_level = input("请选择伤势程度编号(1-3): ").strip()
        case_info['伤势程度'] = int(injury_level) if injury_level.isdigit() and 1 <= int(injury_level) <= 3 else 1

        # 2. 责任系数
        liability = input("\n⚖️ 责任赔偿系数(0-1之间，如0.7表示70%责任): ").strip()
        case_info['责任赔偿系数'] = float(liability) if liability.replace('.', '').isdigit() else 0.5

        # 3. 手术次数
        surgery_count = input("\n🏥 手术次数: ").strip()
        case_info['手术次数统计'] = int(surgery_count) if surgery_count.isdigit() else 0

        # 4. 治疗情况
        print("\n💊 治疗情况选择:")
        for treatment, code in self.feature_mappings['治疗情况'].items():
            print(f"  {code}: {treatment}")
        treatment_code = input("请选择治疗情况编号(1-3): ").strip()
        case_info['治疗情况'] = int(treatment_code) if treatment_code.isdigit() and 1 <= int(treatment_code) <= 3 else 2

        # 5. 损伤数量输入 - 调整顺序，将其他损伤类数量放在最后
        print("\n🤕 损伤数量输入:")
        injury_types = [
            '骨折类数量', '软组织损伤类数量', '颅脑损伤类数量',
            '内脏损伤类数量', '神经损伤类数量', '其他损伤类数量'
        ]

        for injury_type in injury_types:
            count = input(f"{injury_type}: ").strip()
            case_info[injury_type] = int(count) if count.isdigit() else 0

        # 6. 护理费数量
        nursing_fee = input("\n🩺 护理费数量(护理天数): ").strip()
        case_info['护理费数量'] = int(nursing_fee) if nursing_fee.isdigit() else 0

        # 7. 地域信息
        print("\n🌍 地域信息:")
        province_code = input("请输入省份代码(如31-上海, 44-广东等): ").strip()
        case_info['地域'] = province_code if province_code else '31'

        return case_info

    def preprocess_case_data(self, case_info):
        """预处理案件数据"""
        # 创建基础特征DataFrame
        base_features = [
            '伤势程度', '手术次数统计', '治疗情况', '责任赔偿系数',
            '骨折类数量', '护理费数量', '软组织损伤类数量', '颅脑损伤类数量',
            '内脏损伤类数量', '神经损伤类数量', '其他损伤类数量', '地域'
        ]

        # 确保所有特征都存在
        for feature in base_features:
            if feature not in case_info:
                case_info[feature] = 0

        # 创建DataFrame
        case_df = pd.DataFrame([case_info])

        # 确保province_code列存在
        if 'province_code' not in case_df.columns and '地域' in case_df.columns:
            case_df['province_code'] = case_df['地域'].astype(str).str[:2]

        # 添加经济区域信息
        case_df = self.model.add_economic_regions(case_df)

        # 提取特征
        available_features = [f for f in self.model.feature_names if f in case_df.columns and f != '地域']
        X_case = case_df[available_features].copy()

        # 处理分类变量
        categorical_cols = X_case.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if hasattr(self.model, 'label_encoders') and col in self.model.label_encoders:
                le = self.model.label_encoders[col]
                X_case[col] = le.transform(X_case[col].astype(str))
            else:
                X_case[col] = X_case[col].astype(str).astype('category').cat.codes

        # 添加区域特征
        if hasattr(self.model, 'region_means') and 'economic_region' in case_df.columns:
            region = case_df['economic_region'].iloc[0]

            X_case['region_mean_cost'] = self.model.region_means.get(region,
                                                                     getattr(self.model, 'national_mean', 10000))
            X_case['region_median_cost'] = self.model.region_medians.get(region,
                                                                         getattr(self.model, 'national_median', 8000))
            X_case['region_std_cost'] = self.model.region_std.get(region, getattr(self.model, 'national_std', 3000))
            X_case['region_sample_count'] = self.model.region_counts.get(region, 0)

            national_mean = getattr(self.model, 'national_mean', 10000)
            X_case['cost_vs_region_mean'] = national_mean / X_case['region_mean_cost']
            X_case['cost_vs_national_mean'] = 1.0

            # 区域编码
            if hasattr(self.model, 'region_encoder'):
                try:
                    X_case['economic_region_encoded'] = self.model.region_encoder.transform([region])[0]
                except:
                    X_case['economic_region_encoded'] = 0

        return X_case

    def predict_case_cost(self, case_info=None):
        """预测案件费用"""
        if case_info is None:
            case_info = self.get_case_input()

        print("\n" + "=" * 60)
        print("🔮 正在进行费用预测...")
        print("=" * 60)

        try:
            # 预处理案件数据
            X_case = self.preprocess_case_data(case_info)

            # 特征标准化
            X_case_scaled = self.model.scaler.transform(X_case)

            # 使用95%分位数模型进行预测
            if 0.95 in self.model.models:
                model = self.model.models[0.95]['model']
                prediction = model.predict(X_case_scaled)[0]
                predictions = {'95%分位数': round(prediction, 2)}
            else:
                print("❌ 95%分位数模型未训练")
                return None

            # 显示预测结果
            predicted_value = self.display_prediction_results(case_info, predictions)

            return predictions, predicted_value

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def display_prediction_results(self, case_info, predictions):
        """显示预测结果"""
        print("\n🎯 预测结果:")
        print("=" * 40)

        # 显示输入信息摘要
        print("📋 案件信息摘要:")
        print(f"  伤势程度: {self._get_key_by_value(self.feature_mappings['伤势程度'], case_info.get('伤势程度', 1))}")
        print(f"  责任系数: {case_info.get('责任赔偿系数', '未知')}")
        print(f"  手术次数: {case_info.get('手术次数统计', 0)}")
        print(f"  治疗情况: {self._get_key_by_value(self.feature_mappings['治疗情况'], case_info.get('治疗情况', 2))}")

        # 显示损伤情况
        injury_features = ['骨折类数量', '软组织损伤类数量', '颅脑损伤类数量',
                           '内脏损伤类数量', '神经损伤类数量', '其他损伤类数量']
        injury_summary = []
        for feature in injury_features:
            count = case_info.get(feature, 0)
            if count > 0:
                injury_summary.append(f"{feature}: {count}")

        if injury_summary:
            print(f"  损伤情况: {', '.join(injury_summary)}")

        # 显示预测费用
        print("\n💰 费用预测结果:")
        for quantile, cost in predictions.items():
            print(f"  {quantile}: ¥{cost:,.2f}")

        # 业务解读
        print("\n💡 业务解读:")
        if '95%分位数' in predictions:
            high_cost = predictions['95%分位数']
            print(f"  • 风险预估(95%分位数): ¥{high_cost:,.2f}")
            print(f"  • 此预估考虑了高风险情况，适合作为最高费用参考")

        print("=" * 40)

        # 返回预测值用于异常检测
        return high_cost if '95%分位数' in predictions else None

    def _get_key_by_value(self, dictionary, value):
        """根据值获取字典中的键"""
        for key, val in dictionary.items():
            if val == value:
                return key
        return "未知"


class BoxplotAnomalyDetector:
    def __init__(self, file_path=None, data_series=None, column_name='fee_all'):
        """
        初始化异常检测器
        :param file_path: Excel文件路径
        :param data_series: 数据序列 (DataFrame列)
        :param column_name: 要分析的列名
        """
        self.file_path = file_path
        self.column_name = column_name
        self.data = None
        self.percentile_95 = None
        self.yellow_threshold = None
        self.orange_threshold = None
        self.red_threshold = None

        # 测试数据计数器
        self.test_count = 0
        self.test_data_list = []

        # 读取数据
        if not self.load_data(file_path, data_series):
            return

        # 计算统计量
        self.calculate_statistics()

    def load_data(self, file_path, data_series):
        """读取数据 - 支持文件路径或数据序列"""
        try:
            if data_series is not None:
                # 使用提供的数据序列
                self.data = data_series.dropna()
                print(f"成功读取数据序列，形状: {len(self.data)}")
                return True
            elif file_path is not None and os.path.exists(file_path):
                # 从文件读取数据
                df = pd.read_excel(file_path)

                # 检查列是否存在
                if self.column_name not in df.columns:
                    available_columns = ", ".join(df.columns)
                    print(f"错误: 列 '{self.column_name}' 在数据中不存在")
                    print(f"可用的列有: {available_columns}")
                    return False

                self.data = df[self.column_name].dropna()
                print(f"成功从文件读取 {len(self.data)} 条数据")
                return True
            else:
                print("错误: 没有提供有效的数据源")
                return False

        except Exception as e:
            print(f"读取数据时出错: {e}")
            return False

    def calculate_statistics(self):
        """计算百分位数和预警阈值"""
        try:
            # 计算95%分位数
            self.percentile_95 = self.data.quantile(0.95)

            # 修正阈值计算逻辑 - 直接基于95%分位数的倍数
            self.yellow_threshold = self.percentile_95 * 1.1  # 10%增幅
            self.orange_threshold = self.percentile_95 * 1.3  # 30%增幅
            self.red_threshold = self.percentile_95 * 1.5  # 50%增幅

            print("\n=== 异常检测统计量计算完成 ===")
            print(f"95%分位数: {self.percentile_95:.2f}")
            print(f"黄色预警阈值 (95%分位数的1.1倍): {self.yellow_threshold:.2f}")
            print(f"橙色预警阈值 (95%分位数的1.3倍): {self.orange_threshold:.2f}")
            print(f"红色预警阈值 (95%分位数的1.5倍): {self.red_threshold:.2f}")

        except Exception as e:
            print(f"计算统计量时出错: {e}")

    def update_statistics(self):
        """当测试数据达到100次时更新统计量"""
        try:
            if len(self.test_data_list) >= 100:
                print(f"\n测试数据已达到 {len(self.test_data_list)} 条，更新统计量...")

                # 合并原始数据和测试数据
                combined_data = pd.concat([self.data, pd.Series(self.test_data_list)])

                # 重新计算统计量
                self.percentile_95 = combined_data.quantile(0.95)
                self.yellow_threshold = self.percentile_95 * 1.1
                self.orange_threshold = self.percentile_95 * 1.3
                self.red_threshold = self.percentile_95 * 1.5

                print("=== 统计量已更新 ===")
                print(f"新的95%分位数: {self.percentile_95:.2f}")
                print(f"新的黄色预警阈值: {self.yellow_threshold:.2f}")
                print(f"新的橙色预警阈值: {self.orange_threshold:.2f}")
                print(f"新的红色预警阈值: {self.red_threshold:.2f}")

                # 重置计数器
                self.test_data_list = []
                self.test_count = 0

                return True
            return False
        except Exception as e:
            print(f"更新统计量时出错: {e}")
            return False

    def is_initialized(self):
        """检查检测器是否成功初始化"""
        return self.data is not None and self.percentile_95 is not None

    def detect_anomaly(self, value):
        """
        检测单个值是否为异常值
        :param value: 要检测的数值
        :return: 检测结果字符串和异常类型
        """
        if not self.is_initialized():
            return "错误: 检测器未正确初始化，无法进行异常检测", "error"

        try:
            value = float(value)

            # 记录测试数据
            self.test_count += 1
            self.test_data_list.append(value)

            # 检查是否需要更新统计量
            self.update_statistics()

            # 根据新的预警规则进行分类
            if value > self.red_threshold:
                return f"值 {value:,.2f} 属于 **红色预警** (超过95%分位数50%以上)", "red"
            elif value > self.orange_threshold:
                return f"值 {value:,.2f} 属于 **橙色预警** (超过95%分位数30%-50%)", "orange"
            elif value > self.yellow_threshold:
                return f"值 {value:,.2f} 属于 **黄色预警** (超过95%分位数10%-30%)", "yellow"
            elif value > self.percentile_95:
                return f"值 {value:,.2f} 属于 **轻微超出** (超过95%分位数但在10%以内)", "slight"
            else:
                return f"值 {value:,.2f} 属于 **正常范围**", "normal"

        except ValueError:
            return "错误: 请输入有效的数值", "error"

    def classify_anomaly_category(self, value):
        """
        分类异常类别（返回类别编号）
        :param value: 要检测的数值
        :return: 异常类别编号 (0:正常, 1:轻微超出, 2:黄色预警, 3:橙色预警, 4:红色预警)
        """
        if not self.is_initialized():
            return -1  # 错误代码

        try:
            value = float(value)

            if value > self.red_threshold:
                return 4  # 红色预警
            elif value > self.orange_threshold:
                return 3  # 橙色预警
            elif value > self.yellow_threshold:
                return 2  # 黄色预警
            elif value > self.percentile_95:
                return 1  # 轻微超出
            else:
                return 0  # 正常范围

        except ValueError:
            return -1  # 错误代码

    def print_warning_stats(self):
        """打印预警统计信息"""
        if not self.is_initialized():
            return

        try:
            # 计算各预警级别的数量
            normal = self.data[self.data <= self.percentile_95]
            slight_outliers = self.data[(self.data > self.percentile_95) & (self.data <= self.yellow_threshold)]
            yellow_warnings = self.data[(self.data > self.yellow_threshold) & (self.data <= self.orange_threshold)]
            orange_warnings = self.data[(self.data > self.orange_threshold) & (self.data <= self.red_threshold)]
            red_warnings = self.data[self.data > self.red_threshold]

            print("\n=== 异常检测预警统计 ===")
            print(f"数据总量: {len(self.data)}")
            print(f"95%分位数: {self.percentile_95:.2f}")
            print(f"正常范围数量: {len(normal)} ({len(normal) / len(self.data) * 100:.2f}%)")
            print(f"轻微超出数量: {len(slight_outliers)} ({len(slight_outliers) / len(self.data) * 100:.2f}%)")
            print(f"黄色预警数量: {len(yellow_warnings)} ({len(yellow_warnings) / len(self.data) * 100:.2f}%)")
            print(f"橙色预警数量: {len(orange_warnings)} ({len(orange_warnings) / len(self.data) * 100:.2f}%)")
            print(f"红色预警数量: {len(red_warnings)} ({len(red_warnings) / len(self.data) * 100:.2f}%)")

            print(f"\n当前测试数据计数: {self.test_count}")
            print(f"测试数据列表长度: {len(self.test_data_list)}")

            if len(red_warnings) > 0:
                print(f"\n红色预警数据样例:")
                print(red_warnings.head(10).to_string())

        except Exception as e:
            print(f"计算预警统计时出错: {e}")


class IntegratedPredictionSystem:
    """集成的预测和异常检测系统"""

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.regional_model = None
        self.prediction_system = None
        self.anomaly_detector = None
        self.consistency_results = {}

    def initialize_system(self):
        """初始化整个系统"""
        print("正在初始化集成预测系统...")

        try:
            # 1. 初始化区域模型
            self.regional_model = RegionalQuantileModel(self.train_path, self.test_path)
            if self.regional_model.train_data is None:
                print("区域模型初始化失败")
                return False

            # 2. 训练模型
            results = self.regional_model.run_analysis()
            if results is None:
                print("模型训练失败")
                return False

            # 3. 初始化预测系统
            self.prediction_system = CasePredictionSystem(self.regional_model)

            # 4. 初始化异常检测器（使用训练数据）
            self.anomaly_detector = BoxplotAnomalyDetector(
                data_series=self.regional_model.train_df['fee_all']
            )

            if not self.anomaly_detector.is_initialized():
                print("异常检测器初始化失败")
                return False

            # 5. 计算预测值与真实值异常类别的一致性
            self.calculate_anomaly_consistency()

            print("\n✅ 集成系统初始化完成!")
            return True

        except Exception as e:
            print(f"系统初始化失败: {e}")
            return False

    def calculate_anomaly_consistency(self):
        """计算预测值与真实值异常类别的一致性"""
        print("\n=== 计算异常类别一致性 ===")

        try:
            # 获取测试集的预测值和真实值
            if 0.95 in self.regional_model.models:
                model_info = self.regional_model.models[0.95]
                metrics_all = model_info['metrics'].get('测试集全部数据', {})

                if 'predictions' in metrics_all and 'actuals' in metrics_all:
                    predictions = metrics_all['predictions']
                    actuals = metrics_all['actuals']

                    # 分类预测值和真实值的异常类别
                    pred_categories = [self.anomaly_detector.classify_anomaly_category(pred) for pred in predictions]
                    actual_categories = [self.anomaly_detector.classify_anomaly_category(actual) for actual in actuals]

                    # 过滤掉错误分类
                    valid_indices = [i for i, (p, a) in enumerate(zip(pred_categories, actual_categories))
                                     if p != -1 and a != -1]

                    if len(valid_indices) > 0:
                        pred_categories_valid = [pred_categories[i] for i in valid_indices]
                        actual_categories_valid = [actual_categories[i] for i in valid_indices]

                        # 计算准确率
                        accuracy = accuracy_score(actual_categories_valid, pred_categories_valid)

                        # 统计各类别分布
                        pred_dist = Counter(pred_categories_valid)
                        actual_dist = Counter(actual_categories_valid)

                        # 类别标签映射
                        category_labels = {
                            0: '正常范围',
                            1: '轻微超出',
                            2: '黄色预警',
                            3: '橙色预警',
                            4: '红色预警'
                        }

                        print(f"异常类别一致性准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
                        print(f"有效样本数量: {len(valid_indices)}")

                        print("\n预测值异常类别分布:")
                        for category_id, count in sorted(pred_dist.items()):
                            label = category_labels.get(category_id, f'未知({category_id})')
                            print(f"  {label}: {count}个 ({count / len(pred_categories_valid) * 100:.2f}%)")

                        print("\n真实值异常类别分布:")
                        for category_id, count in sorted(actual_dist.items()):
                            label = category_labels.get(category_id, f'未知({category_id})')
                            print(f"  {label}: {count}个 ({count / len(actual_categories_valid) * 100:.2f}%)")

                        # 保存结果
                        self.consistency_results = {
                            'accuracy': accuracy,
                            'pred_dist': pred_dist,
                            'actual_dist': actual_dist,
                            'valid_samples': len(valid_indices)
                        }
                    else:
                        print("没有有效的样本可用于一致性分析")
                else:
                    print("测试集预测结果不可用")
            else:
                print("95%分位数模型不可用")

        except Exception as e:
            print(f"计算异常类别一致性时出错: {e}")

    def run_interactive_system(self):
        """运行交互式系统"""
        if not self.initialize_system():
            return

        print("\n" + "=" * 70)
        print("🎯 集成预测与异常检测系统")
        print("=" * 70)

        while True:
            print("\n请选择操作:")
            print("1. 单案件预测")
            print("2. 查看异常检测统计")
            print("3. 查看异常类别一致性分析")
            print("4. 退出系统")

            choice = input("请输入选择 (1-4): ").strip()

            if choice == '1':
                self.predict_and_detect()
            elif choice == '2':
                self.show_anomaly_stats()
            elif choice == '3':
                self.show_consistency_analysis()
            elif choice == '4' or choice.lower() in ['quit', 'exit', 'q']:
                print("感谢使用集成预测系统！")
                break
            else:
                print("无效选择，请重新输入")

    def predict_and_detect(self):
        """预测并检测异常"""
        try:
            # 获取预测结果
            predictions, predicted_value = self.prediction_system.predict_case_cost()

            if predictions and predicted_value is not None:

                print("\n" + "=" * 60)
                print("🔍 异常检测分析")
                print("=" * 60)

                # 进行异常检测
                result, warning_type = self.anomaly_detector.detect_anomaly(predicted_value)

                # 根据预警类型添加颜色标识
                if warning_type == "red":
                    print(f"🔴 {result}")
                elif warning_type == "orange":
                    print(f"🟠 {result}")
                elif warning_type == "yellow":
                    print(f"🟡 {result}")
                elif warning_type == "slight":
                    print(f"🟢 {result}")
                elif warning_type == "normal":
                    print(f"✅ {result}")
                else:
                    print(f"❌ {result}")

                # 显示参考范围
                print(f"\n📊 参考范围:")
                print(f"   正常范围: ≤ {self.anomaly_detector.percentile_95:,.2f}")
                print(
                    f"   黄色预警: {self.anomaly_detector.percentile_95:,.2f} ~ {self.anomaly_detector.yellow_threshold:,.2f} (10%-30%增幅)")
                print(
                    f"   橙色预警: {self.anomaly_detector.yellow_threshold:,.2f} ~ {self.anomaly_detector.orange_threshold:,.2f} (30%-50%增幅)")
                print(f"   红色预警: > {self.anomaly_detector.red_threshold:,.2f} (50%以上增幅)")
                print(f"   当前测试计数: {self.anomaly_detector.test_count}")

            else:
                print("❌ 预测失败，无法进行异常检测")

        except Exception as e:
            print(f"预测和检测过程中出错: {e}")

    def show_anomaly_stats(self):
        """显示异常检测统计"""
        if self.anomaly_detector:
            self.anomaly_detector.print_warning_stats()
        else:
            print("异常检测器未初始化")

    def show_consistency_analysis(self):
        """显示异常类别一致性分析结果"""
        if self.consistency_results:
            accuracy = self.consistency_results['accuracy']
            pred_dist = self.consistency_results['pred_dist']
            actual_dist = self.consistency_results['actual_dist']
            valid_samples = self.consistency_results['valid_samples']

            print("\n" + "=" * 60)
            print("📊 异常类别一致性分析")
            print("=" * 60)

            print(f"准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"有效样本数量: {valid_samples}")

            # 类别标签映射
            category_labels = {
                0: '正常范围',
                1: '轻微超出',
                2: '黄色预警',
                3: '橙色预警',
                4: '红色预警'
            }

            print("\n预测值异常类别分布:")
            for category_id, count in sorted(pred_dist.items()):
                label = category_labels.get(category_id, f'未知({category_id})')
                percentage = count / valid_samples * 100
                print(f"  {label}: {count}个 ({percentage:.2f}%)")

            print("\n真实值异常类别分布:")
            for category_id, count in sorted(actual_dist.items()):
                label = category_labels.get(category_id, f'未知({category_id})')
                percentage = count / valid_samples * 100
                print(f"  {label}: {count}个 ({percentage:.2f}%)")

            # 业务解读
            print(f"\n💡 业务解读:")
            if accuracy > 0.8:
                print("  ✅ 模型在异常类别识别上表现优秀")
            elif accuracy > 0.6:
                print("  ✅ 模型在异常类别识别上表现良好")
            elif accuracy > 0.4:
                print("  ⚠️ 模型在异常类别识别上表现一般")
            else:
                print("  ❌ 模型在异常类别识别上表现较差")

        else:
            print("没有可用的异常类别一致性分析结果")


# 主函数
def main():
    """主函数 - 集成预测系统"""
    print("集成预测与异常检测系统")
    print("=" * 50)

    # 文件路径 - 请根据实际情况修改
    train_path = "清洗后数据.xlsx"  # 修改为您的训练集文件路径
    test_path = "测试集.xlsx"  # 修改为您的测试集文件路径

    try:
        # 创建集成系统实例
        integrated_system = IntegratedPredictionSystem(train_path, test_path)

        # 运行交互式系统
        integrated_system.run_interactive_system()

    except FileNotFoundError:
        print(f"错误: 找不到数据文件")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


