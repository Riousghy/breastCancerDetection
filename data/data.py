# 重新加载必要的库和数据 / Reload necessary libraries and data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 创建 "data analysis" 目录 / Create "data analysis" directory if not exist
results_dir = "data_nalysis"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 读取数据 / Load
data_file_path = "data/breast+cancer+wisconsin+diagnostic/wdbc.data"
column_names = [
    "ID", "Diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

df = pd.read_csv(data_file_path, header=None, names=column_names)

# 删除无用的 ID 列 / Drop ID 
df.drop(columns=["ID"], inplace=True)

# 将目标变量 Diagnosis 转换为数值 (M=1, B=0) / Convert target variable (M=1, B=0)（label encoding）
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

# 归一化数据 / Normalize data using StandardScaler
scaler = StandardScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# 计算数据的基本统计信息 / Compute basic statistical information
data_summary = df.describe()

# 计算特征之间的相关性 / Compute feature correlation matrix
correlation_matrix = df.corr()

# 进行 PCA 进行数据降维 / Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df.iloc[:, 1:])  # 忽略 Diagnosis 列 / Ignore Diagnosis column

# 训练随机森林分类器以计算特征重要性 / Train a Random Forest classifier to compute feature importance
X = df.iloc[:, 1:]  # 特征 / Features
y = df["Diagnosis"]  # 目标变量 / Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 获取特征重要性 / Get feature importance
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": clf.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# 输出数据统计信息 / Print Data Summary
print("数据统计信息 / Data Summary")
print(data_summary)

# 输出相关性矩阵 / Print Correlation Matrix
print("相关性矩阵 / Correlation Matrix")
print(correlation_matrix)

# 输出特征重要性 / Print Feature Importance
print("特征重要性 / Feature Importance")
print(feature_importances)

# 可视化目标变量的分布 / Visualizing the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Diagnosis"], palette="coolwarm")
plt.title("Diagnosis Distribution (Benign vs. Malignant)")
plt.xlabel("Diagnosis (0: Benign, 1: Malignant)")
plt.ylabel("Count")
plt.savefig(os.path.join(results_dir, "diagnosis_distribution.png"))  # 保存图片 / Save figure
plt.close()

# 相关性热图 / Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(results_dir, "correlation_heatmap.png"))  # 保存图片 / Save figure
plt.close()

# PCA 结果可视化 / PCA result visualization
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df["Diagnosis"], cmap="coolwarm", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Data")
plt.colorbar(label="Diagnosis (0: Benign, 1: Malignant)")
plt.savefig(os.path.join(results_dir, "pca_projection.png"))  # 保存图片 / Save figure
plt.close()

# 特征重要性可视化 / Feature importance visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances["Importance"], y=feature_importances["Feature"], palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Diagnosis Prediction")
plt.savefig(os.path.join(results_dir, "feature_importance.png"))  # 保存图片 / Save figure
plt.close()

print(f"所有图像已保存在 '{results_dir}' 文件夹中！")
