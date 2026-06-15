# ==================== 1. 导入库 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 设置中文字体（避免乱码，Mac/Win通用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 2. 生成模拟数据（可替换成真实数据）====================
np.random.seed(42)
n_users = 5000

data = {
    '浏览时长(分钟)': np.random.exponential(scale=5, size=n_users),
    '浏览页面数': np.random.poisson(lam=4, size=n_users),
    '设备类型': np.random.choice(['移动端', 'PC端'], size=n_users, p=[0.6, 0.4]),
    '是否会员': np.random.choice(['否', '是'], size=n_users, p=[0.7, 0.3]),
    '历史购买次数': np.random.poisson(lam=1, size=n_users),
    '加购次数': np.random.poisson(lam=0.5, size=n_users),
    '访问时段': np.random.randint(0, 24, size=n_users)
}
df = pd.DataFrame(data)

# 构造购买标签（基于规则的隐式逻辑）
purchase_prob = (
        0.2 * (df['是否会员'] == '是') +
        0.3 * (df['浏览时长(分钟)'] > 4) +
        0.4 * (df['加购次数'] > 0) +
        0.1 * (df['浏览页面数'] > 5)
)
purchase_prob = purchase_prob + np.random.uniform(0, 0.2, size=n_users)
purchase_prob = np.clip(purchase_prob, 0, 1)
df['是否购买'] = (purchase_prob > 0.5).astype(int)

# 特征工程：人均页面停留时长
df['人均停留时长(秒/页)'] = df['浏览时长(分钟)'] * 60 / (df['浏览页面数'] + 1)

# ==================== 3. 准备建模数据 ====================
# 将类别变量转为数值
df_model = df.copy()
df_model['设备类型'] = df_model['设备类型'].map({'移动端': 0, 'PC端': 1})
df_model['是否会员'] = df_model['是否会员'].map({'否': 0, '是': 1})

feature_cols = ['浏览时长(分钟)', '浏览页面数', '设备类型', '是否会员',
                '历史购买次数', '加购次数', '访问时段', '人均停留时长(秒/页)']
X = df_model[feature_cols]
y = df_model['是否购买']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（仅对逻辑回归）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 4. 训练模型 ====================
# 逻辑回归
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# 随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# 评估指标
metrics = {
    '模型': ['逻辑回归', '随机森林'],
    '准确率': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)],
    '召回率': [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf)],
    'AUC': [roc_auc_score(y_test, y_proba_lr), roc_auc_score(y_test, y_proba_rf)]
}
metrics_df = pd.DataFrame(metrics)

# 特征重要性（随机森林）
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# ROC曲线数据
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

# 混淆矩阵（随机森林）
cm = confusion_matrix(y_test, y_pred_rf)
cm_labels = ['未购买', '购买']

# ==================== 5. 生成Plotly图表 ====================
# 5.1 特征分布（交互式箱线图）
fig_dist = px.box(df, x='是否购买', y='浏览时长(分钟)', color='是否购买',
                  title='购买 vs 未购买用户的浏览时长分布',
                  labels={'是否购买': '是否购买', '浏览时长(分钟)': '浏览时长(分钟)'})

# 5.2 特征重要性条形图
fig_imp = px.bar(x=feat_imp.values, y=feat_imp.index, orientation='h',
                 title='随机森林特征重要性', labels={'x': '重要性分数', 'y': '特征'})
fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})

# 5.3 ROC曲线对比
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines', name='逻辑回归 (AUC={:.3f})'.format(metrics['AUC'][0])))
fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name='随机森林 (AUC={:.3f})'.format(metrics['AUC'][1])))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='随机猜测', line=dict(dash='dash')))
fig_roc.update_layout(title='ROC曲线对比', xaxis_title='假正率', yaxis_title='真正率')

# 5.4 混淆矩阵热力图
fig_cm = px.imshow(cm, text_auto=True, x=cm_labels, y=cm_labels,
                   color_continuous_scale='Blues', title='随机森林混淆矩阵')
fig_cm.update_layout(xaxis_title='预测值', yaxis_title='真实值')

# 5.5 模型指标对比表
fig_table = go.Figure(data=[go.Table(
    header=dict(values=list(metrics_df.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[metrics_df['模型'], metrics_df['准确率'], metrics_df['召回率'], metrics_df['AUC']],
               fill_color='lavender', align='left'))
])
fig_table.update_layout(title='模型性能对比')

# ==================== 6. 生成HTML报告 ====================
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>用户购买行为预测分析报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f7fa; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-left: 6px solid #3498db; padding-left: 20px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
        .chart {{ margin-bottom: 40px; }}
        .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 12px; border-top: 1px solid #ecf0f1; padding-top: 20px; }}
        @media (max-width: 600px) {{
            .container {{ padding: 12px; }}
            h1 {{ font-size: 24px; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 用户购买意向预测分析报告</h1>
    <p><strong>项目背景</strong>：基于用户历史行为数据（浏览时长、页面数、会员状态、加购次数等），构建机器学习模型预测用户最终是否会购买商品。本报告展示了数据分析、模型训练及评估结果。</p>

    <h2>📈 关键数据洞察</h2>
    <div class="chart">
        {fig_dist.to_html(full_html=False, include_plotlyjs='cdn')}
    </div>

    <h2>⭐ 特征重要性（随机森林）</h2>
    <div class="chart">
        {fig_imp.to_html(full_html=False, include_plotlyjs='cdn')}
    </div>

    <h2>📉 ROC曲线对比</h2>
    <div class="chart">
        {fig_roc.to_html(full_html=False, include_plotlyjs='cdn')}
    </div>

    <h2>🧮 混淆矩阵</h2>
    <div class="chart">
        {fig_cm.to_html(full_html=False, include_plotlyjs='cdn')}
    </div>

    <h2>📋 模型性能指标</h2>
    <div class="chart">
        {fig_table.to_html(full_html=False, include_plotlyjs='cdn')}
    </div>

    <div class="footer">
        <p>报告生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据量：{n_users} 条用户记录 | 模型：逻辑回归 / 随机森林</p>
        <p>注：本报告基于模拟数据生成，真实业务场景可替换为实际数据。</p>
    </div>
</div>
</body>
</html>
"""

# 保存HTML文件
with open('user_behavior_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ 报告已生成：user_behavior_report.html")
print("👉 你可以：")
print("   1. 双击用浏览器打开（手机可直接发送文件并用浏览器查看）")
print("   2. 截图重要图表，打印成PDF带去面试")
print("   3. 将文件上传到网盘，给HR分享链接")
