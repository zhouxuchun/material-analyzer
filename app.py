# app.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from flask import Flask, request, jsonify, render_template_string
import io
import base64

app = Flask(__name__)

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能材料力学性能分析器</title>
    <style>
        body { font-family: "Microsoft YaHei", sans-serif; margin: 20px; background: #f5f7fa; }
        .container { max-width: 900px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; }
        .upload-box { border: 2px dashed #3498db; padding: 20px; text-align: center; margin: 20px 0; background: #f8fcff; }
        input[type="file"] { margin: 10px 0; }
        button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #2980b9; }
        .result { margin-top: 20px; }
        pre { background: #f1f1f1; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; }
        img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #eee; border-radius: 5px; }
        .error { color: red; background: #ffeaea; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 智能材料力学性能分析器</h1>
        <p style="text-align:center; color:#555;">上传 CSV 文件（需包含 Strain 和 Stress 两列）</p>
        
        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        <form method="POST" enctype="multipart/form-data">
            <div class="upload-box">
                <input type="file" name="file" accept=".csv" required>
                <br><br>
                <button type="submit">开始分析</button>
            </div>
        </form>

        {% if plot_url %}
            <div class="result">
                <h2>📊 分析结果</h2>
                <img src="{{ plot_url }}" alt="应力-应变曲线">
                <pre>{{ results_text }}</pre>
            </div>
        {% endif %}
    </div>
</body>
</html>
'''

class MaterialAnalyzer:
    def __init__(self, stress, strain):
        self.stress = stress
        self.strain = strain
        self.results = {}

    def calculate_properties(self):
        try:
            stress_smooth = self.stress
            if len(self.stress) > 10:
                window_size = min(11, len(self.stress))
                if window_size % 2 == 0:
                    window_size -= 1
                stress_smooth = savgol_filter(self.stress, window_size, 3)

            elastic_region = self.strain <= 0.002
            if np.sum(elastic_region) > 5:
                slope = np.polyfit(self.strain[elastic_region][:10], stress_smooth[elastic_region][:10], 1)[0]
                youngs_modulus = slope / 1e9
            else:
                youngs_modulus = 200

            offset_line = youngs_modulus * 1e9 * (self.strain - 0.002)
            diff = np.abs(stress_smooth - offset_line)
            valid_indices = self.strain > 0.002
            if np.any(valid_indices):
                yield_idx = np.argmin(diff[valid_indices]) + np.argmax(valid_indices)
                yield_strength = stress_smooth[yield_idx]
            else:
                yield_strength = np.max(stress_smooth) * 0.8

            tensile_strength = np.max(stress_smooth)
            fracture_strain = self.strain[np.argmax(stress_smooth)]
            toughness = np.trapz(stress_smooth, self.strain)

            self.results = {
                '弹性模量 (GPa)': max(youngs_modulus, 0),
                '屈服强度 (MPa)': max(yield_strength, 0),
                '抗拉强度 (MPa)': max(tensile_strength, 0),
                '断裂应变': max(fracture_strain, 0),
                '韧性 (MJ/m³)': max(toughness / 1000, 0)
            }
            return True
        except Exception as e:
            print(f"计算错误: {e}")
            return False

    def plot_to_base64(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(self.strain, self.stress, 'b-', alpha=0.7, label='实验数据')
        if self.results:
            try:
                yield_idx = np.argmin(np.abs(self.stress - self.results['屈服强度 (MPa)']))
                ax.plot(self.strain[yield_idx], self.stress[yield_idx], 'ro', label='屈服点')
                max_idx = np.argmax(self.stress)
                ax.plot(self.strain[max_idx], self.stress[max_idx], 'go', label='抗拉强度')
            except:
                pass
        ax.set_xlabel('应变')
        ax.set_ylabel('应力 (MPa)')
        ax.set_title('材料应力-应变曲线')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if self.results:
            textstr = '\n'.join([f'{k}: {v:.2f}' for k, v in self.results.items()])
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE, error="未选择文件")
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return render_template_string(HTML_TEMPLATE, error="请上传 .csv 文件")

        df = pd.read_csv(file)
        if len(df.columns) < 2:
            return render_template_string(HTML_TEMPLATE, error="CSV 至少需要两列数据")

        df.columns = ['Strain', 'Stress'] + list(df.columns[2:])
        strain = df['Strain'].astype(float).values
        stress = df['Stress'].astype(float).values

        analyzer = MaterialAnalyzer(stress=stress, strain=strain)
        success = analyzer.calculate_properties()
        if not success:
            return render_template_string(HTML_TEMPLATE, error="数据分析失败，请检查数据格式")

        plot_url = analyzer.plot_to_base64()
        results_text = "=== 材料力学性能分析报告 ===\n\n"
        for k, v in analyzer.results.items():
            results_text += f"{k}: {v:.2f}\n"

        return render_template_string(HTML_TEMPLATE, plot_url=plot_url, results_text=results_text)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f"处理出错: {str(e)}")

# Vercel 需要这个入口
if __name__ == '__main__':
    app.run()
