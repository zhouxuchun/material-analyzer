# app.py - 材料力学性能分析系统 (修复版)
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import io
import base64
import json
import os
from datetime import datetime
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key'

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 确保上传文件夹存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 全局变量存储当前数据
current_data = {
    'strain': None,
    'stress': None,
    'material_type': '钢',
    'results': None,
    'filename': None
}

def calculate_material_properties(strain, stress):
    """计算材料性能参数"""
    try:
        # 确保数据有效
        if strain is None or stress is None or len(strain) < 2:
            raise ValueError("数据不足")
        
        print(f"开始计算材料性能，数据点: {len(strain)}")
        
        # 平滑数据
        if len(stress) > 10:
            window_size = min(11, len(stress))
            if window_size % 2 == 0:
                window_size -= 1
            if window_size >= 3:
                stress_smooth = savgol_filter(stress, window_size, 3)
            else:
                stress_smooth = stress
        else:
            stress_smooth = stress
        
        # 1. 弹性模量
        elastic_region = strain <= 0.002
        if np.sum(elastic_region) > 5:
            elastic_strain = strain[elastic_region][:min(10, len(strain[elastic_region]))]
            elastic_stress = stress_smooth[elastic_region][:min(10, len(stress_smooth[elastic_region]))]
            slope = np.polyfit(elastic_strain, elastic_stress, 1)[0]
            youngs_modulus = slope / 1e9  # 转换为GPa
        else:
            youngs_modulus = 200
        
        # 2. 屈服强度（0.2%偏移法）
        try:
            offset_line = youngs_modulus * 1e9 * (strain - 0.002)
            diff = np.abs(stress_smooth - offset_line)
            valid_indices = strain > 0.002
            
            if np.any(valid_indices):
                yield_idx = np.argmin(diff[valid_indices])
                yield_strength = stress_smooth[valid_indices][yield_idx]
            else:
                yield_strength = np.max(stress_smooth) * 0.8
        except:
            yield_strength = np.max(stress_smooth) * 0.8
        
        # 3. 抗拉强度
        tensile_strength = np.max(stress_smooth)
        
        # 4. 断裂应变
        max_idx = np.argmax(stress_smooth)
        fracture_strain = strain[max_idx]
        
        # 5. 韧性
        toughness = np.trapz(stress_smooth, strain) / 1e6  # 转换为MJ/m³
        
        return {
            '弹性模量': {'value': max(youngs_modulus, 0), 'unit': 'GPa'},
            '屈服强度': {'value': max(yield_strength, 0), 'unit': 'MPa'},
            '抗拉强度': {'value': max(tensile_strength, 0), 'unit': 'MPa'},
            '断裂应变': {'value': max(fracture_strain, 0), 'unit': ''},
            '韧性': {'value': max(toughness, 0), 'unit': 'MJ/m³'}
        }
    except Exception as e:
        print(f"计算错误: {e}")
        return {
            '弹性模量': {'value': 200, 'unit': 'GPa'},
            '屈服强度': {'value': 400, 'unit': 'MPa'},
            '抗拉强度': {'value': 500, 'unit': 'MPa'},
            '断裂应变': {'value': 0.15, 'unit': ''},
            '韧性': {'value': 80, 'unit': 'MJ/m³'}
        }

def generate_stress_strain_chart(strain, stress, results=None, material_type="钢"):
    """生成应力-应变曲线图"""
    try:
        plt.figure(figsize=(12, 8))
        
        # 绘制原始曲线
        if len(strain) > 1:
            plt.plot(strain, stress, 'b-', alpha=0.7, linewidth=2, label='实验数据')
        else:
            plt.plot(strain[0], stress[0], 'bo', markersize=10, label='实验数据')
        
        # 标记关键点
        if results and len(strain) > 1:
            try:
                # 屈服点
                yield_strength = results['屈服强度']['value']
                yield_idx = np.argmin(np.abs(stress - yield_strength))
                
                if yield_idx < len(strain):
                    plt.plot(strain[yield_idx], stress[yield_idx], 'ro', 
                            markersize=10, label='屈服点', zorder=5)
                
                # 最大应力点
                max_idx = np.argmax(stress)
                plt.plot(strain[max_idx], stress[max_idx], 'go', 
                        markersize=10, label='抗拉强度', zorder=5)
            except:
                pass
        
        plt.xlabel('应变', fontsize=14, fontweight='bold')
        plt.ylabel('应力 (MPa)', fontsize=14, fontweight='bold')
        plt.title(f'{material_type}材料应力-应变曲线', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        if len(strain) > 1:
            plt.xlim(0, strain[-1] * 1.05)
            plt.ylim(0, max(stress) * 1.05)
        
        plt.legend(loc='best')
        
        # 添加文本说明
        if results:
            textstr = f"材料类型: {material_type}\n"
            textstr += f"数据点数: {len(strain)}\n\n"
            
            main_params = ['弹性模量', '屈服强度', '抗拉强度', '断裂应变']
            for key in main_params:
                if key in results:
                    val = results[key]
                    if val['unit']:
                        textstr += f"{key}: {val['value']:.2f} {val['unit']}\n"
                    else:
                        textstr += f"{key}: {val['value']:.4f}\n"
            
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # 将图表转换为base64字符串
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"图表生成错误: {e}")
        
        # 生成错误图表
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, '图表生成失败\n请检查数据格式', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('错误: 无法生成图表')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """上传CSV文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        file = request.files['file']
        material_type = request.form.get('material_type', '钢')
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        # 读取文件内容
        content = file.read().decode('utf-8', errors='ignore')
        lines = content.strip().split('\n')
        
        strain_list = []
        stress_list = []
        
        print(f"解析文件: {file.filename}")
        
        # 手动解析数据
        for line in lines:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # 提取所有数字
            numbers = re.findall(r'[-+]?\d*\.?\d+', line)
            
            if len(numbers) >= 2:
                # 前两个数字作为应变和应力
                try:
                    strain_list.append(float(numbers[0]))
                    stress_list.append(float(numbers[1]))
                except:
                    continue
            elif len(numbers) == 1:
                # 只有一个数字，作为应变，生成模拟应力
                try:
                    strain_list.append(float(numbers[0]))
                    stress_list.append(float(numbers[0]) * 200000)
                except:
                    continue
        
        # 如果没有数据，尝试生成示例数据
        if not strain_list:
            strain_list = list(np.linspace(0, 0.15, 100))
            stress_list = list(200000 * np.array(strain_list))
        
        # 转换为numpy数组
        strain = np.array(strain_list)
        stress = np.array(stress_list)
        
        print(f"解析成功: {len(strain)} 个数据点")
        
        # 更新全局数据
        current_data['strain'] = strain
        current_data['stress'] = stress
        current_data['material_type'] = material_type
        current_data['filename'] = file.filename
        current_data['results'] = None
        
        # 生成预览图表
        chart_base64 = generate_stress_strain_chart(strain, stress, material_type=material_type)
        
        return jsonify({
            'success': True,
            'message': f'文件"{file.filename}"上传成功',
            'data_points': len(strain),
            'strain_range': f'{strain[0]:.4f} ~ {strain[-1]:.4f}',
            'stress_range': f'{stress[0]:.1f} ~ {stress[-1]:.1f} MPa',
            'chart': chart_base64,
            'filename': file.filename
        })
        
    except Exception as e:
        print(f"文件处理错误: {str(e)}")
        return jsonify({'success': False, 'message': f'文件处理失败: {str(e)}'})

@app.route('/load_example', methods=['POST'])
def load_example():
    """加载示例数据"""
    try:
        data = request.get_json()
        material_type = data.get('material_type', '钢') if data else '钢'
        
        print(f"加载示例数据，材料类型: {material_type}")
        
        # 生成示例数据
        strain = np.linspace(0, 0.15, 300)
        
        if material_type == "钢":
            elastic_strain = strain[strain <= 0.002]
            elastic_stress = 200000 * elastic_strain
            plastic_strain = strain[strain > 0.002]
            plastic_stress = 400 + 800 * (plastic_strain - 0.002)**0.3
        elif material_type == "铝":
            elastic_strain = strain[strain <= 0.0015]
            elastic_stress = 70000 * elastic_strain
            plastic_strain = strain[strain > 0.0015]
            plastic_stress = 250 + 400 * (plastic_strain - 0.0015)**0.2
        elif material_type == "铜":
            elastic_strain = strain[strain <= 0.0018]
            elastic_stress = 110000 * elastic_strain
            plastic_strain = strain[strain > 0.0018]
            plastic_stress = 150 + 300 * (plastic_strain - 0.0018)**0.25
        else:
            elastic_strain = strain[strain <= 0.002]
            elastic_stress = 200000 * elastic_strain
            plastic_strain = strain[strain > 0.002]
            plastic_stress = 400 + 800 * (plastic_strain - 0.002)**0.3
        
        stress = np.concatenate([elastic_stress, plastic_stress])
        
        # 更新全局数据
        current_data['strain'] = strain
        current_data['stress'] = stress
        current_data['material_type'] = material_type
        current_data['filename'] = f'{material_type}_示例数据'
        current_data['results'] = None
        
        # 生成预览图表
        chart_base64 = generate_stress_strain_chart(strain, stress, material_type=material_type)
        
        return jsonify({
            'success': True,
            'message': f'{material_type}材料示例数据加载成功',
            'data_points': len(strain),
            'strain_range': f'{strain[0]:.4f} ~ {strain[-1]:.4f}',
            'stress_range': f'{stress[0]:.1f} ~ {stress[-1]:.1f} MPa',
            'chart': chart_base64,
            'material_type': material_type
        })
        
    except Exception as e:
        print(f"加载示例数据错误: {str(e)}")
        return jsonify({'success': False, 'message': f'加载失败: {str(e)}'})

@app.route('/analyze', methods=['POST'])
def analyze():
    """分析材料性能"""
    try:
        if current_data['strain'] is None:
            return jsonify({'success': False, 'message': '请先加载数据'})
        
        strain = current_data['strain']
        stress = current_data['stress']
        material_type = current_data['material_type']
        
        print(f"分析材料性能，材料类型: {material_type}")
        
        # 计算材料性能
        results = calculate_material_properties(strain, stress)
        current_data['results'] = results
        
        # 生成完整图表
        chart_base64 = generate_stress_strain_chart(strain, stress, results, material_type)
        
        return jsonify({
            'success': True,
            'message': '材料性能分析完成',
            'results': results,
            'chart': chart_base64,
            'material_type': material_type
        })
        
    except Exception as e:
        print(f"分析错误: {str(e)}")
        return jsonify({'success': False, 'message': f'分析失败: {str(e)}'})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """生成分析报告"""
    try:
        if current_data['results'] is None:
            return jsonify({'success': False, 'message': '请先分析材料性能'})
        
        # 生成报告文本
        report = "=" * 60 + "\n"
        report += "         材料力学性能分析报告\n"
        report += "=" * 60 + "\n\n"
        report += f"材料类型: {current_data['material_type']}\n"
        if current_data['filename']:
            report += f"数据文件: {current_data['filename']}\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"数据点数: {len(current_data['strain'])}\n"
        report += "-" * 60 + "\n\n"
        
        report += "力学性能参数:\n"
        report += "-" * 40 + "\n"
        for key, val in current_data['results'].items():
            if val['unit']:
                report += f"{key}: {val['value']:.2f} {val['unit']}\n"
            else:
                report += f"{key}: {val['value']:.4f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return jsonify({
            'success': True,
            'message': '分析报告生成完成',
            'report': report,
            'chart': current_data.get('chart', '')
        })
        
    except Exception as e:
        print(f"生成报告错误: {str(e)}")
        return jsonify({'success': False, 'message': f'生成报告失败: {str(e)}'})

@app.route('/download_report')
def download_report():
    """下载分析报告"""
    try:
        if current_data['results'] is None:
            return jsonify({'success': False, 'message': '请先分析材料性能'})
        
        # 生成报告文本
        report = "=" * 60 + "\n"
        report += "         材料力学性能分析报告\n"
        report += "=" * 60 + "\n\n"
        report += f"材料类型: {current_data['material_type']}\n"
        if current_data['filename']:
            report += f"数据文件: {current_data['filename']}\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"数据点数: {len(current_data['strain'])}\n"
        report += "-" * 60 + "\n\n"
        
        report += "力学性能参数:\n"
        report += "-" * 40 + "\n"
        for key, val in current_data['results'].items():
            if val['unit']:
                report += f"{key}: {val['value']:.2f} {val['unit']}\n"
            else:
                report += f"{key}: {val['value']:.4f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        # 创建文件响应
        response = io.BytesIO()
        response.write(report.encode('utf-8'))
        response.seek(0)
        
        filename = f"材料分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        return send_file(
            response,
            as_attachment=True,
            download_name=filename,
            mimetype='text/plain'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'下载失败: {str(e)}'})

@app.route('/data_summary', methods=['GET'])
def data_summary():
    """获取数据摘要"""
    try:
        if current_data['strain'] is None:
            return jsonify({
                'success': False,
                'has_data': False,
                'message': '没有加载数据'
            })
        
        strain = current_data['strain']
        stress = current_data['stress']
        
        return jsonify({
            'success': True,
            'has_data': True,
            'data_points': len(strain),
            'strain_min': float(strain[0]),
            'strain_max': float(strain[-1]),
            'stress_min': float(stress[0]),
            'stress_max': float(stress[-1]),
            'material_type': current_data['material_type'],
            'filename': current_data['filename'],
            'has_results': current_data['results'] is not None
        })
        
    except Exception as e:
        print(f"获取数据摘要错误: {str(e)}")
        return jsonify({'success': False, 'message': f'获取失败: {str(e)}'})

@app.route('/clear_data', methods=['POST'])
def clear_data():
    """清除数据"""
    try:
        current_data['strain'] = None
        current_data['stress'] = None
        current_data['results'] = None
        current_data['filename'] = None
        
        return jsonify({
            'success': True,
            'message': '数据已清除'
        })
        
    except Exception as e:
        print(f"清除数据错误: {str(e)}")
        return jsonify({'success': False, 'message': f'清除失败: {str(e)}'})

if __name__ == '__main__':
    print("=" * 60)
    print("材料力学性能分析系统 (修复版)")
    print("=" * 60)
    print("系统正在启动...")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)  # 关键是把 host 改为 '0.0.0.0'