from flask import Flask, render_template, request, jsonify
from prediction_program import InBodyCountPredictor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

app = Flask(__name__)
predictor = InBodyCountPredictor()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/current', methods=['GET'])
def get_current_data():
    """현재 데이터 조회"""
    try:
        df = predictor.load_historical_data()
        if len(df) == 0:
            return jsonify({
                'success': True,
                'current_count': 0,
                'progress': 0,
                'remaining': 200000000,
                'has_data': False
            })
        
        latest = df.iloc[-1]
        current_count = int(latest['누적 건수'])
        target = 200000000
        progress = (current_count / target) * 100
        remaining = target - current_count
        
        return jsonify({
            'success': True,
            'current_count': current_count,
            'progress': round(progress, 2),
            'remaining': remaining,
            'has_data': True,
            'last_update': f"{latest['날짜']} {latest['시간']}"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/add', methods=['POST'])
def add_data():
    """데이터 추가"""
    try:
        data = request.get_json()
        total_count = int(data.get('count', 0))
        
        if total_count <= 0:
            return jsonify({'success': False, 'error': '0보다 큰 숫자를 입력해주세요.'}), 400
        
        if total_count > 500000000:
            return jsonify({'success': False, 'error': '입력값이 너무 큽니다.'}), 400
        
        # 이전 데이터와 비교하여 감소하는지 확인
        df = predictor.load_historical_data()
        if len(df) > 0:
            latest_count = df.iloc[-1]["누적 건수"]
            if total_count < latest_count:
                return jsonify({
                    'success': False, 
                    'error': '입력한 값이 이전 값보다 작습니다.',
                    'previous_count': int(latest_count)
                }), 400
        
        predictor.add_manual_data(total_count)
        
        return jsonify({
            'success': True,
            'message': '데이터가 저장되었습니다.',
            'count': total_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['GET'])
def predict():
    """예측 계산"""
    try:
        df = predictor.load_historical_data()
        
        if len(df) < 2:
            return jsonify({
                'success': False,
                'error': '예측을 위해서는 최소 2개 이상의 데이터 포인트가 필요합니다.'
            }), 400
        
        # 데이터 정렬
        df = df.sort_values(by=["날짜", "시간"]).reset_index(drop=True)
        df["날짜시간"] = pd.to_datetime(df["날짜"] + ' ' + df["시간"])
        
        # 2025-11-17 14:30:40 이전 데이터 필터링
        cutoff_time = pd.to_datetime("2025-11-17 14:30:40")
        df = df[df["날짜시간"] >= cutoff_time].copy()
        
        if len(df) < 2:
            return jsonify({
                'success': False,
                'error': '2025-11-17 14:30:40 이후 데이터가 부족합니다.'
            }), 400
        
        # 감소하는 데이터 필터링
        df["이전 대비 증가"] = df["누적 건수"].diff()
        df_filtered = df[(df["이전 대비 증가"].isna()) | (df["이전 대비 증가"] >= 0)].copy()
        
        if len(df_filtered) < 2:
            return jsonify({
                'success': False,
                'error': '증가하는 데이터가 부족합니다.'
            }), 400
        
        df_filtered["시간 차이(분)"] = df_filtered["날짜시간"].diff().dt.total_seconds() / 60
        df_filtered["분당 증가량"] = df_filtered["이전 대비 증가"] / df_filtered["시간 차이(분)"]
        
        recent_df = df_filtered.tail(min(5, len(df_filtered)))
        valid_increase_rates = recent_df["분당 증가량"].dropna()
        
        if len(valid_increase_rates) == 0:
            return jsonify({'success': False, 'error': '유효한 증가율 데이터가 없습니다.'}), 400
        
        avg_per_min = valid_increase_rates.mean()
        latest_count = df_filtered.iloc[-1]["누적 건수"]
        latest_time = df_filtered.iloc[-1]["날짜시간"]
        target = 200000000
        remaining = target - latest_count
        
        if avg_per_min <= 0:
            return jsonify({'success': False, 'error': '증가율이 0 이하입니다.'}), 400
        
        # 선형 회귀 계산
        dates_numeric = df_filtered["날짜시간"]
        counts_numeric = df_filtered["누적 건수"].values
        time_numeric = np.array([(d - dates_numeric.iloc[0]).total_seconds() / 60 for d in dates_numeric])
        
        coeffs = np.polyfit(time_numeric, counts_numeric, 1)
        slope_per_min = float(coeffs[0])
        intercept = float(coeffs[1])
        
        # 선형식 기반 예측
        minutes_needed_linear = (target - intercept) / slope_per_min
        predicted_time_linear = dates_numeric.iloc[0] + timedelta(minutes=minutes_needed_linear)
        
        # 평균 증가율 기반 예측
        minutes_needed = remaining / avg_per_min
        predicted_time = latest_time + timedelta(minutes=minutes_needed)
        
        # 시간대별 분석
        df_filtered["시간대"] = df_filtered["날짜시간"].dt.hour.apply(lambda h: "오전" if h < 12 else "오후")
        df_filtered["시간대_세부"] = df_filtered["날짜시간"].dt.hour.apply(lambda h: 
            "새벽(0-5시)" if h < 6 else 
            "오전(6-11시)" if h < 12 else 
            "오후(12-17시)" if h < 18 else 
            "저녁(18-23시)")
        
        time_period_stats = {}
        for period in ["오전", "오후"]:
            period_df = df_filtered[df_filtered["시간대"] == period]
            if len(period_df) > 1:
                period_rates = period_df["분당 증가량"].dropna()
                if len(period_rates) > 0:
                    time_period_stats[period] = {
                        "평균": float(period_rates.mean()),
                        "데이터수": int(len(period_rates)),
                        "최소": float(period_rates.min()),
                        "최대": float(period_rates.max())
                    }
        
        detailed_stats = {}
        for period in ["새벽(0-5시)", "오전(6-11시)", "오후(12-17시)", "저녁(18-23시)"]:
            period_df = df_filtered[df_filtered["시간대_세부"] == period]
            if len(period_df) > 1:
                period_rates = period_df["분당 증가량"].dropna()
                if len(period_rates) > 0:
                    detailed_stats[period] = {
                        "평균": float(period_rates.mean()),
                        "데이터수": int(len(period_rates)),
                        "최소": float(period_rates.min()),
                        "최대": float(period_rates.max())
                    }
        
        # 데이터 테이블 준비
        display_df = df_filtered[["날짜", "시간", "누적 건수", "이전 대비 증가", "분당 증가량"]].copy()
        display_df["이전 대비 증가"] = display_df["이전 대비 증가"].fillna(0).astype(int)
        display_df["분당 증가량"] = display_df["분당 증가량"].fillna(0)
        
        data_table = []
        for _, row in display_df.iterrows():
            data_table.append({
                '날짜': str(row['날짜']),
                '시간': str(row['시간']),
                '누적건수': int(row['누적 건수']),
                '이전대비증가': int(row['이전 대비 증가']),
                '분당증가량': round(float(row['분당 증가량']), 2)
            })
        
        return jsonify({
            'success': True,
            'current_count': int(latest_count),
            'remaining': int(remaining),
            'avg_per_min': round(avg_per_min, 2),
            'slope_per_min': round(slope_per_min, 2),
            'intercept': int(intercept),
            'predicted_time_avg': predicted_time.strftime('%Y-%m-%d, %H:%M:%S'),
            'predicted_time_linear': predicted_time_linear.strftime('%Y-%m-%d, %H:%M:%S'),
            'days_remaining': round((predicted_time_linear - datetime.now()).total_seconds() / (60 * 60 * 24), 1),
            'time_period_stats': time_period_stats,
            'detailed_stats': detailed_stats,
            'data_table': data_table,
            'filtered_count': len(df) - len(df_filtered)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # 로컬 실행용
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

