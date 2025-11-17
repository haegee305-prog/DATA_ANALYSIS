import pandas as pd
from datetime import datetime, timedelta
import json
import os


class InBodyCountPredictor:
    def __init__(self):
        self.data_file = "InBodyDATA/InBodydata_Analysis/Data"
            
            
    def load_historical_data(self):
        """저장된 과거 데이터 로드"""
        if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        return pd.DataFrame(data)
            except Exception as e:
                print(f"데이터 로드 중 오류: {e}")
        return pd.DataFrame(columns=["날짜", "시간", "누적 건수", "세부 데이터"])
        
    def save_data(self, df):
        """데이터 저장"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        df_dict = df.to_dict('records')
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(df_dict, f, ensure_ascii=False, indent=2)
            
    def add_manual_data(self, total_count):
        """사용자가 직접 입력한 데이터 추가 (시간은 자동 기록)"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        new_row = {
            "날짜": date_str,
            "시간": time_str,
            "누적 건수": total_count,
            "세부 데이터": {}
        }
        
        df = self.load_historical_data()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_data(df)
        
        return total_count
        
    def predict_200m_reach_time(self):
        """2억 도달 시각 예측"""
        df = self.load_historical_data()
        
        if len(df) < 2:
            print("예측을 위해서는 최소 2개 이상의 데이터 포인트가 필요합니다.")
            return None
            
        # 데이터 정렬 (시간순)
        df = df.sort_values(by=["날짜", "시간"]).reset_index(drop=True)
        
        # 이전 대비 증가 계산
        df["이전 대비 증가"] = df["누적 건수"].diff()
        
        # 분 단위 증가량 계산
        df["날짜시간"] = pd.to_datetime(df["날짜"] + ' ' + df["시간"])
        df["시간 차이(분)"] = df["날짜시간"].diff().dt.total_seconds() / 60
        
        # 분당 증가량 계산 (마지막 데이터 포인트 기준)
        df["분당 증가량"] = df["이전 대비 증가"] / df["시간 차이(분)"]
        
        # 최근 데이터만 사용 (최근 5개 포인트의 평균 증가율 사용)
        recent_df = df.tail(min(5, len(df)))
        valid_increase_rates = recent_df["분당 증가량"].dropna()
        
        if len(valid_increase_rates) == 0:
            print("유효한 증가율 데이터가 없습니다.")
            return None
            
        # 평균 분당 증가량 계산
        avg_per_min = valid_increase_rates.mean()
        
        # 2억 도달 예상 시각 계산
        latest_count = df.iloc[-1]["누적 건수"]
        latest_time = df.iloc[-1]["날짜시간"]
        
        remaining = 200000000 - latest_count
        
        if avg_per_min <= 0:
            print("증가율이 0 이하입니다. 예측할 수 없습니다.")
            return None
            
        minutes_needed = remaining / avg_per_min
        predicted_time = latest_time + timedelta(minutes=minutes_needed)
        
        # 결과 출력
        print("\n=== 데이터 분석 결과 ===")
        print(df[["날짜", "시간", "누적 건수", "이전 대비 증가", "분당 증가량"]].to_string(index=False))
        print(f"\n현재 누적 건수: {latest_count:,}")
        print(f"목표: 200,000,000")
        print(f"남은 건수: {remaining:,}")
        print(f"평균 분당 증가량: {avg_per_min:.2f}")
        print(f"예상 소요 시간: {minutes_needed:.1f}분 ({minutes_needed/60:.2f}시간)")
        
        # 요청한 형식으로 출력: "2025-00-00, 16:00:00" (24시간 형식)
        formatted_time = predicted_time.strftime("%Y-%m-%d, %H:%M:%S")
        print(f"\n예상 2억 도달 시각: {formatted_time}")
        
        return predicted_time

