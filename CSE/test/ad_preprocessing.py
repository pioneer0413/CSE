import csv
from collections import defaultdict

def process_data(input_file, output_file):
    # 데이터를 저장할 딕셔너리: {date: {col: data}}
    data_dict = defaultdict(dict)

    # CSV 파일 읽기
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더 건너뛰기: date,data,cols
        for row in reader:
            date, data, col = row
            date = int(date)  # date를 정수로 변환 (필요시 조정)
            data_dict[date][col] = float(data)

    # 모든 열 이름 수집 및 정렬
    all_cols = set()
    for cols in data_dict.values():
        all_cols.update(cols.keys())
    all_cols = sorted(all_cols)  # col_0, col_1, ... 순서로 정렬

    # timestamp 순서대로 정렬
    sorted_dates = sorted(data_dict.keys())

    # 출력 파일 작성
    with open(output_file, 'w') as f:
        for date in sorted_dates:
            row_data = data_dict[date]
            # 각 열에 대해 값 가져오기, 없으면 0.0 (또는 다른 기본값)
            values = [row_data.get(col, 0.0) for col in all_cols]
            # 소수점 6자리로 포맷팅
            formatted_values = [f"{v:.6f}" for v in values]
            # 콤마로 구분해서 쓰기
            f.write(','.join(formatted_values) + '\n')

# 사용 예시
if __name__ == "__main__":
    input_file = '/home/hwkang/CSE/CSE/data/SMAP.csv'  # 입력 파일 경로
    output_file = '/home/hwkang/CSE/CSE/data/smap.txt'  # 출력 파일 경로
    process_data(input_file, output_file)