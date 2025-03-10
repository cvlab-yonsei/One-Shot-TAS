import re
import pickle

# a.pkl 파일을 불러오는 함수
def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# loss 값을 추가하는 함수
def add_inter_loss(results, a_pkl_data):
    a_pkl_dict = {}
    
    # a.pkl 데이터를 키로 저장 (layer_num, mlp_ratio, num_heads, embed_dim 기준)
    for item in a_pkl_data:
        key = (item['layer_num'], tuple(item['mlp_ratio']), tuple(item['num_heads']), tuple(item['embed_dim']))
        a_pkl_dict[key] = item['loss']

    # results_no_duplicates에 inter_loss 추가
    for config in results:
        key = (config['layer_num'], tuple(config['mlp_ratio']), tuple(config['num_heads']), tuple(config['embed_dim']))
        if key in a_pkl_dict:
            config['inter_loss'] = a_pkl_dict[key]
    
    return results

def find_non_matching_pairs(results):
    non_matching_indices = []

    # results의 길이가 짝수인지 확인
    if len(results) % 2 != 0:
        print("Warning: The number of items in results should be even.")
        return non_matching_indices

    # 각 n과 n+1 쌍을 검사
    for i in range(0, len(results) - 1, 2):  # 2씩 증가시키면서 인덱스 접근
        if results[i] != results[i + 1]:
            non_matching_indices.append(i)  # n 인덱스 추가 (n, n+1 쌍이 일치하지 않음)

    return non_matching_indices

def remove_duplicates(results):
    unique_results = []
    seen = set()

    for config in results:
        # 중복을 검사할 키 생성 (parameters 제외)
        key = (config['layer_num'], tuple(config['mlp_ratio']), tuple(config['num_heads']), tuple(config['embed_dim']))

        if key not in seen:
            # 처음 본 키라면 저장
            seen.add(key)
            unique_results.append(config)

    return unique_results


def parse_evolution_log(file_path):
    results = []
    current_config = {}
    i = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # 모델 설정 정보 추출
            if line.startswith("sampled model config:"):
                config_dict_str = line.split("sampled model config: ")[1]
                current_config = eval(config_dict_str)  # 문자열을 딕셔너리로 변환
            
            # 모델 파라미터 정보 추출
            elif line.startswith("sampled model parameters:"):
                parameters = int(line.split("sampled model parameters: ")[1])
                current_config['parameters'] = parameters
            
            # 성능 지표 정보 추출
            elif line.startswith("* Acc@1"):
                acc1 = float(re.search(r"Acc@1\s(\d+\.\d+)", line).group(1))
                acc5 = float(re.search(r"Acc@5\s(\d+\.\d+)", line).group(1))
                loss = float(re.search(r"loss\s(\d+\.\d+)", line).group(1))
                current_config['acc1'] = acc1
                current_config['acc5'] = acc5
                current_config['loss'] = loss
                # current_config['id'] = int(i/2)
                current_config['id'] = int(i)
                
                # 모든 정보가 취합된 딕셔너리를 결과 리스트에 추가
                results.append(current_config)
                current_config = {}  # 다음 세트를 위해 초기화
                i = i + 1

    return results

# 파일 경로 사용 예
# file_path = "./greedyTAS/greedyTAS-epoch60/autoformer-greedyTAS(09131747).log"

# file_path = "./greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(dss)-20epoch-subnet.log"
file_path = "./log/search_tiny-only-supernet192-minimum_pop1050.log"

results = parse_evolution_log(file_path)
print(len(results))  # 결과 출력

# non_matching_indices = find_non_matching_pairs(results)
# print("Non-matching indices:", non_matching_indices)

results_no_duplicates = remove_duplicates(results)
# results_no_duplicates = results
print(len(results_no_duplicates))  # 중복 제거된 결과 출력
print(results_no_duplicates[0])
print(results_no_duplicates[1])
print(results_no_duplicates[2])
print(results_no_duplicates[-1])

# a.pkl 파일 경로
# a_pkl_path = "./greedyTAS/greedyTAS-epoch60/autoformer-greedyTAS(09131747).log"
# a_pkl_path = "./greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(dss)-20epoch.pkl"  # 실제 파일 경로로 변경하세요

# # a.pkl 파일 로드
# a_pkl_data = load_pkl(a_pkl_path)

# # inter_loss 값을 추가
# results_with_inter_loss = add_inter_loss(results_no_duplicates, a_pkl_data)

# # 결과 출력 (예시)
# print(results_with_inter_loss[0])
# print(results_with_inter_loss[1])
# print(results_with_inter_loss[2])
# print(results_with_inter_loss[-1])

# Save the transformed data to a new pickle file
with open('./log/search_tiny-only-supernet192-minimum_pop1050.pkl', 'wb') as file:
    pickle.dump(results_no_duplicates, file)

print("Data saved successfully.")