import re
import pickle

# 로그 파일을 읽어서 top_k_paths 부분을 추출하는 함수
def parse_log_file(log_file_path):
    # 정규 표현식으로 top_k_paths 추출
    top_k_pattern = re.compile(r"top_k_paths\s*:\s*(\[\(.*?\)\])")
    
    config_list = []
    id = 0

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = top_k_pattern.search(line)
            if match:
                top_k_str = match.group(1)
                # eval을 사용하여 문자열을 실제 리스트로 변환
                top_k_paths = eval(top_k_str)

                for item in top_k_paths:
                    loss = item[0]
                    config = item[1]
                    mlp_ratio = config['mlp_ratio']
                    num_heads = config['num_heads']
                    embed_dim = config['embed_dim']
                    layer_num = config['layer_num']

                    # 각 item의 정보 (loss, mlp_ratio, num_heads, embed_dim, layer_num) 추가
                    config_list.append({
                        'loss': loss,
                        'mlp_ratio': mlp_ratio,
                        'num_heads': num_heads,
                        'embed_dim': embed_dim,
                        'layer_num': layer_num,
                        'id': id
                    })
                    id += 1

    return config_list

# 파싱한 config 리스트를 pkl로 저장하는 함수
def save_config_list_to_pkl(config_list, output_pkl_path):
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(config_list, f)
    print(f"Config list saved to {output_pkl_path}")

# 실행 부분
if __name__ == "__main__":
    log_file_path = './greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(dss)-20epoch.log'  # 로그 파일 경로
    output_pkl_path = './greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(dss)-20epoch.pkl'  # 저장할 pkl 파일 경로

    config_list = parse_log_file(log_file_path)
    save_config_list_to_pkl(config_list, output_pkl_path)
    print(len(config_list))
    print(config_list[100])
