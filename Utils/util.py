import ast
import numpy as np

def print_dict_structure(d, indent=0):
    """Print dictionary structure without expanding large values."""
    prefix = "  " * indent

    if not isinstance(d, dict):
        print(f"{prefix}- {type(d).__name__}")
        return

    for key, value in d.items():
        key_str = str(key)
        if len(key_str) > 60:  # 너무 긴 key는 잘라줌
            key_str = key_str[:60] + "..."

        # value 타입만 출력
        if isinstance(value, dict):
            print(f"{prefix}{key_str}/   (dict)")
            print_dict_structure(value, indent + 1)
        else:
            print(f"{prefix}{key_str}: {type(value).__name__}")


def safe_to_list(x):
    """
    문자열 → list 변환
    numpy array → list 변환
    이미 list / tuple ⇒ 그대로
    그 외 모든 타입 ⇒ 빈 list 반환
    """

    # NaN, None 처리
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []

    # 문자열 처리
    if isinstance(x, str):
        try:
            obj = ast.literal_eval(x)
            # dict 한 개가 들어있다 → 리스트로 감싸기
            if isinstance(obj, dict):
                return [obj]
            # numpy array → list
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # list면 그대로
            if isinstance(obj, list):
                return obj
            return []
        except Exception:
            return []

    # numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    # 목록 타입
    if isinstance(x, (list, tuple)):
        return list(x)

    # dict 단일 객체
    if isinstance(x, dict):
        return [x]

    # 나머지 타입 → 처리 불가
    return []