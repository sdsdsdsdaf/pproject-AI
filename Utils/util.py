import ast
import numpy as np


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