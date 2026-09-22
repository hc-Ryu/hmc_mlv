# [Task] ImplicitPNASolver 인장/압축 변수명(Naming) 교정

대상파일:
1. `C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\AI_design_v1.py`
2. `C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\AI_design_v2.py`
3. `C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\AI_design_v0.py`
4. `C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\uni-section\code\uni_section_v18.py`

@Claude (Synod Agent),
현재 `AI_design_v1.py` 파일 내 `ImplicitPNASolver` 클래스에 정의된 PNA(소성 중립축) 계산 로직의 변수명(이름표)을 실제 측면 충돌 시의 물리적 거동에 맞게 교정하는 작업을 지시합니다.

## 1. 수정 사항 (변수명 스왑)
현재 코드에서는 PNA 위쪽(y > y_mid)을 인장부(F_tens), 아래쪽(y < y_mid)을 압축부(F_comp)로 명명하고 있습니다. 이를 실제 중앙부 측면 충돌 물리 법칙에 맞게 **반대로 변경**하십시오.
*   기존: F_tens = (y > y_mid) 구역 / F_comp = (y < y_mid) 구역
*   수정: F_comp = (y > y_mid) 구역 / F_tens = (y < y_mid) 구역

## 2. 수식 및 알고리즘 무결성 유지 (중요)
*   이 작업은 코드를 읽는 인간 엔지니어의 물리적 이해를 돕기 위한 **순수한 변수명(이름표) 리팩토링**입니다.
*   단면 힘 평형의 대칭성과 모멘트 거리 절대값(abs) 계산의 특성상, 변수명을 뒤집더라도 Bisection 알고리즘이 찾는 중립축(y_pna)의 위치, Mp 산출값, Backward IFT 기울기는 수학적으로 100% 동일하게 도출됩니다.
*   따라서 변수명 교체 외에 기존의 수식적 뼈대나 물리 엔진 알고리즘 로직은 조금도 변경하지 마십시오.

## 3. 엄격한 통제 사항 (Strict Rule)
*   본 지시서에서 명시한 `ImplicitPNASolver` 내부의 **인장/압축 관련 변수명 스왑 외에는 코드의 단 한 줄도 임의로 수정해서는 안 됩니다.**
*   모델 아키텍처, 데이터 파이프라인, 각종 손실 함수 등 나머지 모든 부분은 현재 완벽하게 검증 및 작동 중이므로 절대 건드리지 마십시오.