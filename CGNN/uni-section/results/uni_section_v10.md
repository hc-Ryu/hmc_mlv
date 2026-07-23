dmf     uni_section_v10: command_v10.md 반영 (synod-20260710-103702-cee330 review / synod-20260710-105249-9ab67a design)
      - §1 T_MAX 2.5mm, DELTA_SCALE 1.35, bias 0.03 (두께 인플레이션 차단 + 재보정)
      - §2 collision v5: surface-gap 제곱 소프트컨택, t detach 금지, alpha 삭제, 전쌍 검사, 부호 앵커
      - §3 2D mesh order loss (w_order=1.0 상시) — v8 붕괴(criss-crossing) 근본 대책
      - §4 2단계 두께 학습: sigmoid gate@128 + thickness 그룹 국소 optimizer 리셋 + lr 0.3x
      - §5 비대칭 Huber phys loss (delta=0.05, undershoot 2.0x)
    
    데이터: nodes=torch.Size([93, 8]) | edges=torch.Size([2, 176])
    [gradcheck] dMp/dt OK -- autograd=1.3173e+07, FD=1.3160e+07, rel_err=9.88e-04, dMp/dt>0 확인
    [mass] target_area = 1157.3 mm² (초기 단면적 스냅샷)
    [collision v5] 전쌍(9쌍) 부호 앵커 & clearance (초기 형상 기준):
        sec0 Part0-Part1: 0seg/1pt σ=+1 clr=-0.05 | 1seg/0pt σ=-1 clr=-0.05
        sec0 Part0-Part2: 0seg/2pt σ=+1 clr=0.50
        sec0 Part0-Part3: 0seg/3pt σ=+1 clr=0.50 | 3seg/0pt σ=-1 clr=0.50
        sec0 Part0-Part4: 0seg/4pt σ=+1 clr=0.50 | 4seg/0pt σ=-1 clr=0.50
        sec0 Part1-Part2: 1seg/2pt σ=-1 clr=-0.05
        sec0 Part1-Part3: 1seg/3pt σ=-1 clr=-0.62 | 3seg/1pt σ=+1 clr=-0.62
        sec0 Part1-Part4: 1seg/4pt σ=+1 clr=-1.05 | 4seg/1pt σ=-1 clr=-1.05
        sec0 Part2-Part3: 2seg/3pt σ=+1 clr=0.50 | 3seg/2pt σ=-1 clr=0.50
        sec0 Part2-Part4: 2seg/4pt σ=+1 clr=0.50 | 4seg/2pt σ=-1 clr=0.50
    
    ==============================================================================
    [ uni_section_v10 ] Training  |  Target Mp = 47,421,470 N·mm  |  Epochs: 300
      CGDN: hidden=128, layers=4, heads=4  |  Curriculum: True (0.2, 0.7)
      (§2 collision v5 surface-gap² / §3 mesh order / §4 2-stage gate@128 / §5 asym Huber)
      T_MAX=2.5mm | DELTA_SCALE=1.35 | w_order=1.0 | grad_clip=5.0
      Feasibility 기준: Mp err < 2.0%  AND  l_collision < 0.05
    ==============================================================================
    Epoch ||  Loss  ||  MpErr% |  Smth  |  Area  | Mass(gate) |  Coll  | Order  || tGate | dT | satParts
    00000 || 1.7893 ||  47.17% | 0.3333 | 1162.9 | 0.0000(0.00) | 0.0000 | 0.0000 || 0.00 | +0.00 | 0/5
    00019 || 1.6696 ||  40.10% | 0.3265 | 1190.2 | 0.0008(0.00) | 0.0000 | 0.0000 || 0.00 | -0.00 | 0/5
    00039 || 1.6411 ||  38.39% | 0.3588 | 1211.8 | 0.0022(0.00) | 0.0000 | 0.0000 || 0.00 | -0.00 | 0/5
    00059 || 1.4816 ||  33.59% | 0.3379 | 1288.5 | 0.0128(0.00) | 0.0002 | 0.0000 || 0.00 | -0.00 | 0/5
    00079 || 1.3733 ||  17.77% | 0.7249 | 1504.7 | 0.0901(0.00) | 0.0032 | 0.0000 || 0.00 | +0.00 | 0/5
    00099 || 1.1916 ||   1.06% | 0.7224 | 1692.2 | 0.2136(0.00) | 0.0048 | 0.0000 || 0.00 | -0.00 | 0/5 [FEASIBLE]
    00119 || 1.6814 ||   0.68% | 0.9015 | 1562.8 | 0.1227(0.00) | 0.0048 | 0.0000 || 0.00 | +0.01 | 0/5 [FEASIBLE]
    [Stage 2] epoch 128: thickness_decoder 그룹 AdamW state 리셋, lr → 3.0e-04 (좌표 헤드 모멘텀 보존)
    00139 || 1.6681 ||   4.69% | 0.6906 | 1754.1 | 0.2659(0.51) | 0.0970 | 0.0000 || 1.00 | +1.76 | 5/5
    00159 || 1.6158 ||   3.80% | 0.5945 | 1740.8 | 0.2542(0.53) | 0.0816 | 0.0000 || 1.00 | +1.54 | 5/5
    00179 || 1.6949 ||   4.21% | 0.5432 | 1728.7 | 0.2437(0.52) | 0.0717 | 0.0000 || 1.00 | +1.44 | 5/5
    00199 || 1.3683 ||   1.99% | 0.5096 | 1738.1 | 0.2518(0.57) | 0.0181 | 0.0000 || 1.00 | +1.29 | 3/5 [FEASIBLE]
    00219 || 1.8362 ||   6.85% | 0.4866 | 1660.8 | 0.1893(0.45) | 0.0138 | 0.0000 || 1.00 | +1.31 | 4/5
    00239 || 1.3657 ||   3.75% | 0.4052 | 1703.0 | 0.2223(0.53) | 0.0177 | 0.0000 || 1.00 | +1.37 | 4/5
    00259 || 1.2574 ||   0.05% | 0.4675 | 1720.0 | 0.2364(0.62) | 0.0144 | 0.0000 || 1.00 | +1.39 | 4/5 [FEASIBLE]
    00279 || 1.1783 ||   0.01% | 0.3632 | 1711.8 | 0.2296(0.62) | 0.0154 | 0.0000 || 1.00 | +1.40 | 4/5 [FEASIBLE]
    00299 || 1.2168 ||   0.03% | 0.4649 | 1713.0 | 0.2305(0.62) | 0.0159 | 0.0000 || 1.00 | +1.36 | 4/5 [FEASIBLE]
    
    ──────────────────────────────────────────────────────────────────────────────
    [Feasibility] 첫 만족 epoch: 74  |  최고(Mp err 최소) epoch: 111  |  Mp err: 0.01%  |  l_collision: 0.0109
    [Best-Mp] epoch 111: Mp err = 0.01%, l_collision = 0.0109  (collision 무관 추적)
    ──────────────────────────────────────────────────────────────────────────────
    


    
![png](uni_section_v10_files/uni_section_v10_1_1.png)
    


    
    결과 저장: c:\Users\user\Documents\GitHub\hmc_mlv\CGNN\uni-section\uni_section_v10_result.png
    
    ==============================================================================
    최종 결과 요약 (마지막 epoch 기준)
      Target Mp     :     47,421,470 N·mm
      Final pred_mp :     47,435,896 N·mm
      Final Error   :   0.03%
      Final l_collision : 0.0159
      Final l_order : 0.0000
      Final delta_t : mean=+1.356 mm
    
      ★ Feasible 모델(물리 제약 + Mp 동시 만족) 발견: epoch 111, Mp err=0.01%, l_collision=0.0109
        (best_feasible['state_dict']를 model.load_state_dict()로 복원해 사용 권장)
    ==============================================================================
    
