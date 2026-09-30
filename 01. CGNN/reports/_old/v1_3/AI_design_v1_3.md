    [data] 17-section 그래프(사전 필터링 없음): 노드 1581개, 엣지 2992개
    [target_mp] ['26,444,055', '29,211,881', '31,397,729', '33,087,380', '33,704,378', '32,835,017', '30,467,844', '27,734,809', '26,413,902', '27,985,091', '32,922,019', '39,716,721', '45,526,897', '47,458,156', '44,876,148', '39,505,207', '35,005,895']
    [Stage 0] Static shape/index dry-run 시작...
    [Stage 0] 분할 주입 테스트 통과 - part3이 run 2개로 독립 pooling됨 (그룹 수: 6)
    [Stage 0] GPU 메모리 사용량: 9.2 MB
    [Stage 0] 통과 - 노드 1581개, 엣지 2992개, z_gate shape (17, 5), 5파트 글로벌 두께 불변 + 분할 주입 검증 완료.
    [gradcheck] dMp/dt OK -- autograd=3.4990e+08, FD=3.4976e+08, rel_err=4.09e-04, dMp/dt>0 확인
    [mass] area_init = 24056.9 mm^2 | target_area = 22854.0 mm^2 (감량 목표 95%)
    [collision v5] 153쌍(섹션x파트쌍) 부호 앵커 산정 완료
    
    ==============================================================================
    [ AI_design_v1_1 ] ASWC Training + Dynamic Splitting | 17 sections | Section-aware gating (17x2) | Stage1<100 Stage2<400 Stage3<500
    ==============================================================================
    Epoch    0 || loss=1.1562 | MpErr=8.01% | l_continuity=0.0000(w=1.000) | area=23990.6 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch   10 || loss=13.8300 | MpErr=49.19% | l_continuity=0.0000(w=1.000) | area=11207.0 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    [Gate Active] epoch 11: gate_params lr -> 1e-2 (section-aware 게이트 학습 시작)
    Epoch   20 || loss=26.0331 | MpErr=49.35% | l_continuity=0.0000(w=1.000) | area=11174.2 | g3=0.957(17/17) g4=0.956(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch   30 || loss=13.6132 | MpErr=34.80% | l_continuity=0.0000(w=1.000) | area=13776.0 | g3=0.952(17/17) g4=0.950(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch   40 || loss=6.9570 | MpErr=23.50% | l_continuity=0.0000(w=1.000) | area=15965.8 | g3=0.949(17/17) g4=0.947(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch   50 || loss=7.0743 | MpErr=23.75% | l_continuity=0.0000(w=1.000) | area=15897.2 | g3=0.948(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch   60 || loss=1.9479 | MpErr=0.20% | l_continuity=0.0000(w=1.000) | area=21550.7 | g3=0.948(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch   70 || loss=2.5822 | MpErr=7.57% | l_continuity=0.0000(w=1.000) | area=23461.6 | g3=0.947(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch   80 || loss=1.9346 | MpErr=2.10% | l_continuity=0.0000(w=1.000) | area=20979.9 | g3=0.947(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch   90 || loss=1.9297 | MpErr=1.63% | l_continuity=0.0000(w=1.000) | area=21095.7 | g3=0.948(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  100 || loss=2.1688 | MpErr=4.61% | l_continuity=0.0000(w=1.000) | area=20975.6 | g3=0.948(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  110 || loss=1.9364 | MpErr=3.02% | l_continuity=0.0000(w=1.000) | area=20192.5 | g3=0.948(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  120 || loss=1.9588 | MpErr=0.08% | l_continuity=0.0000(w=1.000) | area=21086.9 | g3=0.948(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  130 || loss=2.0057 | MpErr=1.15% | l_continuity=0.0000(w=1.000) | area=20915.7 | g3=0.948(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  140 || loss=2.0910 | MpErr=0.61% | l_continuity=0.0006(w=1.000) | area=20880.7 | g3=0.948(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  150 || loss=2.1996 | MpErr=0.04% | l_continuity=0.0033(w=1.000) | area=21051.7 | g3=0.948(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  160 || loss=2.3252 | MpErr=1.37% | l_continuity=0.0053(w=1.000) | area=21005.4 | g3=0.947(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  170 || loss=2.4745 | MpErr=1.18% | l_continuity=0.0103(w=1.000) | area=20970.7 | g3=0.947(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  180 || loss=2.6808 | MpErr=2.96% | l_continuity=0.0106(w=1.000) | area=20744.8 | g3=0.947(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  190 || loss=2.8238 | MpErr=0.75% | l_continuity=0.0163(w=1.000) | area=21049.6 | g3=0.947(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  200 || loss=3.0088 | MpErr=0.41% | l_continuity=0.0297(w=1.000) | area=21076.6 | g3=0.946(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  210 || loss=3.1913 | MpErr=0.40% | l_continuity=0.0417(w=1.000) | area=21079.6 | g3=0.946(17/17) g4=0.946(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  220 || loss=3.3462 | MpErr=1.13% | l_continuity=0.0590(w=1.000) | area=21016.3 | g3=0.945(17/17) g4=0.945(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  230 || loss=3.5654 | MpErr=2.89% | l_continuity=0.0589(w=1.000) | area=20995.7 | g3=0.944(17/17) g4=0.945(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  240 || loss=3.6574 | MpErr=1.04% | l_continuity=0.0907(w=1.000) | area=21256.7 | g3=0.942(17/17) g4=0.945(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0
    Epoch  250 || loss=3.6595 | MpErr=0.53% | l_continuity=0.1059(w=1.000) | area=21260.9 | g3=0.940(17/17) g4=0.945(17/17) | groups=5 | log_alpha_sat=14.71% | dead0=0/34 | suspect=0
    Epoch  260 || loss=3.6310 | MpErr=0.65% | l_continuity=0.1248(w=1.000) | area=21430.2 | g3=0.935(17/17) g4=0.944(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  270 || loss=3.7212 | MpErr=1.79% | l_continuity=0.1796(w=1.000) | area=21510.4 | g3=0.930(17/17) g4=0.944(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  280 || loss=3.8579 | MpErr=2.01% | l_continuity=0.4562(w=1.000) | area=21635.0 | g3=0.927(17/17) g4=0.943(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  290 || loss=4.1973 | MpErr=4.79% | l_continuity=0.2281(w=1.000) | area=21633.7 | g3=0.925(17/17) g4=0.943(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  300 || loss=4.4500 | MpErr=7.77% | l_continuity=0.3050(w=1.000) | area=20913.5 | g3=0.923(17/17) g4=0.943(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  310 || loss=3.6971 | MpErr=1.90% | l_continuity=0.1701(w=1.000) | area=21324.6 | g3=0.921(17/17) g4=0.943(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  320 || loss=3.8709 | MpErr=4.49% | l_continuity=0.3216(w=1.000) | area=21726.2 | g3=0.919(17/17) g4=0.942(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  330 || loss=3.5583 | MpErr=0.13% | l_continuity=0.6459(w=0.999) | area=21482.5 | g3=0.916(17/17) g4=0.942(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  340 || loss=3.8706 | MpErr=5.83% | l_continuity=0.2262(w=0.998) | area=21152.6 | g3=0.914(17/17) g4=0.942(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  350 || loss=3.4184 | MpErr=0.78% | l_continuity=0.4266(w=0.994) | area=21533.8 | g3=0.912(17/17) g4=0.942(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  360 || loss=4.6892 | MpErr=6.13% | l_continuity=0.4871(w=0.983) | area=21635.9 | g3=0.910(17/17) g4=0.941(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  370 || loss=3.4558 | MpErr=2.21% | l_continuity=0.5221(w=0.955) | area=21525.5 | g3=0.909(17/17) g4=0.941(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  380 || loss=3.3665 | MpErr=5.32% | l_continuity=0.3914(w=0.887) | area=21957.7 | g3=0.907(17/17) g4=0.941(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  390 || loss=2.9446 | MpErr=3.14% | l_continuity=0.4030(w=0.745) | area=21995.8 | g3=0.905(17/17) g4=0.941(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  400 || loss=3.0683 | MpErr=2.02% | l_continuity=0.2584(w=0.525) | area=21647.5 | g3=0.904(17/17) g4=0.940(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  410 || loss=2.6593 | MpErr=0.96% | l_continuity=0.5859(w=0.305) | area=21766.1 | g3=0.902(17/17) g4=0.940(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  420 || loss=2.4260 | MpErr=2.08% | l_continuity=0.3863(w=0.163) | area=21991.0 | g3=0.901(17/17) g4=0.940(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  430 || loss=2.8632 | MpErr=2.68% | l_continuity=0.3049(w=0.095) | area=21655.4 | g3=0.899(17/17) g4=0.940(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  440 || loss=2.6866 | MpErr=0.76% | l_continuity=0.7589(w=0.067) | area=21738.1 | g3=0.898(17/17) g4=0.939(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  450 || loss=2.3347 | MpErr=0.49% | l_continuity=0.5991(w=0.056) | area=21775.5 | g3=0.896(17/17) g4=0.939(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  460 || loss=2.1938 | MpErr=1.27% | l_continuity=0.4855(w=0.052) | area=22122.2 | g3=0.894(17/17) g4=0.939(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  470 || loss=2.2941 | MpErr=2.91% | l_continuity=0.5323(w=0.051) | area=22187.6 | g3=0.892(17/17) g4=0.938(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  480 || loss=2.1571 | MpErr=0.77% | l_continuity=0.5176(w=0.050) | area=22071.8 | g3=0.890(17/17) g4=0.938(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  490 || loss=2.4449 | MpErr=1.46% | l_continuity=0.7404(w=0.050) | area=22053.7 | g3=0.889(17/17) g4=0.937(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    Epoch  499 || loss=2.6045 | MpErr=3.81% | l_continuity=0.4662(w=0.050) | area=21912.1 | g3=0.887(17/17) g4=0.937(17/17) | groups=5 | log_alpha_sat=20.59% | dead0=0/34 | suspect=0
    
    [Dynamic Split] 분할 이벤트 없음 - candidate는 끝까지 파트당 1 두께 유지 (Phase A/B)
    >>> Stage 4: 존재 확정 + 형상/두께 재수렴 (500 -> 700), 애매 게이트 27개
    [Stage4] epoch  500 || loss=2.6694 | MpErr=3.79% | area=21887.0
    [Stage4] epoch  520 || loss=2.0879 | MpErr=0.64% | area=22256.0
    [Stage4] epoch  540 || loss=2.0481 | MpErr=0.59% | area=22202.8
    [Stage4] epoch  560 || loss=1.9486 | MpErr=0.49% | area=20557.0
    [Stage4] epoch  580 || loss=1.9984 | MpErr=2.11% | area=21221.6
    [Stage4] epoch  600 || loss=1.9240 | MpErr=0.92% | area=20657.1
    [Stage4] epoch  620 || loss=1.8838 | MpErr=0.52% | area=20465.5
    [Stage4] epoch  640 || loss=1.8823 | MpErr=1.20% | area=20433.6
    [Stage4] epoch  660 || loss=1.8319 | MpErr=0.26% | area=19898.3
    [Stage4] epoch  680 || loss=1.8950 | MpErr=2.09% | area=20214.1
    [Stage4] epoch  699 || loss=1.8546 | MpErr=1.76% | area=20056.2
    [viz] 학습 결과 dashboard 저장: reports/figures/AI_design_v1_3_result.png
    [viz] 인터랙티브 3D 저장: reports/figures/AI_design_v1_3_3d.html
    [report] hard existence 판정용 temperature=0.3000 (epoch 499 기준)
    
    # AI_design_v1_2 — Initial vs Final Design Property Report
    
    (hard existence 판정 temperature=0.3000 — 학습 종료 시점 어닐링 온도와 일치)
    
    ## 1. Section-wise Mp (N*mm)
    
    | sec | Mp initial | Mp target | Mp final(soft) | Mp final(hard) | hard/target(%) | flag |
    |----:|-----------:|----------:|---------------:|---------------:|---------------:|:----|
    | 0 | 55,297,616 | 26,444,055 | 28,308,586 | 28,308,586 | 107.1 | target miss |
    | 1 | 52,739,376 | 29,211,881 | 30,087,574 | 30,087,574 | 103.0 | target miss |
    | 2 | 50,241,736 | 31,397,729 | 32,381,520 | 32,381,520 | 103.1 | target miss |
    | 3 | 47,804,696 | 33,087,380 | 33,768,396 | 33,768,396 | 102.1 | target miss |
    | 4 | 45,428,256 | 33,704,378 | 34,201,308 | 34,201,308 | 101.5 |  |
    | 5 | 43,112,400 | 32,835,017 | 33,245,946 | 33,245,946 | 101.3 |  |
    | 6 | 40,857,144 | 30,467,844 | 30,961,660 | 30,961,660 | 101.6 |  |
    | 7 | 38,662,480 | 27,734,809 | 28,208,444 | 28,208,444 | 101.7 |  |
    | 8 | 36,528,412 | 26,413,902 | 26,773,408 | 26,773,408 | 101.4 |  |
    | 9 | 34,454,944 | 27,985,091 | 27,819,698 | 27,819,698 | 99.4 |  |
    | 10 | 32,442,064 | 32,922,019 | 32,739,208 | 32,739,208 | 99.4 |  |
    | 11 | 30,489,774 | 39,716,721 | 39,359,960 | 39,359,960 | 99.1 |  |
    | 12 | 28,598,076 | 45,526,897 | 45,100,608 | 45,100,608 | 99.1 |  |
    | 13 | 26,766,974 | 47,458,156 | 47,465,800 | 47,465,800 | 100.0 |  |
    | 14 | 24,996,464 | 44,876,148 | 45,296,544 | 45,296,544 | 100.9 |  |
    | 15 | 23,286,536 | 39,505,207 | 39,651,136 | 39,651,136 | 100.4 |  |
    | 16 | 21,637,200 | 35,005,895 | 34,786,116 | 34,786,116 | 99.4 |  |
    | **sum** | 633,344,148 | 584,293,129 | 590,155,912 | 590,155,912 | 101.0 | |
    
    ## 2. Total cross-section area (mm^2, 17-section sum)
    
    | initial | final(soft) | final(hard) | delta(hard-init) |
    |--------:|------------:|------------:|-----------------:|
    | 24,056.9 | 19,905.0 | 19,905.0 | -4,151.8 (-17.3%) |
    
    | 감량 목표 | 22,854.0 mm² (초기 대비 95.0%) |
    | 달성률 | 87.1% (100% 이하면 목표 달성) |
    
    - 죽음의 구간(0.08 < z_gate < 0.5) 잔존 게이트: **0개** (임계 0.3 적용 시 구제: 0개)
    
    ## 3. Part thickness (mm, 제조 기준 = pooled t_raw)
    
    | part | initial t | final t (per run) | 존재 (17섹션, O=alive/X=deleted) |
    |:-----|----------:|:------------------|:--------------------------------|
    | #00(Outer) | 2.300 | 2.283 (전 섹션 공통) | 항상 존재 |
    | #03(Plate) | 1.600 | 0.700 (전 섹션 공통) | 항상 존재 |
    | #06(Inner) | 1.600 | 1.387 (전 섹션 공통) | 항상 존재 |
    | #07(Patch1) | 1.400 | 0.700 (run 1개) | OOOOOOOOOOOOOOOOO (17/17 alive) |
    | #08(Patch2) | 1.600 | 0.702 (run 1개) | OOOOOOOOOOOOOOOOO (17/17 alive) |
    
    [report] 비교 리포트 저장: reports/AI_design_v1_3_report.md
    >>> Stage 4 진입 여부: True
    [done] weights/bpillar_17sec_v1_3.pt, reports/figures/AI_design_v1_3_result.png, reports/figures/AI_design_v1_3_3d.html, reports/AI_design_v1_3_report.md 저장 완료
    
