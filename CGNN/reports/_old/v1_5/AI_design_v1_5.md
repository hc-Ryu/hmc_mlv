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
    Epoch    0 || loss=1.1544 | MpErr=7.94% | l_continuity=0.0000(w=1.000) | area=23972.7 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=0.50)
    Epoch   10 || loss=16.5255 | MpErr=50.21% | l_continuity=0.0000(w=1.000) | area=11026.8 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=0.50)
    [Gate Active] epoch 11: gate_params lr -> 1e-2 (section-aware 게이트 학습 시작)
    [R4] epoch 16: 최초로 게이트가 0.5 미만으로 하강함 (정상 신호)
    Epoch   20 || loss=11.3718 | MpErr=30.12% | l_continuity=0.0000(w=1.000) | area=14505.3 | g3=0.496(17/17) g4=0.496(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=0.69)
    Epoch   30 || loss=5.0861 | MpErr=18.19% | l_continuity=0.0000(w=1.000) | area=17066.1 | g3=0.490(17/17) g4=0.490(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=1.29)
    Epoch   40 || loss=5.2638 | MpErr=18.65% | l_continuity=0.0000(w=1.000) | area=16972.8 | g3=0.487(17/17) g4=0.487(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=2.19)
    Epoch   50 || loss=2.2705 | MpErr=7.07% | l_continuity=0.0000(w=1.000) | area=19742.6 | g3=0.485(17/17) g4=0.486(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=3.21)
    Epoch   60 || loss=2.1094 | MpErr=5.57% | l_continuity=0.0000(w=1.000) | area=20122.6 | g3=0.485(17/17) g4=0.486(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=4.13)
    Epoch   70 || loss=1.9760 | MpErr=3.60% | l_continuity=0.0000(w=1.000) | area=20586.1 | g3=0.485(17/17) g4=0.486(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=4.77)
    Epoch   80 || loss=2.0341 | MpErr=2.73% | l_continuity=0.0000(w=1.000) | area=22109.0 | g3=0.487(17/17) g4=0.487(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch   90 || loss=1.9385 | MpErr=2.56% | l_continuity=0.0000(w=1.000) | area=20824.8 | g3=0.489(17/17) g4=0.488(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  100 || loss=1.9608 | MpErr=1.87% | l_continuity=0.0000(w=1.000) | area=21259.1 | g3=0.491(17/17) g4=0.489(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  110 || loss=2.3313 | MpErr=1.36% | l_continuity=0.0000(w=1.000) | area=19846.5 | g3=0.492(17/17) g4=0.490(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0802(w=5.00)
    Epoch  120 || loss=1.9396 | MpErr=1.80% | l_continuity=0.0000(w=1.000) | area=20531.4 | g3=0.494(17/17) g4=0.490(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  130 || loss=1.9884 | MpErr=0.60% | l_continuity=0.0000(w=1.000) | area=20552.9 | g3=0.495(17/17) g4=0.490(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  140 || loss=2.0661 | MpErr=1.70% | l_continuity=0.0030(w=1.000) | area=20362.9 | g3=0.495(17/17) g4=0.490(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  150 || loss=2.1756 | MpErr=0.87% | l_continuity=0.0022(w=1.000) | area=20550.2 | g3=0.497(17/17) g4=0.490(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  160 || loss=2.3055 | MpErr=1.26% | l_continuity=0.0059(w=1.000) | area=20539.0 | g3=0.499(17/17) g4=0.490(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  170 || loss=2.4515 | MpErr=1.05% | l_continuity=0.0097(w=1.000) | area=20530.2 | g3=0.500(17/17) g4=0.490(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  180 || loss=2.6223 | MpErr=0.14% | l_continuity=0.0105(w=1.000) | area=20634.7 | g3=0.503(17/17) g4=0.491(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  190 || loss=2.8155 | MpErr=2.88% | l_continuity=0.0148(w=1.000) | area=20508.0 | g3=0.505(17/17) g4=0.492(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  200 || loss=2.9561 | MpErr=1.91% | l_continuity=0.0268(w=1.000) | area=20568.1 | g3=0.508(17/17) g4=0.493(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  210 || loss=3.1985 | MpErr=1.84% | l_continuity=0.0706(w=1.000) | area=20778.6 | g3=0.510(17/17) g4=0.494(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  220 || loss=3.3109 | MpErr=2.58% | l_continuity=0.0396(w=1.000) | area=20554.2 | g3=0.513(17/17) g4=0.495(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  230 || loss=4.0366 | MpErr=1.81% | l_continuity=0.0990(w=1.000) | area=20831.0 | g3=0.515(17/17) g4=0.496(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.1074(w=5.00)
    Epoch  240 || loss=3.5562 | MpErr=2.76% | l_continuity=0.0315(w=1.000) | area=20575.9 | g3=0.517(17/17) g4=0.498(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  250 || loss=3.6901 | MpErr=1.43% | l_continuity=0.0444(w=1.000) | area=20841.0 | g3=0.519(17/17) g4=0.499(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0000(w=5.00)
    Epoch  260 || loss=4.9154 | MpErr=1.06% | l_continuity=0.0675(w=1.000) | area=20757.5 | g3=0.520(17/17) g4=0.499(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.2551(w=5.00)
    Epoch  270 || loss=4.0167 | MpErr=3.07% | l_continuity=0.1821(w=1.000) | area=20738.6 | g3=0.521(17/17) g4=0.500(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.0845(w=5.00)
    Epoch  280 || loss=5.8613 | MpErr=0.16% | l_continuity=0.3050(w=1.000) | area=20979.8 | g3=0.521(17/17) g4=0.500(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.4692(w=5.00)
    Epoch  290 || loss=6.0679 | MpErr=3.00% | l_continuity=0.1823(w=1.000) | area=20826.6 | g3=0.521(17/17) g4=0.501(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.4988(w=5.00)
    Epoch  300 || loss=7.5381 | MpErr=3.68% | l_continuity=0.5776(w=1.000) | area=21496.8 | g3=0.522(17/17) g4=0.501(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.7890(w=5.00)
    Epoch  310 || loss=6.6252 | MpErr=4.08% | l_continuity=0.3489(w=1.000) | area=21328.3 | g3=0.521(17/17) g4=0.501(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.5958(w=5.00)
    Epoch  320 || loss=7.8220 | MpErr=7.37% | l_continuity=0.6142(w=1.000) | area=21803.7 | g3=0.521(17/17) g4=0.501(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.7432(w=5.00)
    Epoch  330 || loss=7.0016 | MpErr=0.05% | l_continuity=0.2997(w=0.999) | area=21337.2 | g3=0.521(17/17) g4=0.501(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.7524(w=5.00)
    Epoch  340 || loss=7.1532 | MpErr=3.56% | l_continuity=0.3873(w=0.998) | area=21324.7 | g3=0.521(17/17) g4=0.500(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.7557(w=5.00)
    Epoch  350 || loss=7.7924 | MpErr=4.71% | l_continuity=0.3141(w=0.994) | area=20828.0 | g3=0.520(17/17) g4=0.500(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=0.6871(w=5.00)
    Epoch  360 || loss=10.4004 | MpErr=4.81% | l_continuity=0.5827(w=0.983) | area=21386.7 | g3=0.520(17/17) g4=0.499(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.2460(w=5.00)
    Epoch  370 || loss=9.6609 | MpErr=3.42% | l_continuity=0.2710(w=0.955) | area=21653.4 | g3=0.519(17/17) g4=0.499(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.2865(w=5.00)
    Epoch  380 || loss=12.7095 | MpErr=5.09% | l_continuity=0.4742(w=0.887) | area=21982.6 | g3=0.519(17/17) g4=0.498(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.8827(w=5.00)
    Epoch  390 || loss=10.6290 | MpErr=4.29% | l_continuity=0.3197(w=0.745) | area=21905.8 | g3=0.519(17/17) g4=0.497(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.4612(w=5.00)
    Epoch  400 || loss=11.8763 | MpErr=6.67% | l_continuity=0.4322(w=0.525) | area=21991.0 | g3=0.518(17/17) g4=0.497(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000 | l_col_aux=1.6498(w=5.00)
    Epoch  410 || loss=13.6107 | MpErr=3.29% | l_continuity=0.4107(w=0.305) | area=21854.1 | g3=0.517(17/17) g4=0.496(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0100 | l_col_aux=2.0279(w=5.00)
    Epoch  420 || loss=11.4858 | MpErr=1.46% | l_continuity=0.3755(w=0.163) | area=21974.2 | g3=0.517(17/17) g4=0.495(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0200 | l_col_aux=1.5981(w=5.00)
    Epoch  430 || loss=10.8396 | MpErr=0.46% | l_continuity=0.4567(w=0.095) | area=22295.7 | g3=0.516(17/17) g4=0.495(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0300 | l_col_aux=1.5326(w=5.00)
    Epoch  440 || loss=9.0200 | MpErr=4.22% | l_continuity=0.4890(w=0.067) | area=22483.7 | g3=0.515(17/17) g4=0.494(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0400 | l_col_aux=1.1164(w=5.00)
    Epoch  450 || loss=11.8544 | MpErr=1.92% | l_continuity=0.5358(w=0.056) | area=22351.1 | g3=0.515(17/17) g4=0.493(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=1.7449(w=5.00)
    Epoch  460 || loss=9.6531 | MpErr=2.69% | l_continuity=0.5644(w=0.052) | area=22259.8 | g3=0.514(17/17) g4=0.493(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=1.1772(w=5.00)
    Epoch  470 || loss=13.6160 | MpErr=1.99% | l_continuity=0.6282(w=0.051) | area=22351.6 | g3=0.514(17/17) g4=0.492(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=1.9876(w=5.00)
    Epoch  480 || loss=15.2779 | MpErr=0.20% | l_continuity=0.5849(w=0.050) | area=22424.4 | g3=0.513(17/17) g4=0.492(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=2.3146(w=5.00)
    Epoch  490 || loss=7.2433 | MpErr=2.51% | l_continuity=0.3493(w=0.050) | area=21908.6 | g3=0.513(17/17) g4=0.491(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=0.8599(w=5.00)
    Epoch  499 || loss=8.7719 | MpErr=3.09% | l_continuity=0.4352(w=0.050) | area=21943.8 | g3=0.512(17/17) g4=0.491(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0500 | l_col_aux=1.1182(w=5.00)
    
    [Dynamic Split] 분할 이벤트 없음 - candidate는 끝까지 파트당 1 두께 유지 (Phase A/B)
    >>> Stage 4: 존재 확정 + 형상/두께 재수렴 (500 -> 700), 애매 게이트 34개
    [Stage4] epoch  500 || loss=8.3057 | MpErr=3.12% | area=22481.9
    [Stage4] epoch  520 || loss=6.8991 | MpErr=1.40% | area=22188.2
    [Stage4] epoch  540 || loss=6.4134 | MpErr=0.05% | area=21388.4
    [Stage4] epoch  560 || loss=5.9628 | MpErr=0.01% | area=20747.9
    [Stage4] epoch  580 || loss=5.4282 | MpErr=0.76% | area=21028.3
    [Stage4] epoch  600 || loss=5.9381 | MpErr=0.71% | area=20462.9
    [Stage4] epoch  620 || loss=5.8687 | MpErr=1.91% | area=20514.2
    [Stage4] epoch  640 || loss=5.0636 | MpErr=0.25% | area=19979.0
    [Stage4] epoch  660 || loss=5.0900 | MpErr=1.29% | area=20318.5
    [Stage4] epoch  680 || loss=4.7358 | MpErr=1.10% | area=20196.1
    [Stage4] epoch  699 || loss=5.2823 | MpErr=0.39% | area=19830.9
    [viz] 학습 결과 dashboard 저장: reports/figures/AI_design_v1_5_result.png
    [viz] 인터랙티브 3D 저장: reports/figures/AI_design_v1_5_3d.html
    [report] hard existence 판정용 temperature=0.3000 (epoch 499 기준)
    
    # AI_design_v1_2 — Initial vs Final Design Property Report
    
    (hard existence 판정 temperature=0.3000 — 학습 종료 시점 어닐링 온도와 일치)
    
    ## 1. Section-wise Mp (N*mm)
    
    | sec | Mp initial | Mp target | Mp final(soft) | Mp final(hard) | hard/target(%) | flag |
    |----:|-----------:|----------:|---------------:|---------------:|---------------:|:----|
    | 0 | 55,297,616 | 26,444,055 | 30,751,324 | 30,751,324 | 116.3 | target miss |
    | 1 | 52,739,376 | 29,211,881 | 30,588,018 | 30,588,018 | 104.7 | target miss |
    | 2 | 50,241,736 | 31,397,729 | 31,175,232 | 31,175,232 | 99.3 |  |
    | 3 | 47,804,696 | 33,087,380 | 33,801,036 | 33,801,036 | 102.2 | target miss |
    | 4 | 45,428,256 | 33,704,378 | 33,860,744 | 33,860,744 | 100.5 |  |
    | 5 | 43,112,400 | 32,835,017 | 33,010,220 | 33,010,220 | 100.5 |  |
    | 6 | 40,857,144 | 30,467,844 | 31,136,700 | 31,136,700 | 102.2 | target miss |
    | 7 | 38,662,480 | 27,734,809 | 27,464,042 | 27,464,042 | 99.0 |  |
    | 8 | 36,528,412 | 26,413,902 | 26,295,320 | 26,295,320 | 99.6 |  |
    | 9 | 34,454,944 | 27,985,091 | 28,281,148 | 28,281,148 | 101.1 |  |
    | 10 | 32,442,064 | 32,922,019 | 32,528,072 | 32,528,072 | 98.8 |  |
    | 11 | 30,489,774 | 39,716,721 | 39,537,544 | 39,537,544 | 99.5 |  |
    | 12 | 28,598,076 | 45,526,897 | 45,170,356 | 45,170,356 | 99.2 |  |
    | 13 | 26,766,974 | 47,458,156 | 47,404,524 | 47,404,524 | 99.9 |  |
    | 14 | 24,996,464 | 44,876,148 | 45,208,460 | 45,208,460 | 100.7 |  |
    | 15 | 23,286,536 | 39,505,207 | 39,555,976 | 39,555,976 | 100.1 |  |
    | 16 | 21,637,200 | 35,005,895 | 34,778,152 | 34,778,152 | 99.3 |  |
    | **sum** | 633,344,148 | 584,293,129 | 590,546,868 | 590,546,868 | 101.1 | |
    
    ## 2. Total cross-section area (mm^2, 17-section sum)
    
    | initial | final(soft) | final(hard) | delta(hard-init) |
    |--------:|------------:|------------:|-----------------:|
    | 24,056.9 | 20,065.2 | 20,065.2 | -3,991.7 (-16.6%) |
    
    | 감량 목표 | 22,854.0 mm² (초기 대비 95.0%) |
    | 달성률 | 87.8% (100% 이하면 목표 달성) |
    
    - 죽음의 구간(0.08 < z_gate < 0.5) 잔존 게이트: **0개** (임계 0.3 적용 시 구제: 0개)
    
    ## 3. Part thickness (mm, 제조 기준 = pooled t_raw)
    
    | part | initial t | final t (per run) | 존재 (17섹션, O=alive/X=deleted) |
    |:-----|----------:|:------------------|:--------------------------------|
    | #00(Outer) | 2.300 | 2.275 (전 섹션 공통) | 항상 존재 |
    | #03(Plate) | 1.600 | 0.701 (전 섹션 공통) | 항상 존재 |
    | #06(Inner) | 1.600 | 1.401 (전 섹션 공통) | 항상 존재 |
    | #07(Patch1) | 1.400 | run0(sec3-6)=0.700, run1(sec9-16)=0.700 | XXXOOOOXXOOOOOOOO (12/17 alive) |
    | #08(Patch2) | 1.600 | run0(sec3-6)=0.700, run1(sec9-16)=0.700 | XXXOOOOXXOOOOOOOO (12/17 alive) |
    
    [report] 비교 리포트 저장: reports/AI_design_v1_5_report.md
    >>> Stage 4 진입 여부: True
    [done] weights/bpillar_17sec_v1_5.pt, reports/figures/AI_design_v1_5_result.png, reports/figures/AI_design_v1_5_3d.html, reports/AI_design_v1_5_report.md 저장 완료
    
