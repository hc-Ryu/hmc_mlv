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
    Epoch    0 || loss=1.1580 | MpErr=8.07% | l_continuity=0.0000(w=1.000) | area=24007.6 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch   10 || loss=16.1583 | MpErr=49.61% | l_continuity=0.0000(w=1.000) | area=11132.3 | g3=1.000(17/17) g4=1.000(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    [Gate Active] epoch 11: gate_params lr -> 1e-2 (section-aware 게이트 학습 시작)
    Epoch   20 || loss=1.9351 | MpErr=2.01% | l_continuity=0.0000(w=1.000) | area=21161.2 | g3=0.515(17/17) g4=0.515(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch   30 || loss=6.1012 | MpErr=19.16% | l_continuity=0.0000(w=1.000) | area=26807.4 | g3=0.520(17/17) g4=0.520(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch   40 || loss=2.2101 | MpErr=6.56% | l_continuity=0.0000(w=1.000) | area=19939.1 | g3=0.521(17/17) g4=0.520(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch   50 || loss=2.1063 | MpErr=3.60% | l_continuity=0.0000(w=1.000) | area=22582.8 | g3=0.521(17/17) g4=0.521(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch   60 || loss=1.9332 | MpErr=1.95% | l_continuity=0.0000(w=1.000) | area=21103.1 | g3=0.521(17/17) g4=0.520(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch   70 || loss=2.4396 | MpErr=8.30% | l_continuity=0.0000(w=1.000) | area=19469.7 | g3=0.521(17/17) g4=0.519(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch   80 || loss=2.1553 | MpErr=4.20% | l_continuity=0.0000(w=1.000) | area=22722.0 | g3=0.521(17/17) g4=0.519(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch   90 || loss=2.3267 | MpErr=7.52% | l_continuity=0.0000(w=1.000) | area=19665.3 | g3=0.521(17/17) g4=0.519(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  100 || loss=53.5881 | MpErr=7.21% | l_continuity=0.0000(w=1.000) | area=19615.6 | g3=0.521(17/17) g4=0.518(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    [R4] epoch 108: 최초로 게이트가 0.5 미만으로 하강함 (정상 신호)
    Epoch  110 || loss=366.9076 | MpErr=0.22% | l_continuity=0.0000(w=1.000) | area=18214.6 | g3=0.508(17/17) g4=0.521(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  120 || loss=453.2853 | MpErr=8.63% | l_continuity=0.0021(w=1.000) | area=16435.5 | g3=0.478(17/17) g4=0.528(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  130 || loss=408.0626 | MpErr=7.06% | l_continuity=0.0000(w=1.000) | area=16555.8 | g3=0.447(17/17) g4=0.536(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  140 || loss=330.8251 | MpErr=3.73% | l_continuity=0.0000(w=1.000) | area=17369.9 | g3=0.424(17/17) g4=0.542(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  150 || loss=270.5498 | MpErr=7.81% | l_continuity=0.0000(w=1.000) | area=16698.0 | g3=0.401(17/17) g4=0.547(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  160 || loss=387.0188 | MpErr=2.84% | l_continuity=0.0000(w=1.000) | area=18782.0 | g3=0.385(17/17) g4=0.551(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  170 || loss=280.6364 | MpErr=6.02% | l_continuity=0.0000(w=1.000) | area=17100.3 | g3=0.369(17/17) g4=0.555(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  180 || loss=768.0991 | MpErr=3.40% | l_continuity=0.0000(w=1.000) | area=18671.2 | g3=0.352(17/17) g4=0.560(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  190 || loss=1097.0640 | MpErr=2.92% | l_continuity=0.0000(w=1.000) | area=17247.9 | g3=0.335(17/17) g4=0.564(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  200 || loss=1183.5841 | MpErr=9.04% | l_continuity=0.0000(w=1.000) | area=16026.7 | g3=0.319(17/17) g4=0.567(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  210 || loss=499.1568 | MpErr=4.16% | l_continuity=0.0000(w=1.000) | area=17069.4 | g3=0.300(17/17) g4=0.571(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  220 || loss=647.7944 | MpErr=10.18% | l_continuity=0.0000(w=1.000) | area=16256.9 | g3=0.278(17/17) g4=0.575(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  230 || loss=273.5819 | MpErr=9.84% | l_continuity=0.0000(w=1.000) | area=16345.1 | g3=0.263(17/17) g4=0.578(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  240 || loss=440.9009 | MpErr=3.58% | l_continuity=0.0000(w=1.000) | area=18896.0 | g3=0.247(17/17) g4=0.584(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  250 || loss=241.8174 | MpErr=4.43% | l_continuity=0.0000(w=1.000) | area=17298.5 | g3=0.231(17/17) g4=0.590(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  260 || loss=339.2136 | MpErr=7.47% | l_continuity=0.0000(w=1.000) | area=16762.7 | g3=0.217(17/17) g4=0.595(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  270 || loss=259.8337 | MpErr=0.25% | l_continuity=0.0000(w=1.000) | area=18147.8 | g3=0.204(17/17) g4=0.600(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  280 || loss=296.9055 | MpErr=4.59% | l_continuity=0.0000(w=1.000) | area=19098.2 | g3=0.193(17/17) g4=0.604(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  290 || loss=483.8322 | MpErr=7.28% | l_continuity=0.0000(w=1.000) | area=16673.9 | g3=0.182(17/17) g4=0.608(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  300 || loss=252.8250 | MpErr=1.36% | l_continuity=0.0000(w=1.000) | area=18582.9 | g3=0.172(17/17) g4=0.613(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  310 || loss=670.0328 | MpErr=5.12% | l_continuity=0.0000(w=1.000) | area=16879.4 | g3=0.162(17/17) g4=0.617(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    Epoch  320 || loss=563.1531 | MpErr=1.96% | l_continuity=0.0000(w=1.000) | area=17713.4 | g3=0.150(17/17) g4=0.622(17/17) | groups=5 | log_alpha_sat=0.00% | dead0=0/34 | suspect=0 | alpha_ent=0.0000
    [Dynamic Split] epoch 325: DELETED 확정 [[11, 0]] -> candidate run 수 [2, 1] (두께 그룹 6개로 재구성)
    Epoch  330 || loss=228.3650 | MpErr=4.99% | l_continuity=0.0000(w=0.999) | area=17235.7 | g3=0.140(16/17) g4=0.626(17/17) | groups=6 | log_alpha_sat=0.00% | dead0=0/33 | suspect=0 | alpha_ent=0.0000
    Epoch  340 || loss=515.9644 | MpErr=6.97% | l_continuity=0.0000(w=0.998) | area=16623.1 | g3=0.129(16/17) g4=0.632(17/17) | groups=6 | log_alpha_sat=0.00% | dead0=0/33 | suspect=0 | alpha_ent=0.0000
    Epoch  350 || loss=622.8496 | MpErr=3.40% | l_continuity=0.0000(w=0.994) | area=17480.3 | g3=0.119(16/17) g4=0.637(17/17) | groups=6 | log_alpha_sat=0.00% | dead0=0/33 | suspect=0 | alpha_ent=0.0000
    Epoch  360 || loss=298.5237 | MpErr=3.85% | l_continuity=0.0000(w=0.983) | area=17381.6 | g3=0.111(16/17) g4=0.640(17/17) | groups=6 | log_alpha_sat=0.00% | dead0=0/33 | suspect=0 | alpha_ent=0.0000
    Epoch  370 || loss=530.5736 | MpErr=1.59% | l_continuity=0.0000(w=0.955) | area=17793.9 | g3=0.104(16/17) g4=0.644(17/17) | groups=6 | log_alpha_sat=0.00% | dead0=0/33 | suspect=0 | alpha_ent=0.0000
    Epoch  380 || loss=236.7142 | MpErr=5.72% | l_continuity=0.0000(w=0.887) | area=17057.9 | g3=0.095(16/17) g4=0.647(17/17) | groups=6 | log_alpha_sat=0.00% | dead0=0/33 | suspect=0 | alpha_ent=0.0000
    [Dynamic Split] epoch 381: DELETED 확정 [[6, 0]] -> candidate run 수 [3, 1] (두께 그룹 7개로 재구성)
    [Dynamic Split] epoch 389: DELETED 확정 [[10, 0]] -> candidate run 수 [3, 1] (두께 그룹 7개로 재구성)
    Epoch  390 || loss=355.0381 | MpErr=1.43% | l_continuity=0.0000(w=0.745) | area=17950.8 | g3=0.086(14/17) g4=0.652(17/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/31 | suspect=0 | alpha_ent=0.0000
    Epoch  400 || loss=296.1682 | MpErr=7.27% | l_continuity=0.0011(w=0.525) | area=16692.3 | g3=0.076(14/17) g4=0.657(17/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/31 | suspect=0 | alpha_ent=0.0000
    [Dynamic Split] epoch 401: DELETED 확정 [[7, 0]] -> candidate run 수 [3, 1] (두께 그룹 7개로 재구성)
    Epoch  410 || loss=387.9072 | MpErr=7.82% | l_continuity=0.0022(w=0.305) | area=16434.2 | g3=0.069(13/17) g4=0.665(17/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/30 | suspect=0 | alpha_ent=0.0100
    Epoch  420 || loss=560.3532 | MpErr=10.22% | l_continuity=0.0017(w=0.163) | area=15824.6 | g3=0.064(13/17) g4=0.675(17/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/30 | suspect=0 | alpha_ent=0.0200
    [Dynamic Split] epoch 430: DELETED 확정 [[0, 0]] -> candidate run 수 [3, 1] (두께 그룹 7개로 재구성)
    Epoch  430 || loss=465.8373 | MpErr=9.69% | l_continuity=0.0001(w=0.095) | area=16010.1 | g3=0.058(12/17) g4=0.687(17/17) | groups=7 | log_alpha_sat=0.00% | dead0=0/29 | suspect=0 | alpha_ent=0.0300
    [Dynamic Split] epoch 437: DELETED 확정 [[13, 0]] -> candidate run 수 [4, 1] (두께 그룹 8개로 재구성)
    Epoch  440 || loss=291.4778 | MpErr=5.37% | l_continuity=0.0010(w=0.067) | area=16588.8 | g3=0.052(11/17) g4=0.698(17/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/28 | suspect=0 | alpha_ent=0.0400
    Epoch  450 || loss=1190.4641 | MpErr=6.89% | l_continuity=0.0042(w=0.056) | area=16122.4 | g3=0.049(11/17) g4=0.704(17/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/28 | suspect=0 | alpha_ent=0.0500
    Epoch  460 || loss=866.0386 | MpErr=8.22% | l_continuity=0.0005(w=0.052) | area=16136.6 | g3=0.045(11/17) g4=0.710(17/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/28 | suspect=0 | alpha_ent=0.0500
    Epoch  470 || loss=503.9768 | MpErr=5.60% | l_continuity=0.0026(w=0.051) | area=16506.3 | g3=0.040(11/17) g4=0.717(17/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/28 | suspect=0 | alpha_ent=0.0500
    [Dynamic Split] epoch 472: DELETED 확정 [[15, 0]] -> candidate run 수 [5, 1] (두께 그룹 9개로 재구성)
    [Adaptive Extension] loop_max -> 532 (curriculum 스케줄은 500 기준으로 동결 유지)
    [Dynamic Split] epoch 473: DELETED 확정 [[14, 0]] -> candidate run 수 [4, 1] (두께 그룹 8개로 재구성)
    [Adaptive Extension] loop_max -> 533 (curriculum 스케줄은 500 기준으로 동결 유지)
    Epoch  480 || loss=388.9222 | MpErr=6.36% | l_continuity=0.0010(w=0.050) | area=16450.4 | g3=0.031(9/17) g4=0.725(17/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/26 | suspect=0 | alpha_ent=0.0500
    Epoch  490 || loss=285.4128 | MpErr=6.15% | l_continuity=0.0014(w=0.050) | area=16484.8 | g3=0.025(9/17) g4=0.733(17/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/26 | suspect=0 | alpha_ent=0.0500
    Epoch  500 || loss=539.8511 | MpErr=5.60% | l_continuity=0.0036(w=0.050) | area=16456.6 | g3=0.020(9/17) g4=0.740(17/17) | groups=8 | log_alpha_sat=0.00% | dead0=0/26 | suspect=0 | alpha_ent=0.0500
    [Dynamic Split] epoch 502: DELETED 확정 [[8, 0]] -> candidate run 수 [4, 1] (두께 그룹 8개로 재구성)
    [Adaptive Extension] loop_max -> 562 (curriculum 스케줄은 500 기준으로 동결 유지)
    Epoch  510 || loss=354.7264 | MpErr=6.00% | l_continuity=0.0005(w=0.050) | area=16616.6 | g3=0.015(8/17) g4=0.745(17/17) | groups=8 | log_alpha_sat=11.76% | dead0=3/25 | suspect=0 | alpha_ent=0.0500
    


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[1], line 1662
       1658 else:
       1659     assert_thickness_reachability(data.x)          # [v1.3 §3.4] 전 구간 도달성 사전 검증
       1661     model, history, base_coords, final_coords, final_z_gate, final_seg_ids, split_epochs, \
    -> 1662         pruning_state, final_epoch, death_log, enter_stage4 = run_training_multi(
       1663             data, target_mps, max_epochs=args.epochs,
       1664             area_target_ratio=args.area_target_ratio, w_mass=args.w_mass,
       1665             w_area_floor=args.w_area_floor)   # [v1.4] w_area_floor 추가
       1667     torch.save(model.state_dict(), "weights/bpillar_17sec_v1_4.pt")   # [v1.4] v1_3 가중치와 분리
       1669     part_ids = data.x[:, 4]
    

    Cell In[1], line 1452, in run_training_multi(data, target_mps, target_area, max_epochs, lr, weights, curriculum, curriculum_ratio, area_target_ratio, w_mass, w_area_floor)
       1449 area_ref = area_hist.get(epoch - AREA_REGRESS_WINDOW)
       1450 if area_ref is not None and info['area'] > area_ref \
       1451         and any(d["epoch"] > epoch - AREA_REGRESS_WINDOW for d in death_log):
    -> 1452     raise RuntimeError(f"[R3] 질량 역설: 게이트가 죽는 중 area 증가 "
       1453                         f"({area_ref:,.0f} -> {info['area']:,.0f}). "
       1454                         f"ALPHA_ENT_V13를 0.02로 낮춰 재시도할 것 (command_v1_3.md §7.0).")
       1456 # ── [v1.1] 동적 분할 이벤트 (command_v2.md §3.2): DELETED 확정마다 seg_ids 재계산 ──
       1457 if info['newly_deleted'].any():
    

    RuntimeError: [R3] 질량 역설: 게이트가 죽는 중 area 증가 (16,160 -> 16,652). ALPHA_ENT_V13를 0.02로 낮춰 재시도할 것 (command_v1_3.md §7.0).

