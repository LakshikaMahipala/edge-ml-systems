Amdahl Tools 

Goal
- Quantify how much end-to-end speedup is possible when accelerating one stage.

Scripts
1) scripts/amdahl_speedup.py
- computes acceleratable fraction p (inference fraction), serial fraction, and speedup table vs S.

2) scripts/amdahl_what_if.py
- apply speedup S to any stage (pre/inf/post/copy) and see end-to-end impact.

Run later (example)
python scripts/amdahl_speedup.py --t_pre_ms 3 --t_inf_ms 12 --t_post_ms 2 --t_copy_ms 1 --S_list 1,2,3,5,10,1000
python scripts/amdahl_what_if.py --t_pre_ms 3 --t_inf_ms 12 --t_post_ms 2 --t_copy_ms 1 --target inf --S 3
