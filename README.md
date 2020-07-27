Code associated with [AAAI 2020 paper](https://teamcore.seas.harvard.edu/files/teamcore/files/2020_02_teamcore_aaai_signaluncertainty.pdf): To Signal or Not To Signal: Exploiting Uncertain Real-Time Information in Signaling Games for Security and Sustainability, by Elizabeth Bondi, Hoon Oh, Haifeng Xu, Fei Fang, Bistra Dilkina, and Milind Tambe

Abstract: "Motivated by real-world deployment of drones for conservation, this paper advances the state-of-the-art in security games with signaling. The well-known defender-attacker security games framework can help in planning for such strategic deployments of sensors and human patrollers, and warning signals to ward off adversaries. However, we show that defenders can suffer significant losses when ignoring real-world uncertainties despite carefully planned security game strategies with signaling. In fact, defenders may perform worse than forgoing drones completely in this case. We address this shortcoming by proposing a novel game model that integrates signaling and sensor uncertainty; perhaps surprisingly, we show that defenders can still perform well via a signaling strategy that exploits uncertain real-time information. For example, even in the presence of uncertainty, the defender still has an informational advantage in knowing that she has or has not actually detected the attacker; and she can design a signaling scheme to 'mislead' the attacker who is uncertain as to whether he has been detected. We provide theoretical results, a novel algorithm, scale-up techniques, and experimental results from simulation based on our ongoing deployment of a conservation drone system in South Africa."

This is the research code used to produce the results in this paper, particularly in Fig. 3. It has been cleaned from its original version for easier understanding. This includes comments for each method and inline comments in some cases. It has been tested to reproduce the plots in the paper, but it is still research code. Please feel free to contact Elizabeth Bondi at ebondi@g.harvard.edu or open an issue if problems arise.

NOTE: Sometimes, you may see non-smoothness in the defender utility when ignoring uncertainty (e.g., detection uncertainty in plot 3c). The intuition behind this is that we as the defender are ignoring detection uncertainty, so we behave the same (non-optimal) way. However, the attacker behaves a certain way for some time, and then may behave a little differently, causing us as the defender to get a better utility. This can happen since the attacker and defender utilities are not completely correlated. This may also be seen in case_study.py, which may vary slightly for defender ignoring uncertainty if run multiple times.

You can generate .kml files for the case_study code as shown in this demo video: https://drive.google.com/file/d/1MLEEC9syFWYFUwdTeWR00fgVyt3YwQTs/view?usp=sharing

Dependencies:
- networkx
- matplotlib.pyplot
- numpy
- argparse
- scipy
- cplex (may throw exceptions if you try the student version to solve large problems!)

If you find this code useful, please consider citing:
```
@inproceedings{bondi2020signal,
  title={To Signal or Not To Signal: Exploiting Uncertain Real-Time Information in Signaling Games for Security and Sustainability.},
  author={Bondi, Elizabeth and Oh, Hoon and Xu, Haifeng and Fang, Fei and Dilkina, Bistra and Tambe, Milind},
  booktitle={AAAI},
  pages={1369--1377},
  year={2020}
}
```

How to reproduce plots (running sequentially, not tested):
```
python compare_heuristic_script.py 6 18 2 20 "all" -nsh -sp -s -m "3a"
python compare_heuristic_script.py 20 110 10 20 "all" -nsh -sp -s -m "3b"
python compare_defender_utilities_script.py 10 1 3 -nsh -s -sr -np -sp
python number_attackers_uavs_over_gamma_script.py 15 -nsh -w -s -sp
python case_study.py "new.kml" -o "3h.pdf" -s "WITHHELD"
```

How to reproduce plots (on cluster, tested):
```
(run 20+ times, see run_defender_utilities.sh for example:) python compare_defender_utilities_script.py 10 1 3 -nsh -g 1 -sr -np -lp -sg
python plot_cluster_results.py '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' '23' '24' -u -s -nsh

(run 20+ times, see run_timing_3a.sh for example:) python compare_heuristic_script.py 6 18 2 1 "all" -nsh -sp -np
python plot_cluster_results.py '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' '23' '24' -t -s -nsh -m '3a'

(run 20+ times, see run_timing_3b.sh for example:) python compare_heuristic_script.py 20 110 10 1 "all" -nsh -sp -np
python plot_cluster_results.py '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' '23' '24' -t -s -nsh -m '3b'

(run 20+ times, see run_ratio_3fg.sh for example:) python number_attackers_uavs_over_gamma_script.py 15 -nsh -w -lp -g 1 -sr -np -sg
python plot_cluster_results.py '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' '23' '24' -r -s -nsh

python case_study.py "new.kml" -o "3h.pdf" -s "WITHHELD"
```
- TIP: if you let slurm initiate 20 jobs, several will likely run at the exact same time, so graph generation will likely lead to the same graphs - could just loop through initiating jobs, or create a few extra and check, as in the above examples.
- NOTE: for number_attackers_uavs_over_gamma_script.py and compare_defender_utilities_script.py, to reproduce results in the paper, the same utilities must be used for all random graphs. Generate the U_dc, U_du, U_ac, U_au pickles (use -sp), then load them for all as shown above (-lp and -sg). For compare_defender_utilities_script.py, 3d, sometimes need to run more graphs for statistical significance.
- NOTE: for timing tests, particularly 3b, results will likely be faster than reported in the paper due to including 3 additional constraints in the relaxation which were initially omitted.
- NOTE: running number_attackers_uavs_over_gamma_script.py with N=15 often has more timeouts than timing 3b tests. This may be interesting to investigate further.

Other testing done before posting:
```
python compare_heuristic_script.py 20 22 2 2 "all" -s -m "3b" -nsh
python compare_defender_utilities_script.py 10 1 3 -nsh -g 2 -s
python compare_defender_utilities_script.py 10 1 3 -a -nu -nsh -g 2 -w
python compare_defender_utilities_script.py 10 1 3 -rt -nu -nsh -g 1 -w
python compare_defender_utilities_script.py 10 1 3 -o -nr -nsh -g 1 -w
python compare_defender_utilities_script.p 10 1 3 -o -nsh -g 1 -w
python case_study.py "new.kml" -o "3h.pdf" -s "WITHHELD"
python case_study.py "new.kml" -o "3h.pdf" -s "WITHHELD" -r
python number_attackers_uavs_over_gamma_script.py 6 -nsh -g 2 -w -s
```
