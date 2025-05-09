#!/bin/sh

borzoi_bench_sqtl_folds.py -r --span --no_untransform --vcf data/qtl_cat/sqtl -o sqtl_span -d 0 -e borzoi --rc -u --msl 4 --max_proc 12 -q rtx4070 --f_list 3 -c 4 --stats nDi -t targets_rna.txt params_pred.json saved_models
