#!/bin/sh

borzoi_satg_gene.py -o mini_borzois_v2/gasperini_miborzoi_k562_all -f 0,1 -c 0 --rc --untransform_old --track_scale 0.3 --track_transform 0.75 --clip_soft 384.0 -t mini_borzois_v2/k562_all/targets_k562_subset.txt mini_borzois_v2/k562_all/params_pred.json mini_borzois_v2/k562_all gasperini/crispr_genes.gtf

borzoi_satg_gene.py -o mini_borzois_v2/gasperini_miborzoi_k562_dnase_atac_rna -f 0,1 -c 0 --rc --untransform_old --track_scale 0.3 --track_transform 0.75 --clip_soft 384.0 -t mini_borzois_v2/k562_dnase_atac_rna/targets_k562_dnase_atac_rna_subset.txt mini_borzois_v2/k562_dnase_atac_rna/params_pred.json mini_borzois_v2/k562_dnase_atac_rna gasperini/crispr_genes.gtf

borzoi_satg_gene.py -o mini_borzois_v2/gasperini_miborzoi_k562_rna -f 0,1 -c 0 --rc --untransform_old --track_scale 0.3 --track_transform 0.75 --clip_soft 384.0 -t mini_borzois_v2/k562_rna/targets_k562_rna_subset.txt mini_borzois_v2/k562_rna/params_pred.json mini_borzois_v2/k562_rna gasperini/crispr_genes.gtf

borzoi_satg_gene.py -o mini_borzois_v2/gasperini_miborzoi_baseline -f 0,1 -c 0 --rc --untransform_old --track_scale 0.3 --track_transform 0.75 --clip_soft 384.0 -t mini_borzois_v2/baseline/targets_subset.txt mini_borzois_v2/baseline/params_pred.json mini_borzois_v2/baseline gasperini/crispr_genes.gtf

borzoi_satg_gene.py -o mini_borzois_v2/gasperini_miborzoi_human_all -f 0,1 -c 0 --rc --untransform_old --track_scale 0.3 --track_transform 0.75 --clip_soft 384.0 -t mini_borzois_v2/human_all/targets_subset.txt mini_borzois_v2/human_all/params_pred.json mini_borzois_v2/human_all gasperini/crispr_genes.gtf

borzoi_satg_gene.py -o mini_borzois_v2/gasperini_miborzoi_human_dnase_atac_rna -f 0,1 -c 0 --rc --untransform_old --track_scale 0.3 --track_transform 0.75 --clip_soft 384.0 -t mini_borzois_v2/human_dnase_atac_rna/targets_human_dnase_atac_rna_subset.txt mini_borzois_v2/human_dnase_atac_rna/params_pred.json mini_borzois_v2/human_dnase_atac_rna gasperini/crispr_genes.gtf

borzoi_satg_gene.py -o mini_borzois_v2/gasperini_miborzoi_multisp_dnase_atac_rna -f 0,1 -c 0 --rc --untransform_old --track_scale 0.3 --track_transform 0.75 --clip_soft 384.0 -t mini_borzois_v2/multispecies_dnase_atac_rna/targets_human_dnase_atac_rna_subset.txt mini_borzois_v2/multispecies_dnase_atac_rna/params_pred.json mini_borzois_v2/multispecies_dnase_atac_rna gasperini/crispr_genes.gtf

borzoi_satg_gene.py -o mini_borzois_v2/gasperini_miborzoi_multisp_rna -f 0,1 -c 0 --rc --untransform_old --track_scale 0.3 --track_transform 0.75 --clip_soft 384.0 -t mini_borzois_v2/multispecies_rna/targets_human_rna_subset.txt mini_borzois_v2/multispecies_rna/params_pred.json mini_borzois_v2/multispecies_rna gasperini/crispr_genes.gtf

borzoi_satg_gene.py -o mini_borzois_v2/gasperini_miborzoi_multisp_no_unet -f 0,1 -c 0 --rc --untransform_old --track_scale 0.3 --track_transform 0.75 --clip_soft 384.0 -t mini_borzois_v2/multispecies_no_unet/targets_subset.txt mini_borzois_v2/multispecies_no_unet/params_pred.json mini_borzois_v2/multispecies_no_unet gasperini/crispr_genes.gtf
