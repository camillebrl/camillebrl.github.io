#!/bin/bash

# retrive .svg
sftp acps2 > /dev/null 2>&1 << myblock
get /LOCAL/ramdrop/github/mmrec_nusc/preprocess/vis_DTR_cache/DTR.svg

get /LOCAL/ramdrop/github/mmrec_nusc/preprocess/vis_RCS_cache/RCSHR.svg

get /LOCAL/ramdrop/github/milliPlace/postprocess/vis/figures/competing/PR-competing.svg

get /LOCAL/ramdrop/github/mmrec_nusc/postprocess/vis/figures/qualitative/sota_preds/qua-competing.svg

get /LOCAL/ramdrop/github/mmrec_nusc/postprocess/vis/figures/qualitative/openfig/*.png

exit
myblock
echo done!

# get /LOCAL/ramdrop/github/mmrec_nusc/_eval/ablation/recall_at_n-ablation.svg


# convert .svg to .pdf
rsvg-convert -f pdf -o DTR.pdf DTR.svg
rsvg-convert -f pdf -o RCSHR.pdf RCSHR.svg
rsvg-convert -f pdf -o PR-competing.pdf PR-competing.svg
rsvg-convert -f pdf -o qua-competing.pdf qua-competing.svg

# delete .svg
rm *.svg