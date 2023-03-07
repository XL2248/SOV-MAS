home_dir=/path/to/BBC/xlsum/XLSum_input
work_dir=/path/to/feature_extraction
sign=$1 #xlsum/XLSum_input/individual_img
image_path=$home_dir/$sign

modes="train val test"
for file in `ls $image_path`
do
    echo "lang", $file
#    image_dir=
    for mode in ${modes[@]} #$modes
    do
        image_dir=$image_path/$file/$mode
#        out_file=$out_path/$file/${mode}_img_roi.h5
        echo "begining", $image_dir, $out_file, $file, $mode
        python -u $work_dir/flickr30k_proposal_lyl.py --lang $file --split $mode --flickrroot $image_path
    done
done
echo "done!"
