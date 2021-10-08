data_dir=$1
out_dir=$2
question=$3
part=$4

if [ ${question}_${part} == "1_a" ] ; then
	python3 Q1/Q1a.py $data_dir $out_dir
elif [ ${question}_${part} == "1_b" ] ; then
	python3 Q1/Q1b.py $data_dir $out_dir
elif [ ${question}_${part} == "1_c" ] ; then
	python3 Q1/Q1c.py $data_dir $out_dir
elif [ ${question}_${part} == "1_d" ] ; then
	python3 Q1/Q1d.py $data_dir $out_dir
elif [ ${question}_${part} == "1_e" ] ; then
	python3 Q1/Q1e.py $data_dir $out_dir
elif [ ${question}_${part} == "2_a" ] ; then
	python3 Q2/Q2a.py $data_dir $out_dir
elif [ ${question}_${part} == "2_b" ] ; then
	python3 Q2/Q2b.py $data_dir $out_dir
elif [ ${question}_${part} == "2_c" ] ; then
	python3 Q2/Q2c.py $data_dir $out_dir
elif [ ${question}_${part} == "2_d" ] ; then
	python3 Q2/Q2d.py $data_dir $out_dir
elif [ ${question}_${part} == "3_a" ] ; then
	python3 Q3/Q3a.py $data_dir $out_dir
elif [ ${question}_${part} == "3_b" ] ; then
	python3 Q3/Q3b.py $data_dir $out_dir
elif [ ${question}_${part} == "4_a" ] ; then
	python3 Q4/Q4a.py $data_dir $out_dir
elif [ ${question}_${part} == "4_b" ] ; then
	python3 Q4/Q4b.py $data_dir $out_dir
elif [ ${question}_${part} == "4_c" ] ; then
	python3 Q4/Q4c.py $data_dir $out_dir
elif [ ${question}_${part} == "4_d" ] ; then
	python3 Q4/Q4d.py $data_dir $out_dir
elif [ ${question}_${part} == "4_e" ] ; then
	python3 Q4/Q4e.py $data_dir $out_dir
fi