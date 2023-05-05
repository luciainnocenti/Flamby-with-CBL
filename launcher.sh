export FLAMBY_DIR=${HOME}/FLamby
cd "$FLAMBY_DIR" || return
conda activate flamby

dataset=$1
rm -f results_"$dataset".csv; python "$FLAMBY_DIR"/flamby/benchmarks/fed_benchmark.py --GPU=1 --seed 42 --log -cfp "$FLAMBY_DIR"/flamby/config_"$dataset".json