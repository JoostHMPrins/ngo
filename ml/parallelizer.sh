export OMP_NUM_THREADS=1
for i in 0 1 2 3 4 5 6 7 8 9;
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python argparsetrainloopa.py --label $i &
done