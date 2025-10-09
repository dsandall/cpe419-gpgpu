SRC=nbody.c
EXE=nbody
gcc -std=c99 -O3 -fopenmp -DSHMOO -o $EXE $SRC -lm
nvcc nbody.cu -o nbody_cuda

echo $EXE

K=1024
for i in {1..6}; do
  ./$EXE $K
  ./nbody_cuda $K
  K=$(($K * 2))
done
