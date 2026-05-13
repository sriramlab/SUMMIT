python3 ../src/summit.py --geno ./small.bed \
                  --nvecs 100 \
                  --step_size 10000 \
                  --out ./small.single \
                  --dtype float64 \
                  --rand-samp 0.5 \
                  --target-xz-mem 16 \
                  --covar ./small.cov \
                  --num-threads 2
                  #
                  #--annot ./small.2bins_annot.txt \
