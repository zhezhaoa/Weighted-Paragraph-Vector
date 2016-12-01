#run this script and observe the effectiveness of Naive Bayes weighting scheme
cd word2vec
gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -funroll-loops

cat ../data/full-train-pos.txt ../data/full-train-neg.txt ../data/test-pos.txt ../data/test-neg.txt > alldata.txt
awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < alldata.txt > alldata-id.txt
shuf alldata-id.txt > alldata-id-shuf.txt

echo @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
echo original paragraph vector
time ./word2vec -train alldata-id-shuf.txt -train-pos ../data/full-train-pos.txt -train-neg ../data/full-train-neg.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-3 -threads 4 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1 -nb 0
grep '_\*' vectors.txt | sed -e 's/_\*//' | sort -n > sentence_vectors.txt

head sentence_vectors.txt -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > full-train.txt
head sentence_vectors.txt -n 50000 | tail -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > test.txt

../liblinear-1.96/train -s 0 full-train.txt model.logreg
../liblinear-1.96/predict -b 1 test.txt model.logreg out.logreg

#################################
echo @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
echo paragraph vector with Naive bayes weighting scheme 
time ./word2vec -train alldata-id-shuf.txt -train-pos ../data/full-train-pos.txt -train-neg ../data/full-train-neg.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-3 -threads 4 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1 -nb 1
grep '_\*' vectors.txt | sed -e 's/_\*//' | sort -n > sentence_vectors.txt

head sentence_vectors.txt -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > full-train.txt
head sentence_vectors.txt -n 50000 | tail -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > test.txt

../liblinear-1.96/train -s 0 full-train.txt model.logreg
../liblinear-1.96/predict -b 1 test.txt model.logreg out.logreg

