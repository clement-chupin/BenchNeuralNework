echo $0
echo $1

for i in {1..5}
do
for j in {1..5}
do
   echo $i $j
done
done


python main.py --mode manual --env 0 --policy 0 --feature 0