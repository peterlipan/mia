# test the optimizer
# python3 main.py --optimizer adam --lr 0.0001
# python3 main.py --optimizer adam --lr 0.001
# python3 main.py --optimizer adam --lr 0.01
# python3 main.py --optimizer adam --lr 0.1

python3 main.py --optimizer sgd --lr 0.05
python3 main.py --optimizer sgd --lr 0.01
python3 main.py --optimizer sgd --lr 0.005


python3 main.py --optimizer adamw --lr 0.0001
python3 main.py --optimizer adamw --lr 0.001
python3 main.py --optimizer adamw --lr 0.01
python3 main.py --optimizer adamw --lr 0.1
