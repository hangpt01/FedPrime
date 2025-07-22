import argparse
import importlib

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', help='name of the benchmark;', type=str, default='mnist_classification')
    parser.add_argument('--dist', help='type of distribution;', type=int, default=0)
    parser.add_argument('--skew', help='the degree of niid;', type=float, default=0)
    parser.add_argument('--num_clients', help='the number of clients;', type=int, default=100)
    parser.add_argument('--seed', help='random seed;', type=int, default=0)
    parser.add_argument('--missing', help='missing-modality clients;', action='store_true', default=False)
    parser.add_argument('--missing_ratio_train', type=float, default=0.7)
    parser.add_argument('--missing_ratio_test', type=float, default=0.7)
    parser.add_argument('--missing_type_train', type=str, default='both')
    parser.add_argument('--missing_type_test', type=str, default='both')
    parser.add_argument('--both_ratio', type=float, default=0.5)
    parser.add_argument('--max_text_len', help='Max text len (the larger the more informative)', type=int, default=40)

    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

if __name__ == '__main__':
    option = read_option()
    print(option)
    TaskGen = getattr(importlib.import_module('.'.join(['benchmark', option['benchmark'], 'core'])), 'TaskGen')
    generator = TaskGen(
        dist_id = option['dist'],
        skewness = option['skew'],
        num_clients=option['num_clients'],
        seed = option['seed'],
        missing = option['missing'],
        missing_ratio_train = option['missing_ratio_train'],
        missing_ratio_test = option['missing_ratio_test'],
        missing_type_train = option['missing_type_train'],
        missing_type_test = option['missing_type_test'],
        both_ratio = option['both_ratio'],
        max_text_len = option['max_text_len']
    )
    print(generator.taskname)
    generator.run()
