# COL772 A2 - Shubham Mittal 2018EE10957
from runtrain import *

# parse arguments
def add_args(parser):
    parser.add_argument('--input_file', default='A2/inputfile.txt', type=str, help='path to input file (not necessarily csv)')
    parser.add_argument('--output_file', default='A2/predictions', type=str, help='path to output file')
    parser.add_argument('--model_dir', default='A2', type=str, help='path to directory containing saved model')
    # parser.add_argument('--TEST', type=str, help='path to TEST data')
    return parser

def main(args):
    df = pandas.read_csv(args.input_file, header=None, encoding='latin-1', delimiter='\n')
    try:
        df = preprocess(df[0])
    except:
        df = df[0]

    _vectorizer = pickle.load(open(os.path.join(args.model_dir, 'vectorizer'), 'rb'))
    x_test = _vectorizer.transform(df)

    model = pickle.load(open(os.path.join(args.model_dir, 'model'), 'rb'))
    y_pred = np.array(model.predict(x_test)).reshape((-1,1)) * 4

    np.savetxt(args.output_file, y_pred, fmt='%d')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COL 772 Assignment 2 | 2018EE10957')
    parser = add_args(parser)
    args = parser.parse_args()

    # if args.TEST != None:
    #     df = pandas.read_csv(args.TEST, header=None, encoding='latin-1')
    #     df[1] = preprocess(df[1])

    #     _vectorizer = pickle.load(open(os.path.join(args.model_dir, 'vectorizer'), 'rb'))
    #     x_test = _vectorizer.transform(df[1])
    #     model = pickle.load(open(os.path.join(args.model_dir, 'model'), 'rb'))

    #     df[0] = df[0].replace(4,1)
    #     print(model.score(x_test, df[0]))

    # else:
    main(args)
    