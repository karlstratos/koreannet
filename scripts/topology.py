# Author: Karl Stratos (me@karlstratos.com)
import argparse
from numpy import array
from numpy import dot
from numpy import linalg

def read_embeddings(embedding_path, normalize):
    embedding = {}
    dim = 0

    with open(embedding_path, "r") as embedding_file:
        line_num = 0
        for line in embedding_file:
            tokens = line.split()
            if len(tokens) > 0:
                line_num += 1
                word = tokens[0]
                values = []
                for i in range(1, len(tokens)):
                    values.append(float(tokens[i]))

                # Ensure that the dimension matches.
                if dim:
                    assert(len(values) == dim)
                else:
                    dim = len(values)

                # Set the embedding, normalize the length if specified so.
                embedding[word] = array(values)
                if normalize:
                    embedding[word] /= linalg.norm(embedding[word])

    return embedding, dim

def display_nearest_neighbors(args):
    # Need to normalize the vector length for computing cosine similarity.
    embedding, dim = read_embeddings(args.embedding_path, args.dist == 'cos')
    print("Read {0} embeddings of dimension {1}".format(len(embedding), dim))

    while True:
        try:
            word = raw_input("Type a word (dist={0}): ".format(args.dist))
            if not word in embedding:
                print("There is no embedding for word \"{0}\"".format(word))
            else:
                neighbors = []
                for other_word in embedding:
                    if other_word == word: continue

                    u = embedding[word]
                    v = embedding[other_word]

                    if args.dist == 'l1':
                        distvalue = sum(map(abs, u - v))
                    elif args.dist == 'l2':
                        distvalue = linalg.norm(u - v)
                    elif args.dist == 'inf':
                        distvalue = max(map(abs, u - v))
                    elif args.dist == 'cos':
                        distvalue = dot(u, v)
                    else:
                        distvalue = 0.0
                        assert False
                    neighbors.append((distvalue, other_word))

                neighbors.sort(reverse=(args.dist == 'cos'))
                for i in range(min(args.num_neighbors, len(neighbors))):
                    distvalue, buddy = neighbors[i]
                    print("\t\t{0:.4f}\t\t{1}".format(distvalue, buddy))
        except (KeyboardInterrupt, EOFError):
            print()
            exit(0)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("embedding_path", type=str, help="path to word "
                           "embeddings file")
    argparser.add_argument("--num_neighbors", type=int, default=30,
                           help="number of nearest neighbors to display")
    argparser.add_argument("--dist", type=str, default='l2',
                           help="l1, l2, inf, cos")
    parsed_args = argparser.parse_args()
    display_nearest_neighbors(parsed_args)
