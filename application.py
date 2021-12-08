import logging

import evaluation_mode
import exploration_mode
from createindex import *

logging.basicConfig(level=15)


@timed()
def main():
    stop = False
    exploration = exploration_mode.Exploration()

    # Main loop
    while not stop:
        raw_input = input(f"Allowed commands: evaluation {{bm25 or tfidf}} or exploration {{bm25 or tfidf}} {{query}}\nEnter command")

        mode_split = raw_input.split(" ", 3)

        if mode_split[0] == 'stop':
            stop = True

        elif mode_split[0] == 'evaluation':
            if len(mode_split) != 2:
                print(f"Wrong number of arguments : {len(mode_split)} expected 2 arguments, allowed commands are \n"
                      f"evaluation {{bm25 or tfidf}} or exploration {{bm25 or tfidf}} {{query}}")
            elif mode_split[1] == 'tfidf':
                print(f"Calculating results for {mode_split[1]}")
                tifu = evaluation_mode.tf_idf()
                evaluation_mode.printable_res(tifu, mode_split[1])
                print(f"Calculation ended. File printed to out")
            elif mode_split[1] == 'bm25':
                print(f"Calculating results for {mode_split[1]}")
                bm25 = evaluation_mode.bm25(1.25, 0.75)
                evaluation_mode.printable_res(bm25, mode_split[1])
                print(f"Calculation ended. File printed to out")
            elif mode_split[1] == 'tfidf_cs':
                print(f"Calculating results for {mode_split[1]}")
                tifu = evaluation_mode.tf_idf_cosine_similarity()
                evaluation_mode.printable_res(tifu, mode_split[1])
                print(f"Calculation ended. File printed to out")
            else:
                print(f"Wrong input, allowed commands are "
                      f"evaluation {{bm25 or tfidf or tfidf_cs}} or exploration {{bm25 or tfidf or tfidf_cs}} {{query}}")
        elif mode_split[0] == 'exploration':
            if len(mode_split) != 3:
                print(f"Wrong input, expected 3 arguments, allowed commands are"
                      f"evaluation {{bm25 or tfidf}} or exploration {{bm25 or tfidf}} {{query}}")

            elif mode_split[1] == 'tfidf':
                tifu = exploration.tf_idf_simple(mode_split[2])
                results = exploration.print_results(tifu)
                print(results)
                for result in results:
                    print(result)

            elif mode_split[1] == 'bm25':
                bm25 = exploration.bm25(mode_split[2], 1.25, 0.75)
                results = exploration.print_results(bm25)
                for result in results:
                    print(result)
            else:
                print(f"Wrong input, allowed commands are "
                      f"evaluation {{bm25 or tfidf}} or exploration {{bm25 or tfidf}} {{query}}")
        else:
            print(f"Wrong input only commands are: exploration, evaluation")


if __name__ == "__main__":
    main()
